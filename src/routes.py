import pandas as pd
from utils import *
import os
from ors_cycling import *
from heapq import heappush, heappop

import datetime
import folium
import openrouteservice
from openrouteservice import convert
from folium.plugins import BeautifyIcon
import streamlit as st
@st.cache_data
def prepare_df():
    """
    1. Load the dataset and distance/duration matrices.
    2. Convert the distance and duration matrices to dictionaries of dictionaries.
    3. Convert the timestamp column to datetime if not already in that format.
    4. Filter the dataset for entries in May 2025.
    5. Load the retrained model and prepare features for inference.
    6. Predict the number of bikes available using the model.
    7. Return the modified DataFrame with predictions.

    Returns:
        pd.DataFrame: A DataFrame containing the timestamp, station ID, predicted bikes available,
                      and available spaces at each station.
    """
    df = pd.read_csv(f"./data/dataset.csv")
    distances = pd.read_csv(f"./data/distance_matrix.csv")
    times = pd.read_csv(f"./data/duration_matrix.csv")
    times_w = pd.read_csv(f"./data/duration_w_matrix.csv")
    df_bikes = pd.read_csv(f"./data/bike_stations.csv")

    distances.set_index('id', inplace=True)
    times.set_index('id', inplace=True)
    times_w.set_index('id', inplace=True)

    # Conversión a diccionario de diccionarios
    D = distances.to_dict(orient='index') #diccionario de distancias
    # Convertimos todas las claves internas a int
    D = {
        ext_k: {int(inner_k): v for inner_k, v in inner_dict.items()}
        for ext_k, inner_dict in D.items()
    }


    T = times.to_dict(orient='index') #diccionario de tiempos
    T = {
        ext_k: {int(inner_k): v for inner_k, v in inner_dict.items()}
        for ext_k, inner_dict in T.items()
    }

    W = times_w.to_dict(orient='index') #diccionario de tiempos
    W = {
        ext_k: {int(inner_k): v for inner_k, v in inner_dict.items()}
        for ext_k, inner_dict in W.items()
    }

    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    is_may_2025 = (
            (df['timestamp'].dt.year  == 2025) &
            (df['timestamp'].dt.month == 5)
    )
    mayo = df.loc[is_may_2025].copy()
    

    # Load the retrained model
    # model_path = os.path.join('model', 'retrained_model.pkl')
    model = joblib.load(f"./model/retrained_model.pkl")

    # Prepare features and run inference
    TARGET = 'bikes_available'
    feature_cols = [c for c in mayo.columns if c not in ('timestamp', TARGET)]
    X_test = mayo[feature_cols]
    mayo['predicted'] = model.predict(X_test)

    # Filter the DataFrame to keep only relevant columns
    mayo_filtered = mayo[['timestamp','id', 'predicted']]
    mayo_merged = mayo_filtered.merge(df_bikes, on='id', how='left')

    # Calculate available spaces
    mayo_merged['available_spaces'] = mayo_merged['total_spaces'] - round(mayo_merged['predicted'])
    df_preds = mayo_merged.set_index(["id", "timestamp"])
    P = df_preds["available_spaces"].to_dict()

    return T, D, W, P, mayo_merged


def round_to_next_15min(dt):
    # Si ya es múltiplo de 15, no redondeamos
    if dt.minute % 15 == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.replace(second=0, microsecond=0)
    
    # Si no lo es, redondeamos hacia el siguiente múltiplo de 15
    minute = ((dt.minute // 15) + 1) * 15
    if minute == 60:
        dt += timedelta(hours=1)
        minute = 0
    return dt.replace(minute=minute, second=0, microsecond=0)


def a_star_con_distancia(start, goal, salida_datetime, T, D, W, P,mayo_merged, max_tiempo=30):
    
    if start not in T or goal not in T:
        # log (f"🚫 Estación {start} o {goal} no está en los datos de tiempos en bici (T)")
        return None,{}
    if (start not in D) or (goal not in D):
        # log (f"🚫 Estación {start} o {goal} no está en los datos de distancias (D)")
        return None,{}
    if (start not in W) or (goal not in W):
        # log (f"🚫 Estación {start} o {goal} no está en los datos de tiempos andando(W)")
        return None,{}
    
    # if not any(k[0] == start for k in P.keys()) or not any(k[0] == goal for k in P.keys()):
    #     raise ValueError(f"🚫 No hay predicciones disponibles para start o goal")
    open_set = []
    heappush(open_set, (0, start, salida_datetime, [start]))
    visited = set()
    info = {}
    
    
    while open_set:
        #sacamos el nodo con manor f (sería como la cota que calculamos)
        f, actual, tiempo_llegada, camino = heappop(open_set)

        # redondeamos el tiempo de llegada al siguiente múltiplo de 15 minutos
        tiempo_pred = round_to_next_15min(tiempo_llegada)

        
        if actual == start:
            try:
                total = mayo_merged.loc[mayo_merged['id'] == start, 'total_spaces'].values[0]
            except IndexError:
                # log(f"🚫 Estación {start} no está en el DataFrame.")
                return None,{}

            huecos = P.get((start, tiempo_pred), 0)
            bicis = total - huecos

            if bicis < 1:
                # log(f"🚫 No hay bicicletas en {start} a las {tiempo_pred}. Buscando estación alternativa andando...")

                alternativa = None
                mejor_tiempo = float('inf')

                for alterna in W.get(start, {}):  # W = matriz de tiempos andando
                    tiempo_a_pie = round(W[start][alterna] / 60)  # segundos a minutos
                    if tiempo_a_pie > max_tiempo:
                        continue

                    nueva_hora_salida = salida_datetime + timedelta(minutes=tiempo_a_pie)
                    nueva_hora_pred = round_to_next_15min(nueva_hora_salida)

                    try:
                        total_alt = mayo_merged.loc[mayo_merged['id'] == alterna, 'total_spaces'].values[0]
                    except IndexError:
                        continue

                    huecos_alt = P.get((alterna, nueva_hora_pred), 0)
                    bicis_alt = total_alt - huecos_alt

                    if bicis_alt > 0 and tiempo_a_pie < mejor_tiempo:
                        mejor_tiempo = tiempo_a_pie
                        alternativa = (alterna, nueva_hora_salida, nueva_hora_pred)

                if alternativa:
                    alt_id, alt_salida, alt_pred = alternativa
                    # log(f"✅ Nueva estación alternativa para salir: {alt_id}")
                    # log(f"   🚶 Tiempo andando desde {start}: {mejor_tiempo:.1f} min")
                    # log(f"   🕒 Hora de salida estimada desde alternativa: {alt_pred}")
                    t_mejor_ini = mejor_tiempo
                    huecos_disp = P.get((start, tiempo_pred), 0)
                    info[start] = [
                        huecos,
                        bicis,
                        t_mejor_ini,
                        salida_datetime,
                        alt_pred
                    ]

                    heappush(open_set, (0, alt_id, alt_salida, [start, alt_id]))
                    continue
                else:
                    # log("❌ No hay estaciones caminando cercanas con bicis disponibles.")
                    return None,{}

            else:
                log(f"🚴 Estación inicial: {start} (total: {total}, bicis: {bicis}, huecos: {huecos})")
        
        start = int(start)
        goal = int(goal)

        t_acum = (tiempo_llegada - salida_datetime).total_seconds() / 60 # tiempo acumulado en minutos

        huecos_disp = P.get((actual, tiempo_pred), 0)
        total = mayo_merged.loc[mayo_merged['id'] == actual, 'total_spaces'].values[0]
        b = total - huecos_disp

        info[actual] = [
            huecos_disp,
            b,
            t_acum,
            tiempo_llegada,
            tiempo_pred
        ]
          
        # log(f"\n🚴 Visitando: {actual}")
        # log(f"  ⏱ Llegada real: {tiempo_llegada} → redondeado a: {tiempo_pred}")
        # log(f"  📦 Huecos disponibles: {P.get((actual, tiempo_pred), 0)}")
        # log(f"  🧭 Ruta parcial: {camino}")
        # log(f"  ⏳ Tiempo acumulado: {t_acum } min")
        if actual == goal:

            if P.get((goal, tiempo_pred), 0) >= 1:
                return camino, info

            else:
                # log(f"[INFO] ❌ Estación destino {goal} sin huecos en {tiempo_pred}. Buscando única alternativa...")

                alternativa = None
                mejor_tiempo = float('inf')

                for alterna in T.get(goal, {}):
                    duracion_extra = T[goal][alterna] / 60  # convertir a minutos
                    if duracion_extra > max_tiempo:
                        continue

                    llegada_alterna = tiempo_llegada + timedelta(minutes=duracion_extra)
                    llegada_alterna_pred = round_to_next_15min(llegada_alterna)

                    if P.get((alterna, llegada_alterna_pred), 0) >= 1:
                        if duracion_extra < mejor_tiempo:
                            mejor_tiempo = duracion_extra
                            alternativa = (alterna, llegada_alterna_pred)

                if alternativa:
                    alterna_id, llegada_pred = alternativa
                    otra = tiempo_llegada + timedelta(seconds=round(mejor_tiempo * 60))
                    t_acum = (otra - salida_datetime).total_seconds() / 60
                    #t_acum += mejor_tiempo
                    
                    
                    # log(f"✅ Usamos alternativa final: {alterna_id}")
                    # log(f"   ⏱ Tiempo adicional desde {goal}: {mejor_tiempo:.1f} min")
                    # log(f"   🧭 Llegada real: {otra} → redondeado a: {llegada_pred}")
                    # log(f"   ⌛ Tiempo total desde inicio: {t_acum:.1f} min")
                    huecos_disp = P.get((alterna_id, tiempo_pred), 0)
                    total = mayo_merged.loc[mayo_merged['id'] == alterna_id, 'total_spaces'].values[0]
                    b = total - huecos_disp
                    info[alterna_id] = [
                        huecos_disp,
                        b,
                        t_acum,
                        otra,
                        llegada_pred
                    ]

                    return camino + [alterna_id], info # ✅ finalizamos aquí, no exploramos más



                else:
                    # log(f"❌ No se encontró alternativa con hueco para aparcar")
                    return None,{}

                
        # Si no es el destino, añadimos la estación actual al conjunto de visitados
        if actual != start and P.get((actual, tiempo_pred), 0) < 1 and actual != goal:
            continue
        

        visited.add((actual, tiempo_pred)) #añadimos la estacion que acabamos de visitar y el tiempo predicho de llegada
        #log(f"Open set size: {len(open_set)}")  # Debugging line to check open_set size
        #log(f"Visited size: {len(visited)}")  # Debugging line to check visited size
        
        for vecino in T.get(actual, {}): #buscamos los vecinos de la estacion actual
             # Debugging line to check neighbors
            duracion = T[actual][vecino] #nos quedamos con el tiempo desde la estacion actual al vecino
            duracion = round(duracion / 60)  # Convertimos a minutos
            #log(f"Checking neighbor: {vecino}, duration: {duracion} minutes")  # Debugging line to check neighbor and duration
            if duracion > max_tiempo: #si el tiempo de duración es mayor que el máximo permitido, lo saltamos
                continue # saltamos el resto del bucle

            nuevo_tiempo = tiempo_llegada + timedelta(minutes=duracion) # calculamos el nuevo tiempo de llegada para el vecino, usamos timedelta porque la duration matrix está en segundos
            nuevo_tiempo_pred = round_to_next_15min(nuevo_tiempo) #redondeamos de nuevo a lbloque de 15 min
            
            if (vecino, nuevo_tiempo_pred) in visited: #comporbamos que no hemos visitado ya el vecino en ese tiempo predicho
                continue

            if vecino != goal and P.get((vecino, nuevo_tiempo_pred), 0) < 1:
                continue

            # Heurística: distancia desde vecino al destino, que sería la cota optimista que usamos 
            h = D.get(vecino, {}).get(goal, float('inf'))  # km entre vecino y destino
            g = (nuevo_tiempo - salida_datetime).total_seconds() / 60  # tiempo acumulado en minutos
            f_score = g + h  # puedes ponderar h si quieres convertir a tiempo estimado

            heappush(open_set, (f_score, vecino, nuevo_tiempo, camino + [vecino]))
        

    return None,0  # no hay ruta válida

def mapear_ruta(ruta: list, 
                df_estaciones: 'pd.DataFrame', 
                info: dict,
                api_key: str) -> folium.Map:
    """
    Dibuja la ruta en bici usando ORS entre estaciones sobre un mapa interactivo.

    Args:
        ruta (list): Lista de IDs de estaciones por las que pasa la ruta.
        df_estaciones (pd.DataFrame): Debe tener columnas 'id', 'lat', 'lon'.
        api_key (str): Clave de OpenRouteService.

    Returns:
        folium.Map: Mapa interactivo con la ruta ciclista.
    """
    if not ruta or df_estaciones.empty:
        print("⚠️ Ruta vacía o sin datos.")
        return None 
    start = ruta[0] if ruta else None
    last = ruta[-1] if ruta else None
    last_l = ruta[-2] if len(ruta) > 1 else None

    

    client = openrouteservice.Client(key=api_key)
    coords = []

    for est_id in ruta:
        fila = df_estaciones.loc[df_estaciones['id'] == est_id]
        if not fila.empty:
            lat = fila['lat'].values[0]
            lon = fila['lon'].values[0]
            coords.append((lon, lat))  # ORS espera (lon, lat)
        else:
            log(f"⚠️ Estación {est_id} no encontrada en el DataFrame.")

    m = folium.Map(location=(coords[0][1], coords[0][0]), zoom_start=14)

    for i, (lon, lat) in enumerate(coords):
        huecos, bicis, t_acum, llegada_real, llegada_pred = info[ruta[i]]
        ad = df_estaciones.loc[df_estaciones['id'] == ruta[i], ['address']].values[0][0]
        if ruta[i] == start:    
            c = 'cadetblue' 
            cb = 'teal'       
            label = f"""
                <div style="width:400px;">
                <h2>🚴 Start - Station {ruta[i]}</h2><br>
                <b>Address:</b> {ad}<br>
                🕒 <b>Real arrival hour:</b> {llegada_real.strftime('%Y-%m-%d %H:%M')}<br>
                ⏱ <b>Time:</b> {t_acum:.1f} min<br>
                🚲 <b>Available bikes at {llegada_pred.strftime('%H:%M')}:</b> {bicis}<br>
                🅿️ <b>Available spots at {llegada_pred.strftime('%H:%M')}:</b> {huecos}
                </div>
                """
        elif ruta[i] == last:
            c = 'cadetblue'
            cb = 'teal'  
            label = f"""
                <div style="width:400px;">
                <h2>🚴 End - Station {ruta[i]}</h2><br>
                <b>Address:</b> {ad}<br>
                🕒 <b>Real arrival hour:</b> {llegada_real.strftime('%Y-%m-%d %H:%M')}<br>
                ⏱ <b>Time:</b> {t_acum:.1f} min<br>
                🚲 <b>Available bikes at {llegada_pred.strftime('%H:%M')}:</b> {bicis}<br>
                🅿️ <b>Available spots at {llegada_pred.strftime('%H:%M')}:</b> {huecos}
                </div>
                """
    
        else:
            c = 'CornflowerBlue'
            cb = 'RoyalBlue'
            label = f"""
                <div style="width:400px;">
                <h2>🚴 Bike Change Station {ruta[i]}</h2><br>
                <b>Address:</b> {ad}<br>
                🕒 <b>Real arrival hour:</b> {llegada_real.strftime('%Y-%m-%d %H:%M')}<br>
                ⏱ <b>Time:</b> {t_acum:.1f} min<br>
                🚲 <b>Available bikes at {llegada_pred.strftime('%H:%M')}:</b> {bicis}<br>
                🅿️ <b>Available spots at {llegada_pred.strftime('%H:%M')}:</b> {huecos}
                </div>
                """
        
        folium.Marker(
            location=(lat, lon),
            popup=label,
            tooltip=f"Station {i+1} / {len(ruta)}",
            icon=BeautifyIcon(
                icon_shape='marker',
                number=ruta[i],
                border_color=cb,
                background_color=c,
                text_color='white'
            )
        ).add_to(m)


    # Añadir rutas reales entre pares consecutivos
        for i in range(len(coords) - 1):
            est_actual = ruta[i]
            est_siguiente = ruta[i + 1]

            # Valor por defecto (color azul)
            color = 'cadetblue'

            # Tramo inicial: sin bicis
            if est_actual == start and info[est_actual][1] == 0:
                color = 'mediumseagreen'

            # Tramo final: sin huecos en penúltima
            if est_actual == last_l and info[est_actual][0] == 0:
                color = 'salmon'

            segmento = client.directions(
                coordinates=[coords[i], coords[i + 1]],
                profile='cycling-regular',
                format='geojson'
            )

            

            folium.GeoJson(
                segmento,
                name=f"Tramo {i+1}",
                style_function=lambda x, col=color: {
                    'color': col,
                    'weight': 5,
                    'opacity': 0.8,
                    'dashArray': '5,5'  # Puedes quitar esta línea si no quieres línea discontinua
                }
            ).add_to(m)

    legend_html = '''
    <div style="
        position: fixed;
        top: 20px;
        right: 20px;
        width: 250px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:11px;
        padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        ">
        <i class="fa fa-map-marker fa-2x" style="color:cadetblue"></i> Start or Finish Station<br>
        <i class="fa fa-map-marker fa-2x" style="color:CornflowerBlue"></i> Bike Change Station <br>
        <svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="cadetblue" stroke-width="3"/></svg> Path Cycling <br>
        <svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="mediumseagreen" stroke-width="3"/></svg> Path Walking <br>
        <svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="salmon" stroke-width="3"/></svg> Path Cycling (search of bike parking) <br>
        
    </div>
    '''


    m.get_root().html.add_child(folium.Element(legend_html))
       

    return m

import pandas as pd
import folium
from folium.plugins import BeautifyIcon
def crear_mapa_estaciones(df: pd.DataFrame, timestamp_filtro: pd.Timestamp) -> folium.Map:
    # Asegurarse de que el índice es timestamp para mejorar rendimiento
    if df.index.name != 'timestamp':
        df = df.set_index('timestamp')
        df = df.sort_index()

    # Intentar acceder directamente al timestamp (muy eficiente si hay match exacto)
    try:
        filtrado = df.loc[[timestamp_filtro]]
    except KeyError:
        raise ValueError(f"No hay datos exactos para {timestamp_filtro}")

    if filtrado.empty:
        raise ValueError(f"No hay datos para {timestamp_filtro}")

    # Crear el mapa centrado en la media de coordenadas
    center_lat = filtrado["lat"].mean()
    center_lon = filtrado["lon"].mean()
    mapa = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Iterar eficientemente
    for row in filtrado.itertuples():
        popup_text = f"""
        <div style="width:300px;">
        <b>Address:</b> {row.address}<br>
        🚲 <b>Available bike spaces at {row.Index.strftime('%H:%M')}:</b> {row.available_spaces}<br>
        🅿️ <b>Predicted bikes available at {row.Index.strftime('%H:%M')}:</b> {round(row.predicted)}
        </div>
        """
        folium.Marker(
            location=[row.lat, row.lon],
            popup=popup_text,
            icon=BeautifyIcon(
                icon_shape='marker',
                number=row.id,
                border_color='cadetblue',
                background_color='teal',
                text_color='white'
            )
        ).add_to(mapa)

    return mapa



def main():
    T, D, W, P, mayo_merged = prepare_df()
    log("[INFO] Data prepared for routing")

    start = 1
    goal = 112
    salida_datetime = datetime(2025, 5, 1, 8, 0)  # Hora de salida
    max_tiempo = 30  # Tiempo máximo en minutos
    log(f"[INFO] Starting A* search from {start} to {goal} at {salida_datetime} with max time {max_tiempo} min")
    ruta, info = a_star_con_distancia(start, goal, salida_datetime, T, D, W, P, mayo_merged, max_tiempo)
    if ruta is None:
        log("[FAIL] No valid route found")
        return
    log(f"[OK] Route found: {ruta}")
    log("[INFO] Mapping route on Folium map")

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    config = dotenv_values(env_path)
    api_key = config.get("ORS_API_KEY")
    if not api_key:
        log(f"[FAIL] ORS_API_KEY not found in {env_path}")

        return
    
    map_ruta = mapear_ruta(ruta, mayo_merged, info, api_key)
    log("[OK] Map created")

if __name__ == '__main__':
    main()
