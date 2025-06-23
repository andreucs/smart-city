from st_on_hover_tabs import on_hover_tabs
import datetime as dt
import streamlit as st
from routes import * 
import os 
from streamlit_folium import st_folium



st.set_page_config(layout="wide")
# Establecer los límites del calendario: solo mayo 2025

env_path = os.path.join(os.path.dirname(__file__), ".env")
config = dotenv_values(env_path)
api_key = config.get("ORS_API_KEY")

# st.header("Custom tab component for on-hover navigation bar")
#st.markdown('<style>' + open('./src/style.css').read() + '</style>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
with st.sidebar:
    st.title("Echo Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial primero
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 👉 El input debe ir al final del bloque, para que se renderice debajo del historial
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = f"Echo: {prompt}"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


            # tabs = on_hover_tabs(tabName=['Home', 'Map', 'Chat'], 
        #                      iconName=['home', 'map', 'code'],
        #                      styles = {'navtab': {'background-color':'#111',
        #                                           'color': '#818181',
        #                                           'font-size': '18px',
        #                                           'transition': '.3s',
        #                                           'white-space': 'nowrap',
        #                                           'text-transform': 'uppercase'},
        #                                'tabStyle': {':hover :hover': {'color': 'red',
        #                                                               'cursor': 'pointer'}},
        #                                'tabStyle' : {'list-style-type': 'none',
        #                                              'margin-bottom': '30px',
        #                                              'padding-left': '30px'},
        #                                'iconStyle':{'position':'fixed',
        #                                             'left':'7.5px',
        #                                             'text-align': 'left'},
        #                                },
        #                      key="1")
        
# with st.sidebar:
#     tabs = on_hover_tabs(tabName=['Dashboard', 'Money', 'Economy'], 
#                          iconName=['dashboard', 'money', 'economy'], default_choice=0)



for key in ["mapa_generado", "mapa_ruta", "ultimo_timestamp"]:
    if key not in st.session_state:
        st.session_state[key] = None if "mapa" in key or "timestamp" in key else False

st.title("Map View")
start_date = dt.date(2025, 5, 1)
end_date = dt.date(2025, 5, 31)

# Mostrar el calendario con esos límites
selected_date = st.date_input(
    "Select the day in which you are interested:",
    value=start_date,
    min_value=start_date,
    max_value=end_date
)


# Crear un rango de horas para mayo de 2025 (simulado en cualquier fecha)
hora_inicio = dt.time(0, 0)
hora_fin = dt.time(23, 59)

# Slider de tiempo
hora_seleccionada = st.slider(
    "Select a time in the day:",
    min_value=hora_inicio,
    max_value=hora_fin,
    value=dt.time(8, 0),
    step=dt.timedelta(minutes=1)
)

# Combinar fecha y hora seleccionadas
timestamp = dt.datetime.combine(selected_date, hora_seleccionada)
T, D, W, P, mayo_merged = prepare_df()
left, _, _, right = st.columns(4)
#log(f"[INFO] Data prepared for May 2025: {mayo_merged.shape[0]} rows")
r = left.toggle("Route Algorithm")


if not r:
# Botón para generar el mapa
    if right.button("🗺️ Prediction Map Generation"):
        m1 = crear_mapa_estaciones(mayo_merged, timestamp)
        st.session_state["mapa_generado"] = True
        st.session_state["ultimo_timestamp"] = timestamp

    # Mostrar el mapa si fue generado
    if st.session_state.get("mapa_generado", False):
        # Vuelve a generar el mapa solo para visualizarlo, pero no lo guardes
        m1 = crear_mapa_estaciones(mayo_merged, st.session_state["ultimo_timestamp"])
        st_folium(m1, width=725, returned_objects=[])
elif r:
    start = st.text_input("Where do you want to start the bike journey?", value = 1)
    end = st.text_input("Where do you want to end the bike journey?", value = 2)

    start = int(start) if start.isdigit() else 1
    end = int(end) if end.isdigit() else 2

    route, info = a_star_con_distancia(start, end, timestamp, T, D, W, P,mayo_merged, max_tiempo=30)
    st.write(f":blue-background[Route from station {start} to station {end}:]")
    for s,t in info.items():
        st.write(f" - **Station {s}**: estimated time at {t[3]}")
        

    # st.write(info)
    # st.write(route)


    if right.button("🗺️ Route Map Generation"):
        m2 = mapear_ruta(route, mayo_merged, info, api_key)
        st.session_state["mapa_ruta"] = True
        st.session_state["ultimo_timestamp"] = timestamp
    

    # Mostrar el mapa si fue generado
    if st.session_state.get("mapa_ruta", False):
        # Vuelve a generar el mapa solo para visualizarlo, pero no lo guardes
        m2 = mapear_ruta(route, mayo_merged, info, api_key)
        st_folium(m2, width=725, returned_objects=[])



