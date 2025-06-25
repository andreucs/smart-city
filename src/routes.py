import pandas as pd
from src.utils import *
try:
    from src.utils import *
    from src.ors_cycling import *
except:
    from utils import *
    from ors_cycling import *
import os
from heapq import heappush, heappop
from datetime import datetime, timedelta
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

    D = distances.to_dict(orient='index') 
    D = {
        ext_k: {int(inner_k): v for inner_k, v in inner_dict.items()}
        for ext_k, inner_dict in D.items()
    }


    T = times.to_dict(orient='index') 
    T = {
        ext_k: {int(inner_k): v for inner_k, v in inner_dict.items()}
        for ext_k, inner_dict in T.items()
    }

    W = times_w.to_dict(orient='index') 
    W = {
        ext_k: {int(inner_k): v for inner_k, v in inner_dict.items()}
        for ext_k, inner_dict in W.items()
    }

    
    mayo = pd.read_parquet("./data/may_with_predictions.parquet")

    # Filter the DataFrame to keep only relevant columns
    mayo_filtered = mayo[['timestamp','id', 'predicted']]
    mayo_merged = mayo_filtered.merge(df_bikes, on='id', how='left')

    # Calculate available spaces
    mayo_merged['available_spaces'] = mayo_merged['total_spaces'] - round(mayo_merged['predicted'])
    df_preds = mayo_merged.set_index(["id", "timestamp"])
    P = df_preds["available_spaces"].to_dict()

    return T, D, W, P, mayo_merged


def round_to_next_15min(dt):
    if dt.minute % 15 == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.replace(second=0, microsecond=0)
    
    minute = ((dt.minute // 15) + 1) * 15
    if minute == 60:
        dt += datetime.timedelta(hours=1)
        minute = 0
    return dt.replace(minute=minute, second=0, microsecond=0)


def round_to_next_15min(dt):
    # If the time is already a multiple of 15 minutes, return it as is
    if dt.minute % 15 == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.replace(second=0, microsecond=0)

    # Otherwise, round up to the next multiple of 15
    minute = ((dt.minute // 15) + 1) * 15
    if minute == 60:
        dt += timedelta(hours=1)
        minute = 0
    return dt.replace(minute=minute, second=0, microsecond=0)


def a_star_distance(start, goal, departure_datetime, T, D,W, P,mayo_merged, max_duration=30):
    if start not in T or goal not in T:
        return None, {}
    if start not in D or goal not in D:
        return None, {}
    if start not in W or goal not in W:
        return None, {}

    open_set = []
    heappush(open_set, (0, start, departure_datetime, [start]))
    visited = set()
    best_initial_time = 0  # Accumulated time at start
    info = {}

    while open_set:
        # Extract the node with the lowest f-score (estimated cost)
        f, current, arrival_time, path = heappop(open_set)

        # Round arrival time to the next 15-minute slot
        prediction_time = round_to_next_15min(arrival_time)

        if current == start:
            try:
                total_docks = mayo_merged.loc[mayo_merged['id'] == start, 'total_spaces'].values[0]
            except IndexError:
                log(f"🚫 Station {start} not found in the DataFrame.")
                return None, {}

            free_docks = P.get((start, prediction_time), 0)
            available_bikes = total_docks - free_docks

            if available_bikes < 1:
                log(f"🚫 No bikes at {start} at {prediction_time}. Searching for nearby walking alternative...")

                alternative = None
                best_walk_time = float('inf')

                for alt in W.get(start, {}):  # W = walking time matrix
                    walk_time = round(W[start][alt] / 60)  # convert seconds to minutes
                    if walk_time > max_duration:
                        continue

                    new_departure = departure_datetime + timedelta(minutes=walk_time)
                    new_pred_time = round_to_next_15min(new_departure)

                    try:
                        total_alt_docks = mayo_merged.loc[mayo_merged['id'] == alt, 'total_spaces'].values[0]
                    except IndexError:
                        continue

                    alt_free_docks = P.get((alt, new_pred_time), 0)
                    alt_available_bikes = total_alt_docks - alt_free_docks

                    if alt_available_bikes > 0 and walk_time < best_walk_time:
                        best_walk_time = walk_time
                        alternative = (alt, new_departure, new_pred_time)

                if alternative:
                    alt_id, alt_departure, alt_pred_time = alternative
            
                    best_initial_time = best_walk_time
                    info[start] = [free_docks, available_bikes, best_initial_time, departure_datetime, alt_pred_time]

                    heappush(open_set, (0, alt_id, alt_departure, [start, alt_id]))
                    continue
                else:
                    log("❌ No nearby walking station with available bikes.")
                    return None, {}

            else:
                log(f"🚴 Starting station: {start} (total: {total_docks}, bikes: {available_bikes}, free docks: {free_docks})")

        start = int(start)
        goal = int(goal)

        accumulated_time = (arrival_time - departure_datetime).total_seconds() / 60  # minutes

        free_docks = P.get((current, prediction_time), 0)
        total_docks = mayo_merged.loc[mayo_merged['id'] == current, 'total_spaces'].values[0]
        available_bikes = total_docks - free_docks

        info[current] = [free_docks, available_bikes, accumulated_time, arrival_time, prediction_time]

        if current == goal:
            if P.get((goal, prediction_time), 0) >= 1:
                return path, info
            else:
                log(f"❌ Destination {goal} has no docks at {prediction_time}. Searching for nearby alternatives...")

                alternative = None
                best_extra_time = float('inf')

                for alt in T.get(goal, {}):
                    extra_duration = T[goal][alt] / 60
                    if extra_duration > max_duration:
                        continue

                    alt_arrival = arrival_time + timedelta(minutes=extra_duration)
                    alt_arrival_pred = round_to_next_15min(alt_arrival)

                    if P.get((alt, alt_arrival_pred), 0) >= 1:
                        if extra_duration < best_extra_time:
                            best_extra_time = extra_duration
                            alternative = (alt, alt_arrival_pred)

                if alternative:
                    alt_id, arrival_pred = alternative
                    final_arrival = arrival_time + timedelta(seconds=round(best_extra_time * 60))
                    accumulated_time = (final_arrival - departure_datetime).total_seconds() / 60

                    free_docks = P.get((alt_id, prediction_time), 0)
                    total_docks = mayo_merged.loc[mayo_merged['id'] == alt_id, 'total_spaces'].values[0]
                    available_bikes = total_docks - free_docks
                    info[alt_id] = [free_docks, available_bikes, accumulated_time, final_arrival, arrival_pred]

                    return path + [alt_id], info  # ✅ End here

                else:
                    log(f"❌ No nearby alternative found with available docks")
                    return None, {}

        if current != start and P.get((current, prediction_time), 0) < 1 and current != goal:
            continue

        visited.add((current, prediction_time))  # Mark current station and time as visited

        for neighbor in T.get(current, {}):
            duration = T[current][neighbor]
            duration = round(duration / 60)  # Convert to minutes
            if duration > max_duration:
                continue

            new_arrival_time = arrival_time + timedelta(minutes=duration)
            new_pred_time = round_to_next_15min(new_arrival_time)

            if (neighbor, new_pred_time) in visited:
                continue

            if neighbor != goal and P.get((neighbor, new_pred_time), 0) < 1:
                continue

            # Heuristic: distance between neighbor and goal
            h = D.get(neighbor, {}).get(goal, float('inf'))
            g = (new_arrival_time - departure_datetime).total_seconds() / 60
            f_score = g + h

            heappush(open_set, (f_score, neighbor, new_arrival_time, path + [neighbor]))

    return None, 0  # No valid route found

def map_route(
    route: list,
    stations_df: 'pd.DataFrame',
    station_info: dict,
    api_key: str
) -> folium.Map:
    """
    Draws a bike route using ORS between stations on an interactive map.

    Args:
        route (list): List of station IDs representing the route.
        stations_df (pd.DataFrame): Must contain columns 'id', 'lat', 'lon'.
        station_info (dict): Contains info per station like available bikes, etc.
        api_key (str): OpenRouteService API key.

    Returns:
        folium.Map: Interactive map with the bike route.
    """

    if not route or stations_df.empty:
        log("⚠️ Route is empty or station data is missing.")
        return None

    start = route[0] if route else None
    end = route[-1] if route else None
    penultimate = route[-2] if len(route) > 1 else None

    client = openrouteservice.Client(key=api_key)
    coords = []

    for station_id in route:
        row = stations_df.loc[stations_df['id'] == station_id]
        if not row.empty:
            lat = row['lat'].values[0]
            lon = row['lon'].values[0]
            coords.append((lon, lat))  # ORS expects (lon, lat)
        else:
            log(f"⚠️ Station {station_id} not found in the DataFrame.")

    m = folium.Map(location=(coords[0][1], coords[0][0]), zoom_start=14)

    for i, (lon, lat) in enumerate(coords):
        available_docks, available_bikes, time_accum, actual_arrival, predicted_arrival = station_info[route[i]]
        address = stations_df.loc[stations_df['id'] == route[i], ['address']].values[0][0]

        # Custom marker and popup content
        if route[i] == start:
            color = 'cadetblue'
            border_color = 'teal'
            label = f"""
                <div style="width:400px; font-size:11px;">
                <h5>🚴 Start - Station {route[i]}</h5><br>
                <b>Address:</b> {address}<br>
                🕒 <b>Real arrival time:</b> {actual_arrival.strftime('%Y-%m-%d %H:%M')}<br>
                ⏱ <b>Elapsed time:</b> {time_accum:.1f} min<br>
                🚲 <b>Bikes at {predicted_arrival.strftime('%H:%M')}:</b> {available_bikes}<br>
                🅿️ <b>Spots at {predicted_arrival.strftime('%H:%M')}:</b> {available_docks}
                </div>
            """
        elif route[i] == end:
            color = 'cadetblue'
            border_color = 'teal'
            label = f"""
                <div style="width:400px; font-size:11px;">
                <h5>🚴 End - Station {route[i]}</h5><br>
                <b>Address:</b> {address}<br>
                🕒 <b>Real arrival time:</b> {actual_arrival.strftime('%Y-%m-%d %H:%M')}<br>
                ⏱ <b>Elapsed time:</b> {time_accum:.1f} min<br>
                🚲 <b>Bikes at {predicted_arrival.strftime('%H:%M')}:</b> {available_bikes}<br>
                🅿️ <b>Spots at {predicted_arrival.strftime('%H:%M')}:</b> {available_docks}
                </div>
            """
        else:
            color = 'CornflowerBlue'
            border_color = 'RoyalBlue'
            label = f"""
                <div style="width:400px; font-size:11px;">
                <h5>🚴 Bike Station {route[i]}</h5><br>
                <b>Address:</b> {address}<br>
                🕒 <b>Real arrival time:</b> {actual_arrival.strftime('%Y-%m-%d %H:%M')}<br>
                ⏱ <b>Elapsed time:</b> {time_accum:.1f} min<br>
                🚲 <b>Bikes at {predicted_arrival.strftime('%H:%M')}:</b> {available_bikes}<br>
                🅿️ <b>Spots at {predicted_arrival.strftime('%H:%M')}:</b> {available_docks}
                </div>
            """

        folium.Marker(
            location=(lat, lon),
            popup=label,
            tooltip=f"Station {i+1} / {len(route)}",
            icon=BeautifyIcon(
                icon_shape='marker',
                number=route[i],
                border_color=border_color,
                background_color=color,
                text_color='white'
            )
        ).add_to(m)

    # Draw real routes between consecutive station pairs
    for i in range(len(coords) - 1):
        current_station = route[i]
        next_station = route[i + 1]

        color = 'cadetblue'  # Default color

        # If first station has no bikes
        if current_station == start and station_info[current_station][1] == 0:
            color = 'yellow'

        # If penultimate station has no free docks
        if current_station == penultimate and station_info[current_station][0] == 0:
            color = 'salmon'

        try:
            segment = client.directions(
                coordinates=[coords[i], coords[i + 1]],
                profile='cycling-regular',
                format='geojson'
            )
        except Exception:
            st.warning("The API key is invalid or has expired.", icon="⚠️")
            return None

        folium.GeoJson(
            segment,
            name=f"Segment {i+1}",
            style_function=lambda x, col=color: {
                'color': col,
                'weight': 5,
                'opacity': 0.8,
                'dashArray': '5,5'  # Remove for solid lines
            }
        ).add_to(m)

    # Add a legend
    legend_html = '''
    <div style="
        position: fixed;
        top: 20px;
        right: 200px;
        width: 240px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:11px;
        padding: 10px;
        color: black;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <i class="fa fa-map-marker fa-2x" style="color:cadetblue"></i> Start or End Station<br>
        <i class="fa fa-map-marker fa-2x" style="color:CornflowerBlue"></i> Intermediate Station<br>
        <svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="cadetblue" stroke-width="3"/></svg> Cycling<br>
        <svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="yellow" stroke-width="3"/></svg> Walking<br>
        <svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="salmon" stroke-width="3"/></svg> Cycling (no parking)<br>
    </div>
    '''

    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def create_station_map(df: pd.DataFrame, 
                       timestamp_filter: pd.Timestamp) -> folium.Map:
    """
    Creates a map displaying bike stations with availability at a specific timestamp.

    Args:
        df (pd.DataFrame): Must contain 'timestamp' index and columns: 'id', 'lat', 'lon',
                           'address', 'available_spaces', 'predicted'.
        timestamp_filter (pd.Timestamp): The timestamp to filter station data by.

    Returns:
        folium.Map: An interactive map with station markers and availability info.
    """

    # Round timestamp to the next 15-minute interval
    timestamp_filter = round_to_next_15min(timestamp_filter)

    # Ensure the index is 'timestamp'
    if df.index.name != 'timestamp':
        df = df.set_index('timestamp')
        df = df.sort_index()

    # Try to filter by the exact timestamp
    try:
        filtered = df.loc[[timestamp_filter]]
    except KeyError:
        raise ValueError(f"No data found for timestamp {timestamp_filter}")

    if filtered.empty:
        raise ValueError(f"No data available at {timestamp_filter}")

    # Create a map centered on the average coordinates
    center_lat = filtered["lat"].mean()
    center_lon = filtered["lon"].mean()
    map_ = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Add markers for each station
    for row in filtered.itertuples():
        popup_html = f"""
        <div style="width:400px; font-size:11px;">
            <h6>🚴 Station {row.id}</h6><br>
            <b>Address:</b> {row.address}<br>
            🚲 <b>Available spaces at {timestamp_filter.strftime('%H:%M')}:</b> {row.available_spaces}<br>
            🅿️ <b>Predicted bikes at {timestamp_filter.strftime('%H:%M')}:</b> {round(row.predicted)}
        </div>
        """

        folium.Marker(
            location=[row.lat, row.lon],
            popup=popup_html,
            icon=BeautifyIcon(
                icon_shape='marker',
                number=row.id,
                border_color='cadetblue',
                background_color='teal',
                text_color='white'
            )
        ).add_to(map_)

    return map_




def main():
    #ejemplo de uso de la función a_star_con_distancia y de mapear_ruta
    T, D, W, P, mayo_merged = prepare_df()
    log("[INFO] Data prepared for routing")

    start = 1
    goal = 112
    salida_datetime = datetime(2025, 5, 1, 8, 0)  # Hora de salida
    max_tiempo = 30  # Tiempo máximo en minutos
    log(f"[INFO] Starting A* search from {start} to {goal} at {salida_datetime} with max time {max_tiempo} min")
    ruta, info = a_star_distance(start, goal, salida_datetime, T, D, W, P, mayo_merged, max_tiempo)
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
    
    map_ruta = map_route(ruta, mayo_merged, info, api_key)
    log("[OK] Map created")

if __name__ == '__main__':
    main()
