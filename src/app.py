from st_on_hover_tabs import on_hover_tabs
import datetime as dt
import streamlit as st
from routes import * 
from streamlit_folium import st_folium



st.set_page_config(layout="wide")
# Establecer los límites del calendario: solo mayo 2025


st.header("Custom tab component for on-hover navigation bar")
#st.markdown('<style>' + open('./src/style.css').read() + '</style>', unsafe_allow_html=True)


with st.sidebar:
        tabs = on_hover_tabs(tabName=['Home', 'Map', 'Chat'], 
                             iconName=['home', 'map', 'code'],
                             styles = {'navtab': {'background-color':'#111',
                                                  'color': '#818181',
                                                  'font-size': '18px',
                                                  'transition': '.3s',
                                                  'white-space': 'nowrap',
                                                  'text-transform': 'uppercase'},
                                       'tabStyle': {':hover :hover': {'color': 'red',
                                                                      'cursor': 'pointer'}},
                                       'tabStyle' : {'list-style-type': 'none',
                                                     'margin-bottom': '30px',
                                                     'padding-left': '30px'},
                                       'iconStyle':{'position':'fixed',
                                                    'left':'7.5px',
                                                    'text-align': 'left'},
                                       },
                             key="1")
        
# with st.sidebar:
#     tabs = on_hover_tabs(tabName=['Dashboard', 'Money', 'Economy'], 
#                          iconName=['dashboard', 'money', 'economy'], default_choice=0)

if tabs =='Home':
    st.title("Navigation Bar")
    st.write('Name of option is {}'.format(tabs))

elif tabs == 'Map':
    st.title("Map View")
    st.write('Name of option is {}'.format(tabs))
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
        step=dt.timedelta(minutes=15)
    )

    # Mostrar el resultado seleccionado
    st.write(f"Has seleccionado: {selected_date}")
    st.write(f"Hora seleccionada: {hora_seleccionada}")
    # Combinar fecha y hora seleccionadas
    timestamp = dt.datetime.combine(selected_date, hora_seleccionada)
    T, D, W, P, mayo_merged = prepare_df()
    #log(f"[INFO] Data prepared for May 2025: {mayo_merged.shape[0]} rows")
            
    # Botón para generar el mapa
    if st.button("🗺️ Prediction Map Generation"):
        try:
            m1 = crear_mapa_estaciones(mayo_merged, timestamp)
            st.session_state["mapa_generado"] = True
            st.session_state["ultimo_timestamp"] = timestamp
        except ValueError as e:
            st.warning(str(e))
            st.session_state["mapa_generado"] = False

    # Mostrar el mapa si fue generado
    if st.session_state.get("mapa_generado", False):
        # Vuelve a generar el mapa solo para visualizarlo, pero no lo guardes
        m1 = crear_mapa_estaciones(mayo_merged, st.session_state["ultimo_timestamp"])
        st_folium(m1, width=725, returned_objects=[])





elif tabs == 'Chat':
    st.title("Tom")
    st.write('Name of option is {}'.format(tabs))
