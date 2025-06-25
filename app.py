from st_on_hover_tabs import on_hover_tabs
import datetime as dt
import streamlit as st
from src.routes import * 
from src.rag_chat_aux import *
import os 
from streamlit_folium import st_folium
import asyncio
from typing import Optional
import nest_asyncio
import torch
# import src.config as config


st.set_page_config(
    layout="wide"
)
# Establecer los límites del calendario: solo mayo 2025
st.logo('figures/logo-contract.png')
env_path = os.path.join(os.path.dirname(__file__), "src/.env")
config = dotenv_values(env_path)
api_key = st.secrets['ORS_API_KEY']

# st.header("Custom tab component for on-hover navigation bar")
st.markdown('<style>' + open('src/style.css').read() + '</style>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar container for chat
with st.sidebar:
    
    st.divider()
    tabs = on_hover_tabs(tabName=['Home', 'Map', 'AskValenBisi'], 
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
                        default_choice=0,
                        
                        key="1")
    st.divider()
    st.markdown(
    """
    <span style='color: #818181; font-weight:'>
    Built by A-squared:<br>
    - Andreu Bonet Pavia <br>
    - Anna Gil Moliner
 
    </span>
    """,
    unsafe_allow_html=True
)

if tabs == "Home":
    st.header("Welcome to the ValenBisi Route App", divider="gray")
    txt = """
        This application allows you to visualize ValenBisi availability predictions made with a LightGBM model for a specific date and time within May 2025, updated every fifteen minutes.
        Not only will you be able to visualize availability, but you can also plan routes between stations using our custom route algorithm (A*) or even ask questions about the regulations of the ValenBisi service to a chatbot.

    """
    st.write(txt)
    st.image('figures/valenbici_bici.jpeg')


    st.warning('It is required to have installed the model llama3.2 so that the chat bot can function properly.', icon="⚠️")
    st.subheader("How to install model llama3.2 for :blue-background[MacOS]:", divider="gray")
    st.markdown("""
        1. Open a terminal.
        2. Navigate to the directory where you want to install the model.
        3. Run the following commands:
        ```bash
        brew install ollama
        brew services start ollama
        ollama pull llama3.2
        ```
        4. After installation, you can use the model in your Python scripts.
        5. If you want to stop the Ollama service, you can run:
        ```bash
        brew services stop ollama
        ```
    """)
    st.subheader("How to install model llama3.2 for :blue-background[Windows]:", divider="gray")

    st.markdown("""

        1. Go to the official Ollama website:  [https://ollama.com/download](https://ollama.com/download)
        2. Download the installer for Windows and run it as administrator.
        3. Open PowerShell or Command Prompt and run the following command to download and run the model:
        ```bash
        ollama run llama3.2
        ```     
        4. The model **LLaMA3.2** will be automatically downloaded the first time you run it.
        5. To stop the model when you're not using it:
        ```bash
        ollama stop llama3.2
        ```
    """)
    st.subheader("How to install model llama3.2 for :blue-background[Linux]:", divider="gray")
    st.markdown("""
        1. Open a terminal.
        2. Install Ollama by running:
        ```bash
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull llama3.2
        ollama run llama3.2
        ```
        3. To stop the model:
        ```bash
        ollama stop llama3.2
        ```
    """)

    url = "https://ollama.com/library/llama3.2"
    st.info("For more information about the installation check [link](%s)" % url)

elif tabs == "Map":
    st.header("ValenBisi Stations Map", divider="gray")
    txt = """
        In this section, you can explore ValenBisi stations along with :gray-background[predicted availability] for a selected date and time in **May 2025**.
        Use the slider to choose the time of day and the calendar to select the specific date. Finally click the button to generate the map with the stations and their predicted availability.
        Additionally, you can plan :gray-background[routes] by selecting start and end stations and enabling the route algorithm option.

        REMEMBER: when chanching the date or time, you must click the button to generate the map again.
    """
    st.write(txt)
    for key in ["mapa_generado", "mapa_ruta", "ultimo_timestamp"]:
        if key not in st.session_state:
            st.session_state[key] = None if "mapa" in key or "timestamp" in key else False

    start_date = dt.date(2025, 5, 1)
    end_date = dt.date(2025, 5, 31)

    # Mostrar el calendario con esos límites
    selected_date = st.date_input(
        "**Select the day in which you are interested:**",
        value=start_date,
        min_value=start_date,
        max_value=end_date
    )


    # Crear un rango de horas para mayo de 2025 (simulado en cualquier fecha)
    hora_inicio = dt.time(0, 0)
    hora_fin = dt.time(23, 59)

    # Slider de tiempo
    hora_seleccionada = st.slider(
        "**Select a time in the day:**",
        min_value=hora_inicio,
        max_value=hora_fin,
        value=dt.time(8, 0),
        step=dt.timedelta(minutes=1)
    )

    # Combinar fecha y hora seleccionadas
    timestamp = dt.datetime.combine(selected_date, hora_seleccionada)
    T, D, W, P, mayo_merged = prepare_df()
    with st.container(height=600, border=True):
        
        left, right = st.columns([3, 1], border=True)
        #log(f"[INFO] Data prepared for May 2025: {mayo_merged.shape[0]} rows")


        with left:
        # Botón para generar el mapa
            
            if st.button("🗺️ All Station Generation"):
                m1 = create_station_map(mayo_merged, timestamp)
                st.session_state["mapa_generado"] = True
                st.session_state["ultimo_timestamp"] = timestamp

            # Mostrar el mapa si fue generado
            if st.session_state.get("mapa_generado", False):
                # Vuelve a generar el mapa solo para visualizarlo, pero no lo guardes
                m1 = create_station_map(mayo_merged, st.session_state["ultimo_timestamp"])
                st_folium(m1, width=750, height=450, returned_objects=[])
            
        with right:
            r = st.toggle("Route Algorithm")
            if r:
                start = st.text_input("Where do you want to start the bike journey?", value = 1)
                end = st.text_input("Where do you want to end the bike journey?", value = 2)

                start = int(start) if start.isdigit() else 1
                end = int(end) if end.isdigit() else 2
                if start in [168, 105, 146]:
                    st.warning("The start station is not available for routes. Please choose another station.", icon="⚠️")
                if end in [168, 105, 146]:
                    st.warning("The end station is not available for routes. Please choose another station.", icon="⚠️")

                route, info = a_star_distance(start, end, timestamp, T, D, W, P,mayo_merged, max_duration=30)
                st.write(f":blue-background[Route from station {start} to station {end}:]")
                for s,t in info.items():
                    st.write(f" - **Station {s}**: at {t[3].strftime('%H:%M')}")

    if r: 
        st.subheader("Route Map")
        text = "Click the button below to generate the route map between the selected stations. " \
               "The route will be displayed on the map with the stations and the path taken."
        st.markdown(text)

        # Botón para generar el mapa de la ruta
        # Verifica si la ruta fue calculada antes de intentar mostrarla
        # Si no hay ruta, no se puede generar el mapa
        # Si no hay info, no se puede generar el mapa
        # Si no hay mayo_merged, no se puede generar el mapa
        # Si no hay api_key, no se puede generar el mapa
        if st.button("🗺️ Route Map Generation"):
            m2 = map_route(route, mayo_merged, info, api_key)
            st.session_state["mapa_ruta"] = True
            st.session_state["ultimo_timestamp"] = timestamp


        # Mostrar el mapa si fue generado
        if st.session_state.get("mapa_ruta", False):
            # Vuelve a generar el mapa solo para visualizarlo, pero no lo guardes
            m2 = map_route(route, mayo_merged, info, api_key)
            if m2 is not None:
                st_folium(m2, width=1000, height= 500, returned_objects=[])


        # st.write(info)
        # st.write(route)

elif tabs == "AskValenBisi":
    col1, col2 = st.columns([5,1])  # Adjust column ratios as needed

    with col1:
        st.subheader("AskValenBisi", divider="gray")

    with col2:
        if st.button("Clean chat history"):
            st.session_state.messages = []
            if "chatbot_client" in st.session_state:
                del st.session_state.chatbot_client
            st.rerun()

    # st.subheader("AskValenBisi")

    # Sidebar UI
    # with st.sidebar:
    #     st.header("RAG-based Chatbot")
    #     st.markdown(
    #         """
    #         **Built by:**
    #         - A-squared
    #         """
    #     )
    #     st.divider()
    #     if st.button("Clean chat history"):
    #         st.session_state.messages = []
    #         if "chatbot_client" in st.session_state:
    #             del st.session_state.chatbot_client
    #         st.rerun()

    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "message": "Hi! I'm AskValenBici. You can ask me anything about Valenbici.",
            "timestamp": dt.datetime.now().strftime("%H:%M:%S")
        }]

    if "chatbot_client" not in st.session_state:
        with st.spinner("Initializing RAG model..."):
            st.session_state.chatbot_client = setup_chatbot()
            if st.session_state.chatbot_client:
                st.success("RAG model ready!")
            else:
                st.error("Failed to initialize RAG model")
                st.stop()

    chatbot_client = st.session_state.chatbot_client

    # Display chat history
    for msg in st.session_state.messages:
        role: str = msg["role"]
        content: str = msg.get("response_text") or msg.get("message")
        with st.chat_message(role):
            st.write(content)

    # Handle user input
    if user_input := st.chat_input("Write your question here..."):
        timestamp: str = dt.datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "message": user_input,
            "timestamp": timestamp
        })
        st.rerun()

    # Generate assistant response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_msg: str = st.session_state.messages[-1]["message"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    bot_result = chatbot_client.query_llama(user_msg)
                    bot_text: str = bot_result.get('response', 'Error: No response received')
                    log(f"[INFO] Generated response of length {len(bot_text)}")
                except Exception as e:
                    bot_text = f"Error generating response: {e}"
                    log(f"[ERROR] {bot_text}")
                st.write(bot_text)
        st.session_state.messages.append({
            "role": "assistant",
            "response_text": bot_text,
            "timestamp": dt.datetime.now().strftime("%H:%M:%S")
        })
        st.rerun()