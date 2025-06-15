import os
import asyncio
from typing import Optional
from datetime import datetime
import nest_asyncio
import streamlit as st
import torch
# import src.config as config

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Fix PyTorch class path issue
if hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []


def log(message: str) -> None:
    """
    Log a timestamped message to the console with severity tags.

    Args:
        message (str): The message to log, including severity (e.g., [INFO], [OK], [WARNING], [ERROR]).
    """
    timestamp: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def setup_event_loop() -> None:
    """
    Configure asyncio event loop without DeprecationWarning and apply nest_asyncio.
    """
    try:
        # Avoid DeprecationWarning if no loop is running
        asyncio.get_running_loop()
        log("[OK] Retrieved existing asyncio event loop")
    except RuntimeError:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        log("[INFO] Created and set new asyncio event loop")


# Initialize event loop
try:
    setup_event_loop()
    nest_asyncio.apply()
    log("[OK] nest_asyncio applied successfully")
except Exception as e:
    log(f"[ERROR] Error during event loop setup: {e}")
    st.error(f"Error setting up event loop: {e}")

# Import the chatbot class
try:
    from src.ChatBot import ChatBot
    log("[OK] Imported Llama3.2 successfully")
except ImportError as e:
    log(f"[ERROR] Error importing Llama3.2: {e}")
    st.error(f"Error importing ChatBot: {e}")
    st.stop()

@st.cache_resource
def setup_chatbot() -> Optional[ChatBot]:
    """
    Initialize the Llama3.2 chatbot client.

    Returns:
        Optional[Llama3.2]: Chatbot instance or None on failure.
    """
    try:
        setup_event_loop()
        bot = ChatBot()
        log("[OK] RAG model initialized successfully")
        return bot
    except Exception as e:
        log(f"[ERROR] Error initializing RAG model: {e}")
        st.error(f"Error initializing RAG model: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="RAG-based Chatbot",
    layout="wide",
)

# Sidebar UI
with st.sidebar:
    st.header("RAG-based Chatbot")
    st.markdown(
        """
        **Built by:**
        - A-squared
        """
    )
    st.divider()
    if st.button("Clean chat history"):
        st.session_state.messages = []
        if "chatbot_client" in st.session_state:
            del st.session_state.chatbot_client
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "message": "Hi! I'm AskValenBici. You can ask me anything about Valenbici.",
        "timestamp": datetime.now().strftime("%H:%M:%S")
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
    timestamp: str = datetime.now().strftime("%H:%M:%S")
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
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    st.rerun()