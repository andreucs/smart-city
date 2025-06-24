import os
import asyncio
from typing import Optional
from datetime import datetime
import nest_asyncio
import streamlit as st
import torch

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