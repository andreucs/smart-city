import os
import asyncio
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import nest_asyncio
import torch
import faiss
import src.config as config #este tenia src.config

from langchain_huggingface import HuggingFaceEmbeddings
from src.vectorstore_builder import connect_faiss #este tenia src.vectorstore_builder
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Set static directory to script location
os.environ["STATIC_DIRECTORY"] = str(Path(__file__).resolve().parent)

# Fix PyTorch class path issue
if hasattr(torch.classes, '__dict__'):
    torch.classes.__dict__['_path'] = []


def log(message: str) -> None:
    """
    Log a timestamped message to the console with severity tags.

    Args:
        message (str): Message to log, including severity (e.g., [INFO], [OK], [WARNING], [ERROR]).
    """
    ts: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")


class ChatBot:
    """
    ChatBot uses FAISS-backed retrieval with LangChain and Ollama LLM.
    """

    def __init__(self, faiss_index_dir: str = config.VECTORDB_PATH) -> None:
        self._setup_event_loop()
        self._init_embeddings()
        self._init_llm()
        self._init_faiss(faiss_index_dir)
        self._init_chain()
        log("[OK] ChatBot initialization complete")


    def _setup_event_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        nest_asyncio.apply()
        log("[INFO] AsyncIO event loop configured")


    def _init_embeddings(self) -> None:

        if torch.cuda.is_available():
            device: str = 'cuda'
        elif torch.backends.mps.is_available():
            device: str = 'mps'
        else:
            device: str = 'cpu'

        log(f"[INFO] Using device for embeddings: {device}")

        self.model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': device}
        )
        log("[OK] Embedding model initialized")


    def _init_llm(self) -> None:
        try:
            self.llm = ChatOllama(model='llama3.2', temperature=0.3)
            log("[OK] LLM (Ollama) initialized")
        except Exception as e:
            log(f"[ERROR] Error initializing LLM: {e}")
            raise


    def _init_faiss(self, faiss_index_dir: str) -> None:
        try:
            embedding_dim: int = len(self.model.embed_query('test'))
            log(f"[INFO] Embedding dimension determined: {embedding_dim}")
        except Exception as e:
            log(f"[ERROR] Error getting embedding dimension: {e}")
            raise

        index = faiss.IndexFlatL2(embedding_dim)
        self.vector_store = connect_faiss(
            embedding_model=self.model,
            index=index,
            faiss_index_dir=faiss_index_dir
        )

        if hasattr(self.vector_store, 'index'):
            ntotal = self.vector_store.index.ntotal
            dim = self.vector_store.index.d
            log(f"[OK] FAISS initialized - Vectors: {ntotal} | Dimension: {dim}")


    def _init_chain(self) -> None:
        template: str = (
            "[Instructions] You are a friendly assistant. Answer the question based only on the following context.\n"
            "Your priority is to help the user. Give complete and concise information with the necessary details.\n"
            "Don't include unnecessary information.\n"
            "If you don't know the answer, say 'I have no information about that', no more details.\n"
            "Context: {context}\n"
            "Question: {input}\n"
            "Required format:\n"
        )
        self.prompt_template = PromptTemplate.from_template(template=template)
        self.document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt_template
        )
        log("[OK] Retrieval chain initialized")


    def query_llama(self, query: str, k_context: int = 3) -> Dict[str, Any]:
        """
        Query the LLM via a retrieval chain using FAISS.

        Args:
            query (str): User query string.
            k_context (int): Number of context documents to retrieve.

        Returns:
            Dict[str, Any]: Contains 'response' key with the LLM answer.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={'k': k_context})
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=self.document_chain
        )
        result = retrieval_chain.invoke({'input': query})
        answer: str = result.get('answer', '')
        log(f"[INFO] Query processed: '{query}' | Response length: {len(answer)} characters")
        return {'response': answer}


# if __name__ == '__main__':
#     bot = ChatBot()
    # Example usage
    # response = bot.query_llama('What is LangChain?')
    # log(f"[OK] Example response: {response['response']}")