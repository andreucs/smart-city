import os
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import faiss
import torch
import pandas as pd
import PyPDF2
import config

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Disable parallel tokenizers warnings and set static directory
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STATIC_DIRECTORY"] = str(Path.cwd())
# Clear PyTorch classes path to avoid conflicts
torch.classes.path = []


def log(message: str) -> None:
    """
    Log a timestamped message to the console.

    Args:
        message (str): The message to log, with severity tags (e.g., [INFO], [OK], [WARNING]).
    """
    timestamp: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def process_single_pdf(
    file_path: str,
    pdf_id: int,
    chunk_size: int,
    chunk_overlap: int
) -> List[Tuple[str, int, int]]:
    """
    Process a PDF by extracting text, cleaning, and splitting into chunks.

    Args:
        file_path (str): Path to the PDF file.
        pdf_id (int): Unique identifier for this PDF.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[Tuple[str, int, int]]: List of (chunk_text, page_number, pdf_id).
            Returns an empty list if processing fails.
    """
    pdf_chunks: List[Tuple[str, int, int]] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as e:
                    log(f"[WARNING] Could not decrypt '{file_path}' (ID: {pdf_id}): {e}")
                    return []

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if not text:
                        continue
                    cleaned = text.replace("\xa0", " ").strip()
                    for chunk in splitter.split_text(cleaned):
                        pdf_chunks.append((chunk, page_num, pdf_id))
                except Exception as page_error:
                    log(f"[WARNING] Error on page {page_num} of '{file_path}' (ID: {pdf_id}): {page_error}")
                    continue

    except FileNotFoundError:
        log(f"[ERROR] File not found: '{file_path}' (ID: {pdf_id})")
    except PyPDF2.errors.PdfReadError as pdf_error:
        log(f"[ERROR] PDF read error for '{file_path}' (ID: {pdf_id}): {pdf_error}")
    except Exception as e:
        log(f"[ERROR] Unexpected error processing '{file_path}' (ID: {pdf_id}): {e}")

    log(f"[OK] Processed {len(pdf_chunks)} chunks from '{file_path}' (ID: {pdf_id})")
    return pdf_chunks


def connect_faiss(
    embedding_model: HuggingFaceEmbeddings,
    index: faiss.Index,
    faiss_index_dir: str = config.VECTORDB_PATH
) -> FAISS:
    """
    Load existing FAISS index or create a new one.

    Args:
        embedding_model (HuggingFaceEmbeddings): Embedding model.
        index (faiss.Index): FAISS index instance.
        faiss_index_dir (str): Directory for index persistence.

    Returns:
        FAISS: Vector store instance.
    """
    try:
        store = FAISS.load_local(
            faiss_index_dir,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        log(f"[OK] Loaded FAISS index from '{faiss_index_dir}'")
    except Exception:
        store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        log(f"[INFO] Created new FAISS index in memory")
    return store


def commit_faiss(
    vector_store: FAISS,
    faiss_index_dir: str = config.VECTORDB_PATH
) -> bool:
    """
    Save FAISS index to disk.

    Args:
        vector_store (FAISS): FAISS vector store.
        faiss_index_dir (str): Directory path.

    Returns:
        bool: True if saved successfully.
    """
    vector_store.save_local(faiss_index_dir)
    log(f"[OK] Saved FAISS index to '{faiss_index_dir}'")
    return True


def insert_pdf_to_faiss(
    chunks: List[Tuple[str, int, int]],
    vector_store: FAISS
) -> bool:
    """
    Insert text chunks into FAISS index.

    Args:
        chunks (List[Tuple[str, int, int]]): (text, page, pdf_id).
        vector_store (FAISS): Vector store.

    Returns:
        bool: True if insertion succeeded.
    """
    embedding_model = vector_store.embedding_function
    documents = [
        Document(page_content=text, metadata={"page": page})
        for text, page, _ in chunks
    ]
    vector_store.add_documents(documents, embedding=embedding_model)
    log("[OK] Inserted chunks into FAISS index")
    return True


def insert_csv_to_faiss(
    csv_path: str,
    vector_store: FAISS
) -> bool:
    """
    Insert CSV file entries into FAISS index.

    Args:
        csv_path (str): Path to CSV.
        vector_store (FAISS): Vector store.

    Returns:
        bool: True if insertion succeeded.
    """
    log(f"[INFO] Loading CSV data from '{csv_path}'")
    df = pd.read_csv(csv_path, encoding="utf-8", sep=";")
    documents = [
        Document(
            page_content=f"{row['q']} {row['a']}",
            metadata={"section": row.get('section')}
        )
        for _, row in df.iterrows()
    ]
    vector_store.add_documents(documents, embedding=vector_store.embedding_function)
    log(f"[OK] Inserted data from '{csv_path}' into FAISS index")
    return True


def main() -> None:
    """
    Main routine to set up and populate FAISS vector store.
    """
    log("[INFO] Starting FAISS vector store setup")

    # Determine compute device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log(f"[INFO] Using device: {device}")

    # Initialize embedding model
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    embedding_dim = len(model.embed_query("test"))
    log(f"[OK] Loaded embedding model with dimension {embedding_dim}")

    # Create FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    log("[INFO] Created FAISS index with L2 metric")

    # Load or create vector store
    vector_store = connect_faiss(model, index)
    commit_faiss(vector_store)

    # Process PDF and insert chunks
    pdf_chunks = process_single_pdf(
        f"{config.DATA_DOCS_PATH}/CGAUS_en_valenbisi.pdf",
        pdf_id=1,
        chunk_size=500,
        chunk_overlap=150
    )
    insert_pdf_to_faiss(pdf_chunks, vector_store)

    # Insert CSV data and save
    insert_csv_to_faiss(f"{config.DATA_DOCS_PATH}/FAQ-valenbici.csv", vector_store)
    commit_faiss(vector_store)

    log("[OK] Completed FAISS vector store setup")


if __name__ == "__main__":
    main()