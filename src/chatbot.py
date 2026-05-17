# ============================================================
# src/chatbot.py
# ============================================================
# PURPOSE:
#   This file handles the document ingestion workflow —
#   the steps needed to go from a raw PDF file to a
#   searchable vector store ready for RAG queries.
#
#   Think of this as the "setup" module that prepares
#   everything before the user can start asking questions.
#
# WORKFLOW:
#   PDF File
#     → load_pdf() [pdf_loader.py]
#     → split_pages_into_chunks() [text_splitter.py]  
#     → get_embeddings() [embeddings.py]
#     → create_vector_store() [vector_store.py]
#     → Ready for questions!
# ============================================================

import os
import tempfile

from langchain_community.vectorstores import FAISS

# Import our pipeline modules
from src.pdf_loader import load_pdf
from src.text_splitter import split_pages_into_chunks
from src.embeddings import get_embeddings
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    add_to_vector_store
)


def process_uploaded_pdf(uploaded_file_bytes: bytes, filename: str, use_openai_embeddings: bool = False) -> FAISS:
    """
    Process a newly uploaded PDF file end-to-end.
    
    This function handles the full ingestion pipeline:
    1. Save the uploaded bytes to a temp file
    2. Extract text from the PDF
    3. Split into chunks
    4. Generate embeddings
    5. Create/update the vector store
    
    Args:
        uploaded_file_bytes: Raw bytes of the uploaded PDF (from Streamlit)
        filename: Original filename (for display/citation purposes)
        use_openai_embeddings: Whether to use OpenAI or HuggingFace embeddings
        
    Returns:
        Updated FAISS vector store
    """
    
    # ---- Step 1: Save uploaded bytes to a temporary file ----
    # We need to save to disk because PyMuPDF needs a file path
    os.makedirs("uploaded_docs", exist_ok=True)
    save_path = os.path.join("uploaded_docs", filename)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file_bytes)
    
    print(f"💾 Saved uploaded file to '{save_path}'")
    
    # ---- Step 2: Extract text from the PDF ----
    pages = load_pdf(save_path)
    
    if not pages:
        raise ValueError(f"Could not extract text from '{filename}'. Is it a text-based PDF?")
    
    # ---- Step 3: Split pages into chunks ----
    chunks = split_pages_into_chunks(
        pages=pages,
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
    )
    
    # ---- Step 4: Load embedding model ----
    embeddings = get_embeddings(use_openai=use_openai_embeddings)
    
    # ---- Step 5: Create or update the vector store ----
    # Check if a vector store already exists from previous uploads
    existing_store = load_vector_store(embeddings)
    
    if existing_store is None:
        # First document — create a brand new vector store
        vector_store = create_vector_store(chunks, embeddings)
    else:
        # Additional document — add to the existing store
        vector_store = add_to_vector_store(existing_store, chunks, embeddings)
    
    return vector_store


def get_or_create_vector_store(use_openai_embeddings: bool = False) -> FAISS | None:
    """
    Load an existing vector store, or return None if none exists.
    
    Used when the app starts up — we check if the user has
    already uploaded documents in a previous session.
    
    Args:
        use_openai_embeddings: Which embedding model to use
        
    Returns:
        FAISS vector store or None
    """
    
    embeddings = get_embeddings(use_openai=use_openai_embeddings)
    return load_vector_store(embeddings)


def get_uploaded_documents_list() -> list[str]:
    """
    Return a list of all previously uploaded document filenames.
    
    This is shown in the UI sidebar so the user knows which
    documents are currently loaded.
    
    Returns:
        List of filenames in the uploaded_docs folder
    """
    
    docs_folder = "uploaded_docs"
    
    if not os.path.exists(docs_folder):
        return []
    
    # List only PDF files
    pdf_files = [f for f in os.listdir(docs_folder) if f.lower().endswith(".pdf")]
    
    return sorted(pdf_files)
