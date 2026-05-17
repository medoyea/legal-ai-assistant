# ============================================================
# src/embeddings.py
# ============================================================
# PURPOSE:
#   Embeddings convert text into numbers (vectors) that capture
#   the MEANING of the text. Similar texts get similar vectors.
#
# WHY EMBEDDINGS?
#   We can't search legal text by keywords alone. A question like
#   "Can the party exit the agreement?" should match chunks that 
#   say "termination clause" even though the words are different.
#   Embeddings handle this semantic (meaning-based) matching.
#
# TWO OPTIONS PROVIDED:
#   1. HuggingFace (free, runs locally) — good for students
#   2. OpenAI (paid, higher quality) — better for production
# ============================================================

import os
from dotenv import load_dotenv

# LangChain wrappers for different embedding providers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()


def get_huggingface_embeddings() -> HuggingFaceEmbeddings:
    """
    Create a FREE embedding model using HuggingFace.
    
    Model: BAAI/bge-base-en-v1.5
    - This is one of the best free embedding models available
    - It runs locally on your computer (no API key needed)
    - It understands legal language well
    
    First run will download the model (~400MB) — this is normal!
    
    Returns:
        HuggingFaceEmbeddings object ready to use
    """
    
    print("🤗 Loading HuggingFace embeddings (BAAI/bge-base-en-v1.5)...")
    print("   (First run will download the model — please wait)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        # Run on CPU (works on any computer, no GPU needed)
        model_kwargs={"device": "cpu"},
        # Normalize vectors for better similarity comparison
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("✅ HuggingFace embeddings ready!")
    
    return embeddings


def get_openai_embeddings() -> OpenAIEmbeddings:
    """
    Create embeddings using OpenAI's API.
    
    Model: text-embedding-3-small
    - Higher quality than free models, especially for legal text
    - Requires OPENAI_API_KEY in your .env file
    - Costs money per request (very cheap though)
    
    Returns:
        OpenAIEmbeddings object ready to use
    """
    
    # Make sure the API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found in .env file!")
    
    print("🔑 Using OpenAI embeddings (text-embedding-3-small)...")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    
    print("✅ OpenAI embeddings ready!")
    
    return embeddings


def get_embeddings(use_openai: bool = False):
    """
    Main function to get embeddings — choose which provider to use.
    
    By default, uses the FREE HuggingFace option.
    Set use_openai=True if you want better quality (requires API key).
    
    Args:
        use_openai: If True, use OpenAI. If False, use HuggingFace (default)
        
    Returns:
        An embeddings object that can be used with FAISS vector store
    """
    
    if use_openai:
        return get_openai_embeddings()
    else:
        return get_huggingface_embeddings()
