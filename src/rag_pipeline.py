# ============================================================
# src/rag_pipeline_free.py — FREE VERSION
# ============================================================
# Drop-in replacement for rag_pipeline.py that uses:
#   - Ollama (local, free) as the LLM
#   - OR Groq (free cloud API, faster)
#   - OR Google Gemini (free cloud API, best quality)
#
# To use: rename this file to rag_pipeline.py
# (or copy its contents into the original file)
#
# BEFORE RUNNING:
#   1. Install Ollama: https://ollama.com
#   2. Run: ollama pull llama3.1
#   3. Run: ollama serve
#   4. Then: streamlit run app.py
# ============================================================

import os
import time
from dotenv import load_dotenv
from langchain.schema import Document, SystemMessage, HumanMessage

from src.vector_store import search_vector_store
from src.prompts import (
    LEGAL_SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    QUERY_REFINEMENT_PROMPT,
    CLARIFICATION_NEEDED_PROMPT,
    SUMMARY_PROMPT
)

load_dotenv()

# Words that suggest the question is too vague to search for
VAGUE_KEYWORDS = ["this", "it", "that", "the contract", "the document", "explain"]


def get_llm():
    """
    Get the free LLM based on which option is configured in .env
    
    Priority order:
    1. Groq (if GROQ_API_KEY is set)
    2. Google AI Studio (if GOOGLE_API_KEY is set)
    3. Ollama (default — runs locally)
    
    Returns:
        A LangChain-compatible LLM object
    """
    
    # ---- Option A: Ollama (default — local and free) ----
    try:
        from langchain_ollama import ChatOllama
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        print(f"🦙 Using Ollama local model: {model_name}")
        print("   (Responses take 30-90 seconds on CPU — this is normal!)")
        
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0,
            num_predict=1500
        )
    except ImportError:
        raise ImportError("Please install: pip install langchain-ollama")
    except Exception as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url}\n"
            f"Make sure Ollama is running. Run: ollama serve\n"
            f"Original error: {e}"
        )


def format_context_from_chunks(chunks: list[Document]) -> str:
    """
    Format retrieved document chunks into a single readable text block.
    Each chunk is labeled with its source file and page number.
    """
    
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "Unknown Document")
        page   = chunk.metadata.get("page_number", "?")
        labeled = f"[Excerpt {i+1} from '{source}', Page {page}]\n{chunk.page_content}"
        context_parts.append(labeled)
    
    return "\n\n---\n\n".join(context_parts)


def is_question_vague(question: str) -> bool:
    """Return True if the question is too short or uses only vague words."""
    
    q = question.lower().strip()
    if len(q.split()) < 4:
        return True
    for word in VAGUE_KEYWORDS:
        if q == word:
            return True
    return False


def ask_clarifying_question(question: str, llm) -> str:
    """Generate a helpful clarifying question when the user's input is vague."""
    
    prompt   = CLARIFICATION_NEEDED_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    return response.content


def refine_search_query(original_question: str, llm) -> str:
    """
    When a search returns no useful results, ask the LLM to
    rephrase the question with different legal keywords.
    """
    
    prompt   = QUERY_REFINEMENT_PROMPT.format(original_question=original_question)
    response = llm.invoke(prompt)
    refined  = response.content.strip()
    print(f"🔄 Refined query: '{refined}'")
    return refined


def is_context_sufficient(chunks: list[Document]) -> bool:
    """Check that retrieved chunks contain meaningful content."""
    
    if not chunks:
        return False
    total = sum(len(c.page_content) for c in chunks)
    return total > 200


def answer_legal_question(question: str, vector_store, top_k: int = 4) -> dict:
    """
    Main function: answer a legal question using RAG + a free local/cloud LLM.
    
    Agent loop:
      1. Vague question? → ask clarification
      2. Search vector store
      3. Not enough context? → refine query and retry
      4. Still nothing? → honest "not found" message
      5. Otherwise → build prompt and call the LLM
    
    Args:
        question: The user's legal question
        vector_store: FAISS index with uploaded document chunks
        top_k: How many chunks to retrieve
        
    Returns:
        dict: answer, sources, response_time, num_chunks, is_clarification_request
    """
    
    start_time = time.time()
    
    # ---- Connect to the free LLM ----
    try:
        llm = get_llm()
    except Exception as e:
        return {
            "answer": f"❌ Could not start the AI model.\n\n{str(e)}\n\n"
                      f"Quick fix for Ollama:\n"
                      f"  1. Install: https://ollama.com\n"
                      f"  2. Run in terminal: ollama pull llama3.1\n"
                      f"  3. Run in terminal: ollama serve\n"
                      f"  4. Then reload this page.",
            "sources": [],
            "response_time": 0,
            "num_chunks": 0,
            "is_clarification_request": False
        }
    
    # ---- Step 1: Vague question check ----
    if is_question_vague(question):
        clarification = ask_clarifying_question(question, llm)
        elapsed = time.time() - start_time
        return {
            "answer": f"🤔 Your question seems a bit vague. Could you clarify?\n\n{clarification}",
            "sources": [],
            "response_time": round(elapsed, 2),
            "num_chunks": 0,
            "is_clarification_request": True
        }
    
    # ---- Step 2: Search vector store ----
    print(f"\n🔍 Searching: '{question}'")
    retrieved_chunks = search_vector_store(vector_store, question, top_k=top_k)
    
    # ---- Step 3: Retry with refined query if needed ----
    if not is_context_sufficient(retrieved_chunks):
        print("⚠️  Insufficient results. Trying refined query...")
        refined = refine_search_query(question, llm)
        retrieved_chunks = search_vector_store(vector_store, refined, top_k=top_k)
    
    # ---- Step 4: Honest fallback if still empty ----
    if not is_context_sufficient(retrieved_chunks):
        elapsed = time.time() - start_time
        return {
            "answer": (
                "❌ I could not find enough information in the uploaded documents to answer this question.\n\n"
                "Please make sure you have uploaded a relevant legal PDF, or try rephrasing your question.\n\n"
                "⚠️ This is for educational/research purposes only. Not legal advice."
            ),
            "sources": [],
            "response_time": round(elapsed, 2),
            "num_chunks": 0,
            "is_clarification_request": False
        }
    
    # ---- Step 5: Build prompt with context ----
    context_text = format_context_from_chunks(retrieved_chunks)
    user_prompt  = RAG_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question
    )
    
    # ---- Step 6: Call the LLM ----
    print("🤖 Generating answer...")
    messages = [
        SystemMessage(content=LEGAL_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    response = llm.invoke(messages)
    answer   = response.content
    
    # ---- Collect source citations ----
    sources, seen = [], set()
    for chunk in retrieved_chunks:
        key = f"{chunk.metadata.get('source')}_{chunk.metadata.get('page_number')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file":    chunk.metadata.get("source", "Unknown"),
                "page":    chunk.metadata.get("page_number", "?"),
                "preview": chunk.page_content[:200] + "..."
            })
    
    elapsed = time.time() - start_time
    print(f"✅ Done in {elapsed:.2f}s")
    
    return {
        "answer":                  answer,
        "sources":                 sources,
        "response_time":           round(elapsed, 2),
        "num_chunks":              len(retrieved_chunks),
        "is_clarification_request": False
    }


def summarize_documents(vector_store, top_k: int = 6) -> dict:
    """Generate a structured summary of the uploaded legal documents."""
    
    start_time = time.time()
    llm        = get_llm()
    
    queries    = ["contract parties obligations terms", "agreement scope purpose"]
    all_chunks = []
    for q in queries:
        all_chunks.extend(search_vector_store(vector_store, q, top_k=top_k // 2))
    
    # Deduplicate
    seen, unique = set(), []
    for chunk in all_chunks:
        key = chunk.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    
    context  = format_context_from_chunks(unique[:top_k])
    prompt   = SUMMARY_PROMPT.format(context=context)
    response = llm.invoke(prompt)
    elapsed  = time.time() - start_time
    
    return {
        "answer":       response.content,
        "sources":      [{"file": c.metadata.get("source"), "page": c.metadata.get("page_number")} for c in unique],
        "response_time": round(elapsed, 2),
        "num_chunks":   len(unique),
        "is_clarification_request": False
    }
