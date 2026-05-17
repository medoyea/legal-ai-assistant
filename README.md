# Legal AI Assistant Mohamed Hassanein

**Generative AI Course**
---

## 🗂️ Project Structure

```
legal-ai-assistant/
│
├── app.py                  # Main Streamlit web application (entry point)
├── requirements.txt        # All required Python libraries
├── .env.example            # Template for your API keys (copy to .env)
├── .gitignore              # Files to exclude from Git
├── README.md               # This file
│
├── data/                   # (Optional) Store sample PDFs here
├── vectorstore/            # Auto-created: FAISS index saved here
├── uploaded_docs/          # Auto-created: Uploaded PDFs saved here
│
└── src/                    # All source code modules
    ├── __init__.py         # Makes src/ a Python package
    ├── pdf_loader.py       # Reads PDFs with PyMuPDF, extracts text
    ├── text_splitter.py    # Splits text into chunks for embedding
    ├── embeddings.py       # Generates vector embeddings (HuggingFace or OpenAI)
    ├── vector_store.py     # FAISS vector store: create, save, load, search
    ├── rag_pipeline.py     # Core RAG logic + simple agent behavior
    ├── chatbot.py          # Document ingestion workflow
    └── prompts.py          # All AI prompts in one place
```

---

## How to Use

1. **Upload a PDF** using the sidebar — any legal document works
2. **Wait for indexing** — the system reads and stores the document (30–60 seconds)
3. **Ask a question** in the chat box — use natural language
4. **View the answer** with source citations and document excerpts
5. **Check metrics** — response time and number of chunks retrieved


## How RAG Works (Simple Explanation)

```
Your Question
      ↓
Convert to vector (embedding)
      ↓
Search FAISS for similar document chunks
      ↓
Retrieve top 4 most relevant sections
      ↓
Build prompt: System prompt + Context chunks + Your question
      ↓
Send to Ollama
      ↓
Get grounded answer with citations
```

**Why this prevents hallucinations:**
The LLM is instructed to answer ONLY from the provided context. If the answer isn't in the documents, it says so.

---

## 📦 Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit 
| PDF Reading | PyMuPDF (fitz) 
| Text Splitting | LangChain 
| Embeddings | HuggingFace
| Vector Store | FAISS 
| LLM | Ollama

## Project by

**Mohamed Ahmed Hassanein - 231017770**

