# ============================================================
# app.py — Main Streamlit Application
# ============================================================
# PURPOSE:
#   This is the entry point of the application.
#   It creates the entire web interface using Streamlit.
#
# HOW TO RUN:
#   streamlit run app.py
#
# WHAT THIS FILE DOES:
#   - Creates the UI layout (sidebar + main chat area)
#   - Handles PDF uploads
#   - Manages chat history
#   - Calls the RAG pipeline when the user asks a question
#   - Displays answers with sources and metrics
# ============================================================

import streamlit as st
import os
from dotenv import load_dotenv

# Import our custom modules
from src.chatbot import (
    process_uploaded_pdf,
    get_or_create_vector_store,
    get_uploaded_documents_list
)
from src.rag_pipeline import answer_legal_question, summarize_documents
from src.vector_store import delete_vector_store

# Load environment variables from .env file
load_dotenv()

# ============================================================
# PAGE CONFIGURATION
# Must be the FIRST Streamlit command in the file
# ============================================================
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Dark legal-themed UI
# ============================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500&display=swap');

    /* ---- Global Theme ---- */
    .stApp {
        background-color: #0f1117;
        color: #e8e0d5;
        font-family: 'Source Sans 3', sans-serif;
    }

    /* ---- Main Header ---- */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #2a2d3a;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        color: #c9a84c;
        letter-spacing: 0.02em;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #8a8a9a;
        font-size: 0.95rem;
        font-weight: 300;
    }

    /* ---- Chat Messages ---- */
    .user-message {
        background: #1e2030;
        border-left: 3px solid #c9a84c;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
    .assistant-message {
        background: #161825;
        border-left: 3px solid #4a7c9e;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .user-label {
        font-size: 0.78rem;
        color: #c9a84c;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .assistant-label {
        font-size: 0.78rem;
        color: #4a7c9e;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    /* ---- Source Cards ---- */
    .source-card {
        background: #1a1c2a;
        border: 1px solid #2a2d3a;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        color: #9a9ab0;
    }
    .source-card strong {
        color: #c9a84c;
    }

    /* ---- Metrics Bar ---- */
    .metrics-bar {
        display: flex;
        gap: 1.5rem;
        padding: 0.6rem 0;
        border-top: 1px solid #2a2d3a;
        margin-top: 0.8rem;
        font-size: 0.82rem;
        color: #6a6a80;
    }
    .metric-item {
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #0d0f18 !important;
        border-right: 1px solid #1e2030;
    }
    .sidebar-section {
        border-bottom: 1px solid #1e2030;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    .doc-tag {
        background: #1e2030;
        border-radius: 4px;
        padding: 0.3rem 0.6rem;
        margin: 0.2rem 0;
        font-size: 0.82rem;
        color: #c9a84c;
        display: block;
    }

    /* ---- Disclaimer Banner ---- */
    .disclaimer {
        background: #1a1506;
        border: 1px solid #4a3a06;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #b09040;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* ---- Streamlit component overrides ---- */
    .stTextInput > div > div > input {
        background-color: #1e2030 !important;
        border-color: #2a2d3a !important;
        color: #e8e0d5 !important;
    }
    .stButton > button {
        background-color: #c9a84c !important;
        color: #0f1117 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background-color: #dab95e !important;
    }
    div[data-testid="stFileUploader"] {
        background-color: #1e2030;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stExpander {
        background-color: #161825 !important;
        border-color: #2a2d3a !important;
    }
    
    /* ---- Quick Example Buttons ---- */
    .example-btn {
        background: #1e2030;
        border: 1px solid #2a2d3a;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        font-size: 0.82rem;
        color: #9090aa;
        cursor: pointer;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# Streamlit re-runs the entire script on every interaction.
# st.session_state persists data between re-runs.
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of {role, content, sources, metrics}

if "vector_store" not in st.session_state:
    # Try to load an existing vector store on startup
    st.session_state.vector_store = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = []

if "use_openai_embeddings" not in st.session_state:
    st.session_state.use_openai_embeddings = False


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    
    # Logo / Title
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size: 2.5rem;'>⚖️</div>
        <div style='font-family: Playfair Display, serif; color: #c9a84c; font-size: 1.1rem; font-weight: 600;'>Legal AI Assistant</div>
        <div style='font-size: 0.75rem; color: #555; margin-top: 0.3rem;'>Powered by RAG</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ---- DOCUMENT UPLOAD SECTION ----
    st.markdown("### 📄 Upload Legal Documents")
    st.caption("Supported: PDF files (case law, contracts, NDAs)")
    
    # Embedding choice
    use_openai = st.checkbox(
        "Use OpenAI Embeddings",
        value=False,
        help="OpenAI embeddings are higher quality but require API credits. HuggingFace is free."
    )
    st.session_state.use_openai_embeddings = use_openai
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload legal PDFs to add to the knowledge base"
    )
    
    # Process the upload when a file is selected
    if uploaded_file is not None:
        # Check if this file was already uploaded in this session
        if uploaded_file.name not in st.session_state.documents_loaded:
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                try:
                    # Run the full ingestion pipeline
                    vector_store = process_uploaded_pdf(
                        uploaded_file_bytes=uploaded_file.read(),
                        filename=uploaded_file.name,
                        use_openai_embeddings=st.session_state.use_openai_embeddings
                    )
                    
                    # Save to session state
                    st.session_state.vector_store = vector_store
                    st.session_state.documents_loaded.append(uploaded_file.name)
                    
                    st.success(f"✅ '{uploaded_file.name}' indexed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        else:
            st.info(f"'{uploaded_file.name}' is already loaded.")
    
    # ---- LOADED DOCUMENTS LIST ----
    st.divider()
    st.markdown("### 📚 Loaded Documents")
    
    # Show documents from disk (persisted from previous sessions)
    all_docs = get_uploaded_documents_list()
    
    if all_docs:
        for doc_name in all_docs:
            st.markdown(f'<span class="doc-tag">📄 {doc_name}</span>', unsafe_allow_html=True)
        
        # Load vector store if not already in session
        if st.session_state.vector_store is None:
            with st.spinner("Loading existing documents..."):
                st.session_state.vector_store = get_or_create_vector_store(
                    use_openai_embeddings=st.session_state.use_openai_embeddings
                )
    else:
        st.caption("No documents loaded yet. Upload a PDF above.")
    
    # ---- SETTINGS ----
    st.divider()
    st.markdown("### ⚙️ Settings")
    
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=2,
        max_value=8,
        value=4,
        help="More chunks = more context but slower and more expensive"
    )
    
    # ---- ACTIONS ----
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset DB"):
            delete_vector_store()
            st.session_state.vector_store = None
            st.session_state.documents_loaded = []
            st.rerun()


# ============================================================
# MAIN CONTENT AREA
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ Legal AI Assistant</h1>
    <p>Ask questions about your uploaded legal documents — contracts, case law, NDAs, and more</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Educational Use Only</strong> — This assistant is for research and learning purposes. 
    It does not provide legal advice. Always consult a licensed attorney for legal decisions.
</div>
""", unsafe_allow_html=True)


# ---- QUICK EXAMPLE QUESTIONS ----
if not st.session_state.chat_history:
    st.markdown("#### 💡 Example Questions")
    st.caption("Click to use as your question:")
    
    example_questions = [
        "Does COVID-19 qualify as force majeure?",
        "Is this non-compete clause enforceable in California?",
        "What are the GDPR data obligations in this agreement?",
        "Summarize the key terms of this contract",
        "What are the termination conditions?",
        "Explain the liability limitation clause",
    ]
    
    # Display examples in a 2-column grid
    cols = st.columns(2)
    for i, example in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"🔹 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state["prefilled_question"] = example
                st.rerun()


# ---- CHAT HISTORY DISPLAY ----
st.markdown("### 💬 Conversation")

# Show all previous messages
for message in st.session_state.chat_history:
    
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="user-label">🧑 You</div>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    
    else:  # assistant
        st.markdown(f"""
        <div class="assistant-message">
            <div class="assistant-label">⚖️ Legal Assistant</div>
            {message["content"].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if message.get("sources"):
            with st.expander(f"📄 Sources used ({len(message['sources'])} document sections)", expanded=False):
                for source in message["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>📄 {source.get('file', 'Unknown')}</strong> — Page {source.get('page', '?')}<br>
                        <span style='font-size:0.8rem; color: #6a6a80;'>{source.get('preview', '')}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show evaluation metrics
        if message.get("metrics"):
            metrics = message["metrics"]
            st.markdown(f"""
            <div class="metrics-bar">
                <span class="metric-item">⏱️ {metrics.get('response_time', 0)}s</span>
                <span class="metric-item">📑 {metrics.get('num_chunks', 0)} chunks retrieved</span>
                <span class="metric-item">🔍 RAG-grounded</span>
            </div>
            """, unsafe_allow_html=True)


# ---- INPUT AREA ----
st.markdown("---")

# Pre-fill question from example buttons if clicked
default_question = st.session_state.pop("prefilled_question", "")

# Question input box
user_question = st.text_input(
    "Ask a legal question:",
    value=default_question,
    placeholder="e.g. 'What are the termination conditions in this contract?'",
    key="question_input",
    label_visibility="collapsed"
)

# Buttons row
col_ask, col_summarize, col_spacer = st.columns([1, 1, 3])

with col_ask:
    ask_clicked = st.button("🔍 Ask Question", type="primary", use_container_width=True)

with col_summarize:
    summarize_clicked = st.button("📋 Summarize Docs", use_container_width=True)


# ---- HANDLE ASK QUESTION ----
if ask_clicked and user_question.strip():
    
    # Check if documents are loaded
    if st.session_state.vector_store is None:
        st.error("⚠️ No documents loaded! Please upload a PDF first using the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Show loading spinner while generating answer
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                result = answer_legal_question(
                    question=user_question,
                    vector_store=st.session_state.vector_store,
                    top_k=top_k
                )
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "metrics": {
                        "response_time": result.get("response_time"),
                        "num_chunks": result.get("num_chunks")
                    }
                })
                
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ An error occurred: {str(e)}\n\nPlease check your API key and try again.",
                    "sources": [],
                    "metrics": {}
                })
        
        # Rerun to display the new messages
        st.rerun()


# ---- HANDLE SUMMARIZE DOCUMENTS ----
if summarize_clicked:
    
    if st.session_state.vector_store is None:
        st.error("⚠️ No documents loaded! Please upload a PDF first.")
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "content": "📋 Please summarize the uploaded legal documents."
        })
        
        with st.spinner("📋 Generating document summary..."):
            try:
                result = summarize_documents(
                    vector_store=st.session_state.vector_store,
                    top_k=6
                )
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "metrics": {
                        "response_time": result.get("response_time"),
                        "num_chunks": result.get("num_chunks")
                    }
                })
            
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Summary error: {str(e)}",
                    "sources": [],
                    "metrics": {}
                })
        
        st.rerun()


# ---- FOOTER ----
st.markdown("""
<div style='text-align: center; color: #3a3a4a; font-size: 0.78rem; padding: 2rem 0 1rem 0; border-top: 1px solid #1e2030; margin-top: 2rem;'>
    Legal AI Assistant — Capstone Project | Powered by LangChain + FAISS + OpenAI<br>
    <span style='color: #c9a84c;'>⚠️ For educational purposes only. Not legal advice.</span>
</div>
""", unsafe_allow_html=True)
