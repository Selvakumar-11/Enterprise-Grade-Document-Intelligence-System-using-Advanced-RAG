# ============================================================
#  streamlit_app.py
#  Streamlit UI for RAG Assistant
#  Features: File upload, Q&A interface, metadata display
# ============================================================

import streamlit as st
import os
from pathlib import Path
import shutil

from config      import get_llm
from embeddings  import get_embedding_model, get_reranker
from file_loader import load_files
from vectorstore import chunk_documents, build_vectorstore, load_vectorstore, get_retriever
from rag_chain   import build_rag_chain, run_rag_query

# ============================================================
# CONFIGURATION
# ============================================================
UPLOAD_FOLDER = "uploaded_files"
CHROMA_DB_PATH = "./chroma_db"

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_clear_counter" not in st.session_state:
    st.session_state.question_clear_counter = 0

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_resource
def load_models():
    """Load all models (cached so they're only loaded once)."""
    llm = get_llm()
    embedding = get_embedding_model()
    reranker = get_reranker()
    return llm, embedding, reranker


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to the upload folder and return its path."""
    file_path = Path(UPLOAD_FOLDER) / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def get_uploaded_files_for_indexing():
    """
    Get all files from the upload folder as a list of (filename, department) tuples.
    """
    files = []
    for file_path in Path(UPLOAD_FOLDER).glob("*"):
        if file_path.suffix.lower() in [".pdf", ".docx", ".txt"]:
            files.append((str(file_path), "general"))
    return files


def index_documents(embedding):
    """Index all uploaded documents into ChromaDB."""
    import time
    import uuid
    
    files_to_ingest = get_uploaded_files_for_indexing()
    
    if not files_to_ingest:
        return None
    
    with st.spinner("📚 Loading and indexing documents..."):
        raw_docs = load_files(files_to_ingest)
        chunks = chunk_documents(raw_docs)
        
        # Delete old vectorstore safely (Windows-compatible)
        db_path = CHROMA_DB_PATH
        try:
            if Path(db_path).exists():
                # Try to delete, but if it fails (locked files), use new path
                try:
                    shutil.rmtree(db_path)
                    time.sleep(0.5)  # Brief pause to ensure cleanup
                except (PermissionError, OSError):
                    # On Windows, ChromaDB might lock files - use a new path
                    db_path = f"./chroma_db_{uuid.uuid4().hex[:8]}"
                    st.info(f"Using new database: {db_path}")
        except Exception as e:
            st.warning(f"Note: Could not clear old database, creating new one")
        
        # Build vectorstore with the appropriate path
        from langchain_chroma import Chroma
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=db_path,
        )
    
    return vectorstore


def extract_metadata_from_answer(answer: str):
    """
    Extract the 'Sources:' section from the answer.
    Returns (main_answer, sources)
    """
    if "Sources:" in answer:
        parts = answer.split("Sources:", 1)
        main_answer = parts[0].replace("Answer:", "").strip()
        sources = parts[1].strip()
        return main_answer, sources
    else:
        return answer.replace("Answer:", "").strip(), "No sources found"


# ============================================================
# SIDEBAR - FILE UPLOAD
# ============================================================
st.sidebar.title("📁 Upload Documents")
st.sidebar.markdown("Upload PDF, DOCX, or TXT files to build your knowledge base.")

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="file_uploader"
)

# Save uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_files_list:
            file_path = save_uploaded_file(uploaded_file)
            st.session_state.uploaded_files_list.append(uploaded_file.name)
            st.sidebar.success(f"✅ Saved: {uploaded_file.name}")

# Show uploaded files
if st.session_state.uploaded_files_list:
    st.sidebar.markdown("### 📄 Uploaded Files:")
    for filename in st.session_state.uploaded_files_list:
        st.sidebar.text(f"• {filename}")

# Index button
if st.sidebar.button("🔄 Index Documents", type="primary"):
    if not st.session_state.uploaded_files_list:
        st.sidebar.error("⚠️ Please upload files first!")
    else:
        llm, embedding, reranker = load_models()
        vectorstore = index_documents(embedding)
        
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = get_retriever(vectorstore, k=10)
            st.session_state.rag_chain = build_rag_chain(llm)
            st.session_state.reranker = reranker
            st.session_state.vectorstore_ready = True
            st.sidebar.success("✅ Documents indexed successfully!")
        else:
            st.sidebar.error("⚠️ No valid documents found to index!")

# Clear all button
if st.sidebar.button("🗑️ Clear All Files"):
    # Remove all files from upload folder
    for file_path in Path(UPLOAD_FOLDER).glob("*"):
        try:
            file_path.unlink()
        except:
            pass
    
    # Note about vectorstore cleanup on Windows
    st.sidebar.warning("⚠️ Restart the app to fully clear the database")
    
    # Reset session state
    st.session_state.uploaded_files_list = []
    st.session_state.vectorstore_ready = False
    st.session_state.chat_history = []
    
    st.sidebar.success("✅ Files cleared! Please restart the app.")

# ============================================================
# MAIN AREA - Q&A INTERFACE
# ============================================================
st.title("🤖 RAG Assistant")
st.markdown("Ask questions about your uploaded documents!")

# Check if vectorstore is ready
if not st.session_state.vectorstore_ready:
    st.info("👈 Please upload documents and click 'Index Documents' to get started.")
else:
    # Question input
    st.markdown("### 💬 Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Type your question here:",
            placeholder="e.g., What is the main topic of the document?",
            key=f"question_input_{st.session_state.question_clear_counter}"
        )
    
    with col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear"):
            st.session_state.question_clear_counter += 1
            st.rerun()
    
    ask_button = st.button("🔍 Get Answer", type="primary")
    
    if ask_button and question:
        with st.spinner("🔍 Searching and generating answer..."):
            # Get answer
            llm, embedding, reranker = load_models()
            full_answer = run_rag_query(
                question,
                st.session_state.retriever,
                st.session_state.reranker,
                st.session_state.rag_chain,
            )
            
            # Extract answer and metadata
            answer, sources = extract_metadata_from_answer(full_answer)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })
    
    # Display results
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]
        
        # Answer box
        st.markdown("### ✨ Answer")
        st.success(latest["answer"])
        
        # Metadata box
        st.markdown("### 📎 Sources & Metadata")
        st.info(latest["sources"])
        
        # Show chat history
        if len(st.session_state.chat_history) > 1:
            with st.expander("📜 Previous Questions"):
                for i, entry in enumerate(reversed(st.session_state.chat_history[:-1])):
                    st.markdown(f"**Q{len(st.session_state.chat_history) - i - 1}:** {entry['question']}")
                    st.markdown(f"**A:** {entry['answer'][:200]}...")
                    st.markdown("---")

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("Built with 🤖 LangChain + Streamlit")
st.sidebar.markdown("Embeddings: `all-MiniLM-L6-v2`")
st.sidebar.markdown("Reranker: `ms-marco-MiniLM-L-6-v2`")