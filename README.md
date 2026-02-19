# 🤖 RAG Assistant - Streamlit Edition

A production-ready RAG (Retrieval-Augmented Generation) system with a beautiful Streamlit UI.

## ✨ Features

- 📁 **Multi-file upload** (PDF, DOCX, TXT)
- 🧠 **Smart reranking** using cross-encoder
- 📊 **Metadata tracking** (source, page number, department)
- 💬 **Interactive Q&A** interface
- 📜 **Chat history** with expandable previous questions
- 🎯 **Source citations** in every answer

## 🏗️ Project Structure

```
rag_app/
├── streamlit_app.py    # 🚀 Streamlit UI (RUN THIS)
├── main.py             # 🔧 CLI version
├── config.py           # Azure OpenAI setup
├── embeddings.py       # AI models (embedding + reranker)
├── file_loader.py      # PDF/DOCX/TXT reader
├── vectorstore.py      # ChromaDB storage
├── rag_chain.py        # RAG pipeline
├── requirements.txt    # Dependencies
├── .env                # Your API keys (create this)
└── uploaded_files/     # Auto-created on first upload
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the `rag_app/` folder:

```env
# Azure OpenAI Credentials
SERVICE_LINE=your_service_line
BRAND=your_brand
PROJECT=your_project
HEADER_API_VERSION=your_header_api_version
API_KEY=your_api_key
API_VERSION=your_api_version
END_POINT=https://your-resource.openai.azure.com/
DEPLOYMENT_ID=your_deployment_name
```

### 3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload Files** 📁
   - Click "Browse files" in the sidebar
   - Select PDF, DOCX, or TXT files
   - Files are saved to `uploaded_files/` folder

2. **Index Documents** 🔄
   - Click "Index Documents" button
   - Wait for embedding process to complete
   - ChromaDB is created at `./chroma_db/`

3. **Ask Questions** 💬
   - Type your question in the input box
   - Click "Get Answer"
   - View answer, sources, and metadata

4. **Clear & Start Over** 🗑️
   - Click "Clear All Files" to reset everything

## 🧠 Models Used

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** Azure OpenAI (GPT-4o-128k)

## 🔄 Workflow

```
Upload Files → Save to uploaded_files/ 
     ↓
Index Documents → Chunk → Embed → Store in ChromaDB
     ↓
Ask Question → Retrieve (10) → Rerank (3) → LLM Answer
     ↓
Display: Answer + Sources + Metadata
```

## 📝 CLI Version (Optional)

If you prefer command-line interface:

```bash
python main.py
```

Update `FILES_TO_INGEST` in `main.py` before running.

## 🛠️ Customization

### Change Chunk Size
Edit `vectorstore.py`:
```python
CHUNK_SIZE = 500     # Increase for longer context
CHUNK_OVERLAP = 50   # Increase for better continuity
```

### Change Number of Retrieved Docs
Edit `streamlit_app.py`:
```python
st.session_state.retriever = get_retriever(vectorstore, k=10)  # Change k value
```

### Adjust Reranking Top-N
Edit `rag_chain.py`:
```python
def rerank_docs(..., top_n: int = 3):  # Change top_n value
```

## 🎯 Tips for Best Results

1. **Quality documents** → Upload well-formatted PDFs/DOCX for better extraction
2. **Clear questions** → Be specific in your queries
3. **Re-index when needed** → Click "Index Documents" after uploading new files
4. **Check sources** → Always verify citations in the metadata box

## 🐛 Troubleshooting

**Issue:** "No module named 'streamlit'"
- **Fix:** Run `pip install -r requirements.txt`

**Issue:** Slow indexing
- **Fix:** Normal for first-time embedding. Subsequent runs load from disk.

**Issue:** "I don't have enough information"
- **Fix:** Your documents might not contain the answer. Try rephrasing or upload more relevant docs.

---

**Made with ❤️ using LangChain, Streamlit, and Sentence Transformers**