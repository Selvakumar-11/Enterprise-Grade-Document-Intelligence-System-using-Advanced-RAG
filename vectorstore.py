# ============================================================
#  vectorstore.py
#  Handles document chunking, ChromaDB storage, and retriever
# ============================================================

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PERSIST_DIR = "./chroma_db"

# Chunking config — tweak these if retrieval quality needs tuning
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split raw documents into smaller chunks while preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Total chunks after splitting: {len(chunks)}")
    return chunks


def build_vectorstore(
    docs: list[Document],
    embedding: HuggingFaceEmbeddings,
) -> Chroma:
    """
    Embed and store documents in ChromaDB (persistent).
    Use this when indexing for the first time or re-indexing.
    """
    print("📦 Building ChromaDB vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print("✅ Vectorstore ready.")
    return vectorstore


def load_vectorstore(embedding: HuggingFaceEmbeddings) -> Chroma:
    """
    Load an existing ChromaDB vectorstore from disk.
    Use this on subsequent runs (no need to re-embed).
    """
    print("📂 Loading existing ChromaDB vectorstore...")
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding,
    )


def get_retriever(vectorstore: Chroma, k: int = 10):
    """
    Return a retriever that fetches top-k candidates.
    k=10 gives the reranker enough candidates to work with.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})