# ============================================================
#  embeddings.py
#  Loads the HuggingFace embedding model and cross-encoder reranker
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME  = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and return the sentence-transformer embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_reranker() -> CrossEncoder:
    """Load and return the cross-encoder reranker model."""
    print(f"Loading reranker model: {RERANKER_MODEL_NAME}")
    return CrossEncoder(RERANKER_MODEL_NAME)