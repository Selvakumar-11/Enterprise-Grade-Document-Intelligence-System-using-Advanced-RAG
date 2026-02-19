# ============================================================
#  main.py
#  Entry point — wires all modules together and runs the app
# ============================================================

from config import get_llm
from embeddings import get_embedding_model, get_reranker
from file_loader import load_files
from vectorstore import chunk_documents, build_vectorstore, load_vectorstore, get_retriever
from rag_chain import build_rag_chain, run_rag_query

import os

# ============================================================
# FILES TO INGEST
# Add your files here: (filename, department)
# Supported: .pdf, .docx, .txt
# ============================================================
FILES_TO_INGEST = [
    ("Elon_Musk.pdf",  "general"),
]

# Set to True to re-index documents, False to load existing DB
REBUILD_INDEX = True

# ============================================================
# STARTUP
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  RAG Assistant — Starting Up")
    print("=" * 60)

    # 1. Load models
    llm       = get_llm()
    embedding = get_embedding_model()
    reranker  = get_reranker()

    # 2. Build or load vectorstore
    if REBUILD_INDEX:
        raw_docs  = load_files(FILES_TO_INGEST)
        chunks    = chunk_documents(raw_docs)
        vectorstore = build_vectorstore(chunks, embedding)
    else:
        vectorstore = load_vectorstore(embedding)

    # 3. Get retriever and RAG chain
    retriever = get_retriever(vectorstore, k=10)
    rag_chain = build_rag_chain(llm)

    # 4. Interactive loop
    print("\n" + "=" * 60)
    print("  Ready! Type your question or 'exit' to quit.")
    print("=" * 60)

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            print("Goodbye! 👋")
            break

        print("\n🔍 Searching and reranking...\n")
        answer = run_rag_query(query, retriever, reranker, rag_chain)
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()