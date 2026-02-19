# ============================================================
#  rag_chain.py
#  Reranking logic, prompt template, and full RAG pipeline
# ============================================================

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from sentence_transformers import CrossEncoder

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not found in the context, say "I don't have enough information."

Context:
{context}

Question:
{question}

Provide your response in this exact format:

Answer:
<your answer here>

Sources:
<list each source document and page number used — one per line>
"""


# ── Reranking ────────────────────────────────────────────────

def rerank_docs(
    query: str,
    docs: list[Document],
    reranker: CrossEncoder,
    top_n: int = 3,
) -> list[Document]:
    """
    Score each retrieved doc against the query using the cross-encoder,
    return only the top_n highest-scoring documents.
    """
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


# ── Context Formatting ───────────────────────────────────────

def format_docs_with_metadata(docs: list[Document]) -> str:
    """Format docs with their metadata as context for the LLM."""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        formatted.append(
            f"[Source: {meta.get('source', 'unknown')} | "
            f"Page: {meta.get('page_number', 'N/A')} | "
            f"Dept: {meta.get('department', 'N/A')}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


# ── RAG Pipeline ─────────────────────────────────────────────

def build_rag_chain(llm: AzureChatOpenAI):
    """Build and return the LangChain RAG chain (prompt → LLM → parser)."""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()


def run_rag_query(
    query: str,
    retriever,
    reranker: CrossEncoder,
    rag_chain,
    top_n: int = 3,
) -> str:
    """
    Full RAG pipeline:
      1. Retrieve top-k candidates from vectorstore
      2. Rerank using cross-encoder → keep top_n
      3. Format context with metadata
      4. Call LLM and return the answer
    """
    # Step 1: Retrieve
    retrieved_docs = retriever.invoke(query)

    # Step 2: Rerank
    reranked_docs = rerank_docs(query, retrieved_docs, reranker, top_n=top_n)

    # Step 3: Format context
    context = format_docs_with_metadata(reranked_docs)

    # Step 4: Generate
    return rag_chain.invoke({"context": context, "question": query})