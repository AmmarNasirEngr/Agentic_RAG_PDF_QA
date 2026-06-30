"""
vector_store.py — FAISS vector store management.

Each uploaded document gets its own sub-folder under VECTOR_STORE_PATH so
indices are never mixed:

    vector_store/
        <doc_id>/
            index.faiss
            index.pkl
"""

import os
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import VECTOR_STORE_PATH
from utils.embeddings import get_embeddings


# ── Path helpers ───────────────────────────────────────────────────────────────

def _store_path(doc_id: str) -> str:
    """Return the directory where a document's FAISS index lives."""
    return os.path.join(VECTOR_STORE_PATH, doc_id)


def vector_store_exists(doc_id: str) -> bool:
    """Return True if the FAISS index for *doc_id* has been saved to disk."""
    return os.path.exists(os.path.join(_store_path(doc_id), "index.faiss"))


# ── Write ──────────────────────────────────────────────────────────────────────

def create_vector_store(documents: List[Document], doc_id: str) -> FAISS:
    """
    Build a FAISS index from *documents* and persist it to disk.

    Args:
        documents: Chunked LangChain Documents (output of the chunking strategy).
        doc_id:    Unique identifier for the document.

    Returns:
        The in-memory FAISS vector store (also saved to disk).
    """
    embeddings = get_embeddings()
    path = _store_path(doc_id)
    os.makedirs(path, exist_ok=True)

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(path)

    return vector_store


# ── Read ───────────────────────────────────────────────────────────────────────

def load_vector_store(doc_id: str) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Raises:
        FileNotFoundError: If no index exists for *doc_id*.
    """
    if not vector_store_exists(doc_id):
        raise FileNotFoundError(
            f"No vector store found for document '{doc_id}'. "
            "Process the PDF first."
        )

    embeddings = get_embeddings()
    return FAISS.load_local(
        _store_path(doc_id),
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we wrote these files ourselves
    )


# ── Search ─────────────────────────────────────────────────────────────────────

def search_documents(
    doc_id: str, query: str, top_k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Retrieve the *top_k* most relevant chunks for *query* from *doc_id*'s index.

    Returns:
        List of (Document, similarity_score) tuples, best match first.
    """
    store = load_vector_store(doc_id)
    return store.similarity_search_with_score(query, k=top_k)


def get_retriever(doc_id: str, top_k: int = 5):
    """Return a LangChain-compatible retriever for use in LCEL chains and LangGraph nodes."""
    store = load_vector_store(doc_id)
    return store.as_retriever(search_kwargs={"k": top_k})
