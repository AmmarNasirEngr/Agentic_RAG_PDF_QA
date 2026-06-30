"""
tenant_vector_store.py — Session-isolated FAISS vector store for Parent-Child RAG.

Storage layout:
    storage/vectors/{session_id}/{doc_id}/index.faiss
                                          index.pkl

Design:
  • One FAISS index per (session_id, doc_id) pair — users never share indices.
  • Only CHILD chunks are embedded (small, precise retrieval units).
  • Child metadata stored in FAISS pkl: {"child_id": str, "parent_id": str, "page": int}
  • retrieve_and_package() performs the full retrieval → parent-lookup → deduplication pipeline.
"""

from __future__ import annotations

import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import STORAGE_PATH, TOP_K
from utils.embeddings import get_embeddings
from utils.hierarchical_chunker import ChildChunk, ParentChunk
from utils.parent_store import ParentStore


class TenantVectorStore:
    """
    Session-scoped FAISS manager for the Parent-Child retrieval strategy.

    Args:
        session_id: Unique identifier for the current user/browser session.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._vectors_root = os.path.join(STORAGE_PATH, "vectors", session_id)
        self._parent_store = ParentStore(session_id, storage_root=STORAGE_PATH)
        os.makedirs(self._vectors_root, exist_ok=True)

    # ── Path helpers ───────────────────────────────────────────────────────────

    def _index_path(self, doc_id: str) -> str:
        return os.path.join(self._vectors_root, doc_id)

    def exists(self, doc_id: str) -> bool:
        return os.path.exists(
            os.path.join(self._index_path(doc_id), "index.faiss")
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def build(
        self,
        parents: List[ParentChunk],
        children: List[ChildChunk],
        doc_id: str,
    ) -> None:
        """
        Embed all child chunks and persist the FAISS index + parent mapping.

        Args:
            parents:  All ParentChunk objects for this document.
            children: All ChildChunk objects (their .text field has the injected header).
            doc_id:   Unique document identifier.
        """
        # Build LangChain Documents from child chunks (child text + metadata)
        lc_docs: List[Document] = [
            Document(
                page_content=child.text,           # header-injected text
                metadata={
                    "child_id":  child.child_id,
                    "parent_id": child.parent_id,
                    "page":      child.page,
                    "doc_id":    doc_id,
                },
            )
            for child in children
        ]

        embeddings = get_embeddings()
        index_path = self._index_path(doc_id)
        os.makedirs(index_path, exist_ok=True)

        store = FAISS.from_documents(lc_docs, embeddings)
        store.save_local(index_path)

        # Persist the child→parent JSON mapping
        self._parent_store.save(parents, children, doc_id)

    # ── Read ───────────────────────────────────────────────────────────────────

    def _load_index(self, doc_id: str) -> FAISS:
        index_path = self._index_path(doc_id)
        if not os.path.exists(os.path.join(index_path, "index.faiss")):
            raise FileNotFoundError(
                f"No Parent-Child index found for document '{doc_id}'. "
                "Process the PDF first."
            )
        return FAISS.load_local(
            index_path,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    def retrieve_and_package(
        self,
        query: str,
        doc_id: str,
        top_k: int = TOP_K,
    ) -> List[ParentChunk]:
        """
        Full retrieval pipeline:
          1. Embed query → search child FAISS index (top_k hits)
          2. Extract child_ids from result metadata
          3. Look up ParentStore → get full parent text blocks
          4. Deduplicate parents (multiple children → same parent → returned once)

        Returns:
            Deduplicated list of ParentChunk objects, ordered by first match.
        """
        store = self._load_index(doc_id)
        results: List[Document] = store.similarity_search(query, k=top_k)

        child_ids = [
            doc.metadata["child_id"]
            for doc in results
            if "child_id" in doc.metadata
        ]

        return self._parent_store.get_parents_for_children(child_ids, doc_id)

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete_doc(self, doc_id: str) -> None:
        """Remove the FAISS index and parent mapping for *doc_id*."""
        import shutil
        index_path = self._index_path(doc_id)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        self._parent_store.delete(doc_id)
