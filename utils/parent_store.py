"""
parent_store.py — Persistent child→parent mapping for the Parent-Child RAG strategy.

Storage layout:
    storage/parents/{session_id}/{doc_id}.json

Each JSON file is a flat dict:
    {
        "<child_id>": {
            "parent_id":   str,
            "parent_text": str,
            "doc_name":    str,
            "page":        int,
            "child_ids":   [str, ...]   # all siblings in the same parent
        },
        ...
    }

This flat structure allows O(1) child→parent lookup without any secondary index.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from utils.hierarchical_chunker import ChildChunk, ParentChunk


class ParentStore:
    """
    JSON-backed store that maps child_id → parent block.

    One store instance per (session_id, doc_id) pair.
    """

    def __init__(self, session_id: str, storage_root: str = "storage") -> None:
        self._session_id = session_id
        self._root = os.path.join(storage_root, "parents", session_id)
        os.makedirs(self._root, exist_ok=True)

    # ── Path helpers ───────────────────────────────────────────────────────────

    def _path(self, doc_id: str) -> str:
        return os.path.join(self._root, f"{doc_id}.json")

    # ── Write ──────────────────────────────────────────────────────────────────

    def save(
        self,
        parents: List[ParentChunk],
        children: List[ChildChunk],
        doc_id: str,
    ) -> None:
        """
        Persist the child→parent mapping for *doc_id*.

        Overwrites any existing mapping for this document.
        """
        # Build a lookup: parent_id → ParentChunk
        parent_by_id: Dict[str, ParentChunk] = {p.parent_id: p for p in parents}

        mapping: Dict[str, dict] = {}
        for child in children:
            parent = parent_by_id.get(child.parent_id)
            if parent is None:
                continue
            mapping[child.child_id] = {
                "parent_id":   parent.parent_id,
                "parent_text": parent.text,
                "doc_name":    parent.doc_name,
                "page":        parent.page,
                "child_ids":   parent.child_ids,
            }

        with open(self._path(doc_id), "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    # ── Read ───────────────────────────────────────────────────────────────────

    def _load(self, doc_id: str) -> Dict[str, dict]:
        path = self._path(doc_id)
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def get_parent(self, child_id: str, doc_id: str) -> Optional[ParentChunk]:
        """Return the ParentChunk for a single child_id, or None if not found."""
        mapping = self._load(doc_id)
        entry = mapping.get(child_id)
        if entry is None:
            return None
        return ParentChunk(
            parent_id=entry["parent_id"],
            doc_id=doc_id,
            doc_name=entry["doc_name"],
            page=entry["page"],
            text=entry["parent_text"],
            child_ids=entry.get("child_ids", []),
        )

    def get_parents_for_children(
        self,
        child_ids: List[str],
        doc_id: str,
    ) -> List[ParentChunk]:
        """
        Return deduplicated ParentChunks for a list of child_ids.

        If multiple children map to the same parent, that parent is returned once.
        Order follows first-seen child_id order (preserves retrieval ranking).
        """
        mapping = self._load(doc_id)
        seen_parents: Dict[str, ParentChunk] = {}

        for child_id in child_ids:
            entry = mapping.get(child_id)
            if entry is None:
                continue
            parent_id = entry["parent_id"]
            if parent_id not in seen_parents:
                seen_parents[parent_id] = ParentChunk(
                    parent_id=parent_id,
                    doc_id=doc_id,
                    doc_name=entry["doc_name"],
                    page=entry["page"],
                    text=entry["parent_text"],
                    child_ids=entry.get("child_ids", []),
                )

        return list(seen_parents.values())

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete(self, doc_id: str) -> None:
        """Remove the parent mapping JSON for *doc_id* if it exists."""
        path = self._path(doc_id)
        if os.path.exists(path):
            os.remove(path)

    def exists(self, doc_id: str) -> bool:
        return os.path.exists(self._path(doc_id))
