"""
hierarchical_chunker.py — Parent-Child recursive text splitting.

Strategy:
  1. Parent splitter   (chunk_size=PARENT_CHUNK_SIZE, ~6 000 chars)
     Splits full-page Markdown into large, semantically coherent blocks.
  2. Child splitter    (chunk_size=CHILD_CHUNK_SIZE, ~1 000 chars, 200 overlap)
     Sub-divides each parent into small, precise retrieval units.
  3. Header injection  before each child:
     "[Doc: <name> | Page: <n>]\n\n<child_text>"

Why two levels?
  • Child vectors are small → precise embedding match at query time.
  • Parent text is large → LLM receives full context, not a fragment.
  This directly solves the "context split problem": even if a child chunk is
  retrieved at a section boundary, the LLM sees the whole parent block.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ParentChunk:
    parent_id: str
    doc_id: str
    doc_name: str
    page: int
    text: str
    child_ids: List[str] = field(default_factory=list)


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    doc_id: str
    doc_name: str
    page: int
    raw_text: str      # child text WITHOUT the injected header
    text: str          # child text WITH the injected header (what gets embedded)


# ── Splitters ─────────────────────────────────────────────────────────────────

def _parent_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _child_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ── Public API ────────────────────────────────────────────────────────────────

def build_hierarchy(
    pages: List[Dict[str, Any]],
    doc_id: str,
    doc_name: str,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """
    Build the two-level Parent-Child chunk hierarchy from extracted page dicts.

    Args:
        pages:    Output of layout_extractor.extract_layout()  — list of
                  {"page": int, "text": str, "has_table": bool}
        doc_id:   Unique document identifier.
        doc_name: Original filename (used in child header injection).

    Returns:
        (parents, children) — all ParentChunk and ChildChunk objects.
    """
    ps = _parent_splitter()
    cs = _child_splitter()

    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []

    for page_data in pages:
        page_num: int = page_data["page"]
        page_text: str = page_data["text"]

        # ── Parent split ──────────────────────────────────────────────────────
        parent_texts = ps.split_text(page_text)

        for p_text in parent_texts:
            if not p_text.strip():
                continue

            parent_id = uuid.uuid4().hex[:8]
            parent = ParentChunk(
                parent_id=parent_id,
                doc_id=doc_id,
                doc_name=doc_name,
                page=page_num,
                text=p_text,
            )

            # ── Child split ───────────────────────────────────────────────────
            child_texts = cs.split_text(p_text)

            for c_text in child_texts:
                if not c_text.strip():
                    continue

                child_id = uuid.uuid4().hex[:8]

                # Inject grounding header so the embedding reflects document context
                header = f"[Doc: {doc_name} | Page: {page_num}]"
                embedded_text = f"{header}\n\n{c_text}"

                child = ChildChunk(
                    child_id=child_id,
                    parent_id=parent_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=page_num,
                    raw_text=c_text,
                    text=embedded_text,
                )
                children.append(child)
                parent.child_ids.append(child_id)

            if parent.child_ids:   # only keep parents that have at least one child
                parents.append(parent)

    return parents, children
