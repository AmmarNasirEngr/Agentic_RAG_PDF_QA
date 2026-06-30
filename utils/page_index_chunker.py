"""
page_index_chunker.py — PageIndex RAG chunking strategy.

Builds a section tree from the PDF's visual structure (font sizes, line-level
bold detection), then chunks content by section.  Every chunk carries
section-level metadata so the LLM receives rich context: section title,
hierarchy level, and page range.

Inspired by: "Traditional RAG vs PageIndex RAG" — Ammar Nasir.

Key insight — heading detection is LINE-level, not span-level:
    • A line where ALL spans are bold → heading candidate
    • A line where only SOME spans are bold → body text with inline emphasis
    This prevents mistaking emphasised words ("without any") for headings.

Flow:
    PDF  →  line-level blocks  (PyMuPDF dict mode)
         →  body-size estimation  (median of line max-sizes)
         →  heading detection  (size delta + all-bold + length heuristics)
         →  section tree  [{title, level, page_start, page_end, text}]
         →  sub-chunk each section  (RecursiveCharacterTextSplitter)
         →  LangChain Documents with section metadata

Fallback: if fewer than 3 headings are found the document lacks detectable
structure and the function falls back to standard fixed-size chunking.
"""

import re
from collections import defaultdict
from statistics import median
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_lines(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Return every non-empty line from the PDF with line-level font metadata.

    Each item:
        text      — full line text (all spans joined)
        page      — 1-indexed page number
        size      — max font size across all spans in this line
        all_bold  — True only when EVERY span in the line is bold
    """
    result: List[Dict[str, Any]] = []
    doc = fitz.open(pdf_path)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:          # skip image blocks
                continue
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue

                line_text = " ".join(s["text"].strip() for s in spans)
                all_bold = all(bool(s.get("flags", 0) & 16) for s in spans)
                max_size = max(s.get("size", 12.0) for s in spans)

                result.append({
                    "text": line_text.strip(),
                    "page": page_idx + 1,
                    "size": round(max_size, 1),
                    "all_bold": all_bold,
                })

    doc.close()
    return result


def _body_font_size(lines: List[Dict]) -> float:
    """Estimate body text size as the median of all line sizes."""
    sizes = [ln["size"] for ln in lines if ln["size"] > 0]
    return median(sizes) if sizes else 12.0


def _heading_level(line: Dict, body_size: float) -> Optional[int]:
    """
    Return heading level 1–3, or None if this line is body text.

    Rules (line-level — all_bold means every span in the line is bold):
      H1 — size >= body+6, OR size >= body+4 AND all_bold
      H2 — size >= body+3, OR size >= body+1.5 AND all_bold
      H3 — all_bold AND size >= body AND len < 80 AND not a sentence fragment
    """
    text = line["text"].strip()
    size_delta = line["size"] - body_size
    all_bold = line["all_bold"]

    # Reject implausible headings
    if len(text) < 2 or len(text) > 120:
        return None
    # Pure numbers / punctuation (page numbers, list markers)
    if re.fullmatch(r'[\d\s.\-\(\)/\\]+', text):
        return None

    if size_delta >= 6 or (size_delta >= 4 and all_bold):
        return 1
    if size_delta >= 3 or (size_delta >= 1.5 and all_bold):
        return 2
    # Bold-only heading: must occupy its own line (all_bold) and not look like
    # a sentence fragment (doesn't end mid-word / mid-clause with a comma etc.)
    if all_bold and size_delta >= 0 and len(text) < 80:
        # Exclude fragments that end with comma, ellipsis, or opening parenthesis
        if not re.search(r'[,…(]$', text):
            return 3
    return None


def _build_sections(lines: List[Dict], body_size: float) -> List[Dict[str, Any]]:
    """
    Walk lines in reading order and group them into sections by heading.

    Returns:
        [{title, level, page_start, page_end, text}]
    """
    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {
        "title": "Preamble",
        "level": 1,
        "page_start": lines[0]["page"] if lines else 1,
        "page_end": lines[0]["page"] if lines else 1,
        "body": [],
    }

    for ln in lines:
        level = _heading_level(ln, body_size)
        if level is not None:
            if current["body"]:            # flush non-empty section
                sections.append({
                    "title": current["title"],
                    "level": current["level"],
                    "page_start": current["page_start"],
                    "page_end": ln["page"],
                    "text": " ".join(current["body"]),
                })
            current = {
                "title": ln["text"].strip(),
                "level": level,
                "page_start": ln["page"],
                "page_end": ln["page"],
                "body": [],
            }
        else:
            current["body"].append(ln["text"])
            current["page_end"] = ln["page"]

    # Flush final section
    if current["body"]:
        sections.append({
            "title": current["title"],
            "level": current["level"],
            "page_start": current["page_start"],
            "page_end": current["page_end"],
            "text": " ".join(current["body"]),
        })

    return sections


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_by_page_index(pdf_path: str, doc_id: str) -> List[Document]:
    """
    PageIndex RAG chunking: detect sections from PDF structure, sub-chunk each.

    Each returned Document carries:
        page                — first page of its section
        section_title       — heading that introduced this section
        section_level       — 1 (major), 2 (sub), 3 (minor)
        section_page_start / section_page_end
        chunking_strategy   — "page_index" or "fixed" (fallback)
        chunk_id            — sequential index across the document

    Falls back to fixed-size chunking when fewer than 3 headings are detected.
    """
    lines = _extract_lines(pdf_path)
    if not lines:
        return []

    body_size = _body_font_size(lines)
    sections = _build_sections(lines, body_size)

    # Require at least 3 sections with real body content (any heading level)
    meaningful = [s for s in sections if len(s["text"]) > 30]
    if len(meaningful) < 3:
        return _fixed_fallback(lines, doc_id)

    splitter = _make_splitter()
    documents: List[Document] = []
    chunk_id = 0

    for section in sections:
        body = section["text"].strip()
        if not body:
            continue

        # Prepend section title so every sub-chunk is self-contained
        labelled = f"[Section: {section['title']}]\n{body}"
        for sub in splitter.split_text(labelled):
            if sub.strip():
                documents.append(Document(
                    page_content=sub,
                    metadata={
                        "doc_id": doc_id,
                        "page": section["page_start"],
                        "chunk_id": chunk_id,
                        "section_title": section["title"],
                        "section_level": section["level"],
                        "section_page_start": section["page_start"],
                        "section_page_end": section["page_end"],
                        "chunking_strategy": "page_index",
                    },
                ))
                chunk_id += 1

    return documents


def _fixed_fallback(lines: List[Dict], doc_id: str) -> List[Document]:
    """Standard fixed-size chunking used when no document structure is found."""
    page_texts: Dict[int, List[str]] = defaultdict(list)
    for ln in lines:
        page_texts[ln["page"]].append(ln["text"])

    splitter = _make_splitter()
    documents: List[Document] = []
    chunk_id = 0

    for page_num in sorted(page_texts):
        text = " ".join(page_texts[page_num])
        for sub in splitter.split_text(text):
            if sub.strip():
                documents.append(Document(
                    page_content=sub,
                    metadata={
                        "doc_id": doc_id,
                        "page": page_num,
                        "chunk_id": chunk_id,
                        "chunking_strategy": "fixed",
                    },
                ))
                chunk_id += 1

    return documents
