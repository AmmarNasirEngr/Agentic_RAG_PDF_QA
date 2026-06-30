"""
layout_extractor.py — Layout-aware PDF extraction using PyMuPDF + pymupdf4llm.

Produces per-page Markdown that preserves:
  • Paragraph / heading structure (pymupdf4llm handles font-size heuristics)
  • Tables (detected via fitz.Page.find_tables, rendered as GFM Markdown)
  • Page numbers in metadata

Raises ValueError for scanned/image-only PDFs with no extractable text.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import fitz  # PyMuPDF


def _table_to_markdown(table) -> str:
    """Convert a PyMuPDF Table object to a GitHub-Flavored Markdown table string."""
    rows = table.extract()
    if not rows:
        return ""

    # Normalise cells: replace None with empty string, flatten newlines
    def clean(cell: Any) -> str:
        return str(cell).replace("\n", " ").strip() if cell is not None else ""

    header = rows[0]
    body = rows[1:]

    md_header = "| " + " | ".join(clean(c) for c in header) + " |"
    md_sep = "| " + " | ".join("---" for _ in header) + " |"
    md_rows = [
        "| " + " | ".join(clean(c) for c in row) + " |"
        for row in body
    ]
    return "\n".join([md_header, md_sep] + md_rows)


def extract_layout(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text and tables from a PDF page-by-page, returning Markdown per page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of page dicts:
            {
                "page": int,          1-indexed page number
                "text": str,          Full page Markdown (text + inline tables)
                "has_table": bool,    True if at least one table was found
            }

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the PDF has no extractable text layer (scanned/image-only).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[Dict[str, Any]] = []
    total_chars = 0

    doc = fitz.open(pdf_path)
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1

        # ── Detect tables ─────────────────────────────────────────────────────
        try:
            tables = page.find_tables()
            table_list = list(tables)
        except Exception:
            table_list = []

        has_table = len(table_list) > 0

        # ── Build page Markdown ───────────────────────────────────────────────
        if has_table:
            # Extract text blocks and splice in Markdown tables at the right positions
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            text_blocks = [
                b for b in page_dict.get("blocks", []) if b.get("type") == 0
            ]

            # Collect table bounding boxes to mask them from plain text
            table_bboxes = [fitz.Rect(t.bbox) for t in table_list]

            text_lines: List[str] = []
            table_inserted = set()

            for block in text_blocks:
                block_rect = fitz.Rect(block["bbox"])
                # Check if this block overlaps any table
                overlapping = [
                    i for i, tb in enumerate(table_bboxes)
                    if block_rect.intersects(tb)
                ]
                if overlapping:
                    # Insert the table Markdown once at the first overlapping block
                    for i in overlapping:
                        if i not in table_inserted:
                            md = _table_to_markdown(table_list[i])
                            if md:
                                text_lines.append(md)
                            table_inserted.add(i)
                else:
                    # Plain text block
                    block_text = " ".join(
                        span.get("text", "")
                        for line in block.get("lines", [])
                        for span in line.get("spans", [])
                    ).strip()
                    if block_text:
                        text_lines.append(block_text)

            page_md = "\n\n".join(text_lines)
        else:
            # No tables — use simple plain text extraction (fast path)
            page_md = page.get_text().strip()

        if page_md:
            total_chars += len(page_md)
            pages.append({"page": page_num, "text": page_md, "has_table": has_table})

    doc.close()

    if total_chars < 100:
        raise ValueError(
            "No readable text found in this PDF. "
            "It appears to be a scanned image — try an OCR-processed version."
        )

    return pages
