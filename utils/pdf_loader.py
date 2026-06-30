"""
pdf_loader.py — Extract text from PDF files using PyMuPDF.

Each page is returned as a dict so downstream code keeps page-number metadata.
"""

import os
from typing import List, Dict

import fitz  # PyMuPDF


def load_pdf(file_path: str) -> List[Dict[str, object]]:
    """
    Open a PDF and extract text page by page.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        A list of dicts, one per non-empty page:
            {"text": str, "page": int}   (page is 1-indexed)

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no text can be extracted (e.g. scanned image PDF).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages: List[Dict[str, object]] = []

    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text: str = page.get_text()
            if text.strip():  # skip blank / image-only pages
                pages.append({"text": text, "page": page_num + 1})
        doc.close()
    except Exception as exc:
        raise ValueError(f"Failed to read PDF: {exc}") from exc

    if not pages:
        raise ValueError(
            "No readable text found in this PDF. "
            "It may be a scanned image — try an OCR-processed version."
        )

    return pages
