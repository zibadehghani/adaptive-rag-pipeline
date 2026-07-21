from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz


def clean_extracted_text(text: str) -> str:
    """Clean common PDF extraction artifacts without removing useful content."""

    text = text.replace("\u00ad", "")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Join words broken across lines: diagno-\nsis → diagnosis
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Normalize whitespace while preserving paragraph boundaries.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_pdf(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract every PDF page and return page records plus a parsing report."""

    page_records: list[dict[str, Any]] = []
    total_characters = 0
    empty_pages = 0

    try:
        with fitz.open(pdf_path) as document:
            total_pages = document.page_count

            for page_index in range(total_pages):
                page = document.load_page(page_index)
                raw_text = page.get_text("text", sort=True)
                cleaned_text = clean_extracted_text(raw_text)

                character_count = len(cleaned_text)
                total_characters += character_count

                if not cleaned_text:
                    empty_pages += 1

                page_records.append(
                    {
                        "document_id": pdf_path.stem,
                        "source_file": pdf_path.name,
                        "page_number": page_index + 1,
                        "text": cleaned_text,
                        "character_count": character_count,
                    }
                )

        report = {
            "source_file": pdf_path.name,
            "status": "success",
            "page_count": total_pages,
            "empty_pages": empty_pages,
            "total_characters": total_characters,
            "error": "",
        }

        return page_records, report

    except Exception as exc:
        report = {
            "source_file": pdf_path.name,
            "status": "failed",
            "page_count": 0,
            "empty_pages": 0,
            "total_characters": 0,
            "error": str(exc),
        }

        return [], report