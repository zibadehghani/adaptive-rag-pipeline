from __future__ import annotations

import json

import pandas as pd

from adaptive_rag.config import (
    PAGES_JSONL_PATH,
    PARSING_REPORT_PATH,
    PROCESSED_DIR,
    RAW_PDF_DIR,
)
from adaptive_rag.pdf_parser import parse_pdf


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in: {RAW_PDF_DIR}"
        )

    all_pages: list[dict] = []
    reports: list[dict] = []

    print("=" * 80)
    print("Adaptive RAG — PDF Dataset Preparation")
    print("=" * 80)
    print(f"PDF files found: {len(pdf_files)}\n")

    for pdf_path in pdf_files:
        page_records, report = parse_pdf(pdf_path)

        all_pages.extend(page_records)
        reports.append(report)

        print(
            f"[{report['status'].upper():7}] "
            f"{pdf_path.name} | "
            f"pages={report['page_count']} | "
            f"characters={report['total_characters']}"
        )

    with PAGES_JSONL_PATH.open("w", encoding="utf-8") as output_file:
        for page_record in all_pages:
            output_file.write(
                json.dumps(page_record, ensure_ascii=False) + "\n"
            )

    pd.DataFrame(reports).to_csv(
        PARSING_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    successful_files = sum(
        report["status"] == "success" for report in reports
    )

    print("\n" + "=" * 80)
    print(f"Successful PDFs: {successful_files}/{len(pdf_files)}")
    print(f"Extracted pages: {len(all_pages)}")
    print(f"Pages JSONL: {PAGES_JSONL_PATH}")
    print(f"Parsing report: {PARSING_REPORT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()