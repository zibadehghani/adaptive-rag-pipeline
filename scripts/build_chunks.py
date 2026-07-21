from __future__ import annotations

import json

import pandas as pd

from adaptive_rag.chunking import create_chunks, load_page_documents
from adaptive_rag.config import (
    CHUNK_OVERLAP,
    CHUNK_REPORT_PATH,
    CHUNK_SIZE,
    CHUNKS_DIR,
    CHUNKS_JSONL_PATH,
    PAGES_JSONL_PATH,
)


def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if not PAGES_JSONL_PATH.exists():
        raise FileNotFoundError(
            f"Extracted page dataset was not found: {PAGES_JSONL_PATH}"
        )

    page_documents = load_page_documents(PAGES_JSONL_PATH)

    chunk_records = create_chunks(
        documents=page_documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    with CHUNKS_JSONL_PATH.open("w", encoding="utf-8") as output_file:
        for record in chunk_records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    report_rows = []

    for source_file in sorted(
        {
            record["metadata"]["source_file"]
            for record in chunk_records
        }
    ):
        source_chunks = [
            record
            for record in chunk_records
            if record["metadata"]["source_file"] == source_file
        ]

        character_counts = [
            record["character_count"]
            for record in source_chunks
        ]

        report_rows.append(
            {
                "source_file": source_file,
                "chunk_count": len(source_chunks),
                "average_characters": round(
                    sum(character_counts) / len(character_counts),
                    2,
                ),
                "minimum_characters": min(character_counts),
                "maximum_characters": max(character_counts),
            }
        )

    pd.DataFrame(report_rows).to_csv(
        CHUNK_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("Adaptive RAG — Document Chunking")
    print("=" * 80)
    print(f"Input page documents: {len(page_documents)}")
    print(f"Generated chunks: {len(chunk_records)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Chunks JSONL: {CHUNKS_JSONL_PATH}")
    print(f"Chunking report: {CHUNK_REPORT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()