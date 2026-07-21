from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_page_documents(jsonl_path: Path) -> list[Document]:
    """Load extracted PDF pages from JSONL as LangChain Documents."""

    documents: list[Document] = []

    with jsonl_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)

            text = record["text"].strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "document_id": record["document_id"],
                        "source_file": record["source_file"],
                        "page_number": record["page_number"],
                    },
                )
            )

    return documents


def create_chunks(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Split page documents while preserving source metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        length_function=len,
        is_separator_regex=False,
    )

    split_documents = splitter.split_documents(documents)

    page_chunk_counters: defaultdict[tuple[str, int], int] = defaultdict(int)
    chunk_records: list[dict[str, Any]] = []

    for document in split_documents:
        document_id = document.metadata["document_id"]
        page_number = document.metadata["page_number"]

        counter_key = (document_id, page_number)
        page_chunk_counters[counter_key] += 1

        page_chunk_number = page_chunk_counters[counter_key]

        chunk_id = (
            f"{document_id}"
            f"_page_{page_number:03d}"
            f"_chunk_{page_chunk_number:03d}"
        )

        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "text": document.page_content.strip(),
                "character_count": len(document.page_content.strip()),
                "metadata": {
                    **document.metadata,
                    "page_chunk_number": page_chunk_number,
                },
            }
        )

    return chunk_records