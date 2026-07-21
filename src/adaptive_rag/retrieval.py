from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from adaptive_rag.embeddings import Qwen3Embeddings


class MedicalRetriever:
    """Search normalized query embeddings inside the medical FAISS index."""

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_model: Qwen3Embeddings,
    ) -> None:
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index was not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"FAISS metadata was not found: {metadata_path}"
            )

        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model = embedding_model

        self.index = faiss.read_index(
            str(self.index_path)
        )

        self.metadata_records = self._load_metadata()

        if self.index.ntotal != len(self.metadata_records):
            raise ValueError(
                "FAISS vector count does not match metadata count: "
                f"{self.index.ntotal} != "
                f"{len(self.metadata_records)}"
            )

    def _load_metadata(
        self,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            for line_number, line in enumerate(
                input_file,
                start=1,
            ):
                if not line.strip():
                    continue

                record = json.loads(line)

                expected_id = len(records)

                if record.get("faiss_id") != expected_id:
                    raise ValueError(
                        f"Unexpected faiss_id at metadata line "
                        f"{line_number}."
                    )

                records.append(record)

        return records

    def search(
        self,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "The retrieval query cannot be empty."
            )

        requested_k = min(
            max(int(k), 1),
            int(self.index.ntotal),
        )

        query_vector = np.asarray(
            [
                self.embedding_model.embed_query(
                    cleaned_query
                )
            ],
            dtype=np.float32,
        )

        query_vector = np.ascontiguousarray(
            query_vector,
            dtype=np.float32,
        )

        expected_shape = (
            1,
            self.index.d,
        )

        if query_vector.shape != expected_shape:
            raise ValueError(
                f"Query embedding shape is "
                f"{query_vector.shape}, but expected "
                f"{expected_shape}."
            )

        faiss.normalize_L2(
            query_vector
        )

        scores, vector_ids = self.index.search(
            query_vector,
            requested_k,
        )

        results: list[dict[str, Any]] = []

        for rank, (
            vector_id,
            score,
        ) in enumerate(
            zip(
                vector_ids[0],
                scores[0],
            ),
            start=1,
        ):
            if vector_id < 0:
                continue

            record = self.metadata_records[
                int(vector_id)
            ]

            source_metadata = record["metadata"]

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "faiss_id": int(vector_id),
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "source_file": source_metadata[
                        "source_file"
                    ],
                    "document_id": source_metadata[
                        "document_id"
                    ],
                    "page_number": source_metadata[
                        "page_number"
                    ],
                }
            )

        return results