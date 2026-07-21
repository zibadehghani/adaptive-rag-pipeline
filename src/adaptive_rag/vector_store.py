from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


def load_chunk_records(
    chunks_path: Path,
) -> list[dict[str, Any]]:
    """Load valid chunk records from a JSONL file."""

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk dataset was not found: {chunks_path}"
        )

    records: list[dict[str, Any]] = []

    with chunks_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("text", "").strip()

            if not text:
                continue

            if "chunk_id" not in record:
                raise ValueError(
                    f"Missing chunk_id at JSONL line {line_number}."
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No valid chunk records were found in: {chunks_path}"
        )

    return records


def calculate_file_signature(
    file_path: Path,
) -> str:
    """Calculate a SHA-256 signature for detecting dataset changes."""

    digest = hashlib.sha256()

    with file_path.open("rb") as input_file:
        for block in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_checkpoint(
    checkpoint_path: Path,
    state_path: Path,
    expected_total: int,
    expected_dimension: int,
    expected_signature: str,
) -> tuple[np.ndarray, int]:
    """Load an existing checkpoint or return an empty embedding matrix."""

    empty_vectors = np.zeros(
        shape=(expected_total, expected_dimension),
        dtype=np.float32,
    )

    checkpoint_exists = checkpoint_path.exists()
    state_exists = state_path.exists()

    if not checkpoint_exists and not state_exists:
        return empty_vectors, 0

    if checkpoint_exists != state_exists:
        raise RuntimeError(
            "Checkpoint files are incomplete. Both the NumPy file "
            "and state JSON file must exist."
        )

    vectors = np.load(
        checkpoint_path,
        allow_pickle=False,
    )

    with state_path.open("r", encoding="utf-8") as input_file:
        state = json.load(input_file)

    expected_shape = (
        expected_total,
        expected_dimension,
    )

    if vectors.shape != expected_shape:
        raise ValueError(
            f"Checkpoint shape is {vectors.shape}, "
            f"but expected {expected_shape}."
        )

    completed_chunks = int(
        state.get("completed_chunks", 0)
    )

    if not 0 <= completed_chunks <= expected_total:
        raise ValueError(
            f"Invalid completed_chunks value: {completed_chunks}"
        )

    stored_signature = state.get("chunks_signature")

    if (
        stored_signature is not None
        and stored_signature != expected_signature
    ):
        raise ValueError(
            "The Chunk dataset has changed since this checkpoint "
            "was created."
        )

    return vectors.astype(np.float32), completed_chunks


def save_checkpoint(
    vectors: np.ndarray,
    completed_chunks: int,
    total_chunks: int,
    model_id: str,
    chunks_signature: str,
    checkpoint_path: Path,
    state_path: Path,
) -> None:
    """Safely save vectors and checkpoint state."""

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_checkpoint_path = checkpoint_path.with_suffix(
        ".tmp"
    )

    with temporary_checkpoint_path.open("wb") as output_file:
        np.save(
            output_file,
            vectors,
            allow_pickle=False,
        )

    temporary_checkpoint_path.replace(
        checkpoint_path
    )

    state = {
        "status": (
            "complete"
            if completed_chunks == total_chunks
            else "in_progress"
        ),
        "embedding_model": model_id,
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "next_chunk_index": completed_chunks,
        "vector_dimension": int(vectors.shape[1]),
        "chunks_signature": chunks_signature,
        "checkpoint_path": str(checkpoint_path),
    }

    temporary_state_path = state_path.with_suffix(
        ".tmp"
    )

    with temporary_state_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            state,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_state_path.replace(
        state_path
    )


def build_faiss_index_from_vectors(
    vectors: np.ndarray,
    expected_dimension: int,
) -> faiss.Index:
    """Build an exact cosine-similarity index from prepared vectors."""

    matrix = np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    ).copy()

    if matrix.ndim != 2:
        raise ValueError(
            "The embedding checkpoint must be two-dimensional."
        )

    if matrix.shape[1] != expected_dimension:
        raise ValueError(
            f"Expected dimension {expected_dimension}, "
            f"but received {matrix.shape[1]}."
        )

    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(
        expected_dimension
    )

    index.add(matrix)

    return index


def save_faiss_artifacts(
    index: faiss.Index,
    records: list[dict[str, Any]],
    index_path: Path,
    metadata_path: Path,
) -> None:
    """Save the FAISS index and its ordered Chunk metadata."""

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(index_path),
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for faiss_id, record in enumerate(records):
            metadata_record = {
                "faiss_id": faiss_id,
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "character_count": record["character_count"],
                "metadata": record["metadata"],
            }

            output_file.write(
                json.dumps(
                    metadata_record,
                    ensure_ascii=False,
                )
                + "\n"
            )