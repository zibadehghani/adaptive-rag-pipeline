from __future__ import annotations

import json
from datetime import datetime, timezone

import faiss
import numpy as np

from adaptive_rag.config import (
    CHECKPOINT_STATE_PATH,
    EMBEDDINGS_CHECKPOINT_PATH,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    QWEN_MODEL_ID,
)


def count_jsonl_records() -> int:
    with FAISS_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        return sum(
            1
            for line in input_file
            if line.strip()
        )


def reconstruct_vectors(
    index: faiss.Index,
) -> np.ndarray:
    vectors = np.empty(
        shape=(index.ntotal, index.d),
        dtype=np.float32,
    )

    for vector_id in range(index.ntotal):
        vectors[vector_id] = index.reconstruct(
            vector_id
        )

    return vectors


def main() -> None:
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index was not found: {FAISS_INDEX_PATH}"
        )

    if not FAISS_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file was not found: {FAISS_METADATA_PATH}"
        )

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    metadata_count = count_jsonl_records()

    if index.ntotal != metadata_count:
        raise ValueError(
            "FAISS vector count does not match metadata count: "
            f"{index.ntotal} != {metadata_count}"
        )

    print("=" * 80)
    print("Adaptive RAG — Checkpoint Extraction")
    print("=" * 80)
    print(f"Vectors in FAISS: {index.ntotal}")
    print(f"Vector dimension: {index.d}")
    print("Reconstructing stored vectors...")

    vectors = reconstruct_vectors(index)

    np.save(
        EMBEDDINGS_CHECKPOINT_PATH,
        vectors,
        allow_pickle=False,
    )

    checkpoint_state = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "complete",
        "embedding_model": QWEN_MODEL_ID,
        "completed_chunks": int(index.ntotal),
        "total_chunks": metadata_count,
        "next_chunk_index": metadata_count,
        "vector_dimension": int(index.d),
        "checkpoint_path": str(
            EMBEDDINGS_CHECKPOINT_PATH
        ),
        "metadata_path": str(
            FAISS_METADATA_PATH
        ),
    }

    with CHECKPOINT_STATE_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            checkpoint_state,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    loaded_vectors = np.load(
        EMBEDDINGS_CHECKPOINT_PATH,
        allow_pickle=False,
    )

    if loaded_vectors.shape != vectors.shape:
        raise RuntimeError(
            "Saved checkpoint shape does not match "
            "the reconstructed vectors."
        )

    print()
    print("=" * 80)
    print("CHECKPOINT CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"Checkpoint shape: {loaded_vectors.shape}")
    print(f"Checkpoint dtype: {loaded_vectors.dtype}")
    print(f"Vectors file: {EMBEDDINGS_CHECKPOINT_PATH}")
    print(f"State file: {CHECKPOINT_STATE_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()