from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from adaptive_rag.config import (
    CHECKPOINT_SAVE_EVERY,
    CHECKPOINT_STATE_PATH,
    CHUNKS_JSONL_PATH,
    EMBEDDING_DIMENSION,
    EMBEDDINGS_CHECKPOINT_PATH,
    FAISS_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    FAISS_REPORT_PATH,
    QWEN_MODEL_ID,
)
from adaptive_rag.embeddings import Qwen3Embeddings
from adaptive_rag.vector_store import (
    build_faiss_index_from_vectors,
    calculate_file_signature,
    load_checkpoint,
    load_chunk_records,
    save_checkpoint,
    save_faiss_artifacts,
)


def main() -> None:
    FAISS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Adaptive RAG — Resumable FAISS Index Construction")
    print("=" * 80)

    chunk_records = load_chunk_records(
        CHUNKS_JSONL_PATH
    )

    total_chunks = len(chunk_records)

    source_files = {
        record["metadata"]["source_file"]
        for record in chunk_records
    }

    chunks_signature = calculate_file_signature(
        CHUNKS_JSONL_PATH
    )

    vectors, completed_chunks = load_checkpoint(
        checkpoint_path=EMBEDDINGS_CHECKPOINT_PATH,
        state_path=CHECKPOINT_STATE_PATH,
        expected_total=total_chunks,
        expected_dimension=EMBEDDING_DIMENSION,
        expected_signature=chunks_signature,
    )

    initial_completed_chunks = completed_chunks
    device = "checkpoint"

    print(f"Source articles: {len(source_files)}")
    print(f"Input chunks: {total_chunks}")
    print(
        f"Checkpoint progress: "
        f"{completed_chunks}/{total_chunks}"
    )

    if completed_chunks < total_chunks:
        print("Loading Qwen3 Embedding model...")

        embedding_model = Qwen3Embeddings()
        device = embedding_model.device

        print(f"Device: {device}")
        print("Generating remaining embeddings...")

        for start_index in range(
            completed_chunks,
            total_chunks,
            CHECKPOINT_SAVE_EVERY,
        ):
            end_index = min(
                start_index + CHECKPOINT_SAVE_EVERY,
                total_chunks,
            )

            batch_texts = [
                record["text"]
                for record in chunk_records[
                    start_index:end_index
                ]
            ]

            batch_vectors = np.asarray(
                embedding_model.embed_documents(
                    batch_texts
                ),
                dtype=np.float32,
            )

            expected_batch_shape = (
                end_index - start_index,
                EMBEDDING_DIMENSION,
            )

            if batch_vectors.shape != expected_batch_shape:
                raise ValueError(
                    f"Batch embedding shape is "
                    f"{batch_vectors.shape}, but expected "
                    f"{expected_batch_shape}."
                )

            vectors[
                start_index:end_index
            ] = batch_vectors

            completed_chunks = end_index

            save_checkpoint(
                vectors=vectors,
                completed_chunks=completed_chunks,
                total_chunks=total_chunks,
                model_id=QWEN_MODEL_ID,
                chunks_signature=chunks_signature,
                checkpoint_path=EMBEDDINGS_CHECKPOINT_PATH,
                state_path=CHECKPOINT_STATE_PATH,
            )

            print(
                f"Checkpoint saved: "
                f"{completed_chunks}/{total_chunks}"
            )

    else:
        print(
            "Complete checkpoint found. "
            "Embedding generation was skipped."
        )

    save_checkpoint(
        vectors=vectors,
        completed_chunks=total_chunks,
        total_chunks=total_chunks,
        model_id=QWEN_MODEL_ID,
        chunks_signature=chunks_signature,
        checkpoint_path=EMBEDDINGS_CHECKPOINT_PATH,
        state_path=CHECKPOINT_STATE_PATH,
    )

    print("Building FAISS index from checkpoint...")

    index = build_faiss_index_from_vectors(
        vectors=vectors,
        expected_dimension=EMBEDDING_DIMENSION,
    )

    save_faiss_artifacts(
        index=index,
        records=chunk_records,
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH,
    )

    report = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "embedding_model": QWEN_MODEL_ID,
        "device": device,
        "source_article_count": len(source_files),
        "chunk_count": total_chunks,
        "vector_count": int(index.ntotal),
        "vector_dimension": EMBEDDING_DIMENSION,
        "index_type": "IndexFlatIP",
        "similarity_metric": (
            "cosine similarity using normalized inner product"
        ),
        "resumed_from_chunk": initial_completed_chunks,
        "checkpoint_save_every": CHECKPOINT_SAVE_EVERY,
        "checkpoint_path": str(
            EMBEDDINGS_CHECKPOINT_PATH
        ),
        "checkpoint_state_path": str(
            CHECKPOINT_STATE_PATH
        ),
        "index_path": str(FAISS_INDEX_PATH),
        "metadata_path": str(
            FAISS_METADATA_PATH
        ),
    }

    with FAISS_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            report,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("RESUMABLE FAISS INDEX CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"Stored vectors: {index.ntotal}")
    print(f"Vector dimension: {EMBEDDING_DIMENSION}")
    print(
        f"Resumed from Chunk: "
        f"{initial_completed_chunks}"
    )
    print(f"Index file: {FAISS_INDEX_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()