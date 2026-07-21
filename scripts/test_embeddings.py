from __future__ import annotations

import numpy as np

from adaptive_rag.config import EMBEDDING_DIMENSION
from adaptive_rag.embeddings import Qwen3Embeddings


def main() -> None:
    embedding_model = Qwen3Embeddings()

    query = (
        "How does an expert system select an appropriate "
        "treatment method for warts?"
    )

    passages = [
        (
            "A fuzzy expert system evaluates patient characteristics "
            "and wart properties to recommend a suitable treatment method."
        ),
        (
            "Artificial intelligence can support the detection of "
            "mental health risks during pregnancy."
        ),
    ]

    query_vector = np.asarray(
        embedding_model.embed_query(query),
        dtype=np.float32,
    )

    passage_vectors = np.asarray(
        embedding_model.embed_documents(passages),
        dtype=np.float32,
    )

    similarity_scores = passage_vectors @ query_vector

    print("=" * 80)
    print("Qwen3 Embedding 0.6B — Online Model Test")
    print("=" * 80)
    print(f"Model ID: {embedding_model.model_id}")
    print(f"Cache directory: {embedding_model.cache_dir}")
    print(f"Device: {embedding_model.device}")
    print(f"Embedding dimension: {query_vector.shape[0]}")
    print()

    for index, score in enumerate(
        similarity_scores,
        start=1,
    ):
        print(f"Passage {index} similarity: {score:.4f}")

    print("=" * 80)

    dimension_is_correct = (
        query_vector.shape[0] == EMBEDDING_DIMENSION
    )

    ranking_is_correct = (
        similarity_scores[0] > similarity_scores[1]
    )

    if dimension_is_correct and ranking_is_correct:
        print(
            "TEST PASSED: The model produced the expected dimension "
            "and ranked the relevant passage higher."
        )
    else:
        if not dimension_is_correct:
            print(
                "TEST WARNING: Unexpected embedding dimension. "
                f"Expected {EMBEDDING_DIMENSION}, "
                f"received {query_vector.shape[0]}."
            )

        if not ranking_is_correct:
            print(
                "TEST WARNING: The unrelated passage received "
                "a higher similarity score."
            )


if __name__ == "__main__":
    main()