from __future__ import annotations

from adaptive_rag.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    RETRIEVAL_TOP_K,
)
from adaptive_rag.embeddings import Qwen3Embeddings
from adaptive_rag.retrieval import MedicalRetriever


def main() -> None:
    query = (
        "How does a fuzzy expert system select "
        "a wart treatment method?"
    )

    print("=" * 80)
    print("Adaptive RAG — Direct Retrieval Test")
    print("=" * 80)
    print(f"Query: {query}")
    print("Loading embedding model and FAISS index...")

    embedding_model = Qwen3Embeddings()

    retriever = MedicalRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH,
        embedding_model=embedding_model,
    )

    results = retriever.search(
        query=query,
        k=RETRIEVAL_TOP_K,
    )

    print()

    for result in results:
        preview = result["text"].replace(
            "\n",
            " ",
        )[:220]

        print(
            f"Rank {result['rank']} | "
            f"Score: {result['score']:.4f}"
        )
        print(
            f"Source: {result['source_file']}"
        )
        print(
            f"Page: {result['page_number']} | "
            f"Chunk: {result['chunk_id']}"
        )
        print(f"Text: {preview}...")
        print("-" * 80)

    wart_article_found = any(
        "wart" in result["source_file"].lower()
        for result in results
    )

    if wart_article_found:
        print(
            "TEST PASSED: A wart-treatment article "
            "was found among the top results."
        )
    else:
        print(
            "TEST WARNING: No wart-treatment article "
            "was found among the top results."
        )


if __name__ == "__main__":
    main()