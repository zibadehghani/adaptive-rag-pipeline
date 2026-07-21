from __future__ import annotations

from adaptive_rag.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)
from adaptive_rag.embeddings import (
    Qwen3Embeddings,
)
from adaptive_rag.graph_workflow import (
    create_adaptive_rag_graph,
)
from adaptive_rag.retrieval import (
    MedicalRetriever,
)


def main() -> None:
    query = (
        "How does a fuzzy expert system select "
        "a wart treatment method?"
    )

    print("=" * 80)
    print("Adaptive RAG — Real LangGraph Integration Test")
    print("=" * 80)
    print(f"Query: {query}")
    print("Loading Qwen3 model and FAISS index...")

    embedding_model = Qwen3Embeddings()

    retriever = MedicalRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH,
        embedding_model=embedding_model,
    )

    graph = create_adaptive_rag_graph(
        retriever=retriever,
        candidate_k=RETRIEVAL_CANDIDATE_K,
        top_k=RETRIEVAL_TOP_K,
    )

    output = graph.invoke(
        {
            "original_query": query,
            "trace": [],
            "hyde_attempt_count": 0,
        }
    )

    initial_output = output[
        "initial_output"
    ]

    initial_evaluation = initial_output[
        "evaluation"
    ]

    final_output = output[
        "final_output"
    ]

    final_evaluation = final_output[
        "evaluation"
    ]

    print()
    print("=" * 80)
    print("QUERY DECISION")
    print("=" * 80)
    print(f"Route: {output['route']}")
    print(f"Reason: {output['reason']}")
    print(
        f"Strategy: "
        f"{final_output['strategy']}"
    )
    print(
        f"Transformed query: "
        f"{final_output['transformed_query']}"
    )
    print(
        f"Subqueries: "
        f"{final_output['subqueries']}"
    )

    print()
    print("=" * 80)
    print("RETRIEVAL DECISION")
    print("=" * 80)
    print(
        f"Initial quality: "
        f"{initial_evaluation['quality_status']}"
    )
    print(
        f"Initial average score: "
        f"{initial_evaluation['average_score']:.4f}"
    )
    print(
        f"Initial accepted results: "
        f"{initial_evaluation['accepted_count']}/"
        f"{initial_evaluation['result_count']}"
    )
    print(
        f"HyDE used: "
        f"{output['hyde_used']}"
    )
    print(
        f"HyDE attempts: "
        f"{output['hyde_attempt_count']}"
    )
    print(
        f"Final stage: "
        f"{output['final_stage']}"
    )
    print(
        f"Final quality: "
        f"{final_evaluation['quality_status']}"
    )
    print(
        f"Final average score: "
        f"{final_evaluation['average_score']:.4f}"
    )
    print(
        f"Graph trace: "
        f"{output['trace']}"
    )

    if output["hyde_used"]:
        hypothetical_document = output[
            "hyde_generation"
        ]["hypothetical_document"]

        print()
        print("=" * 80)
        print("TEMPLATE-BASED HYDE DOCUMENT")
        print("=" * 80)
        print(hypothetical_document)

    print()
    print("=" * 80)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 80)

    for result in final_output["results"]:
        preview = result["text"].replace(
            "\n",
            " ",
        )[:250]

        print(
            f"Rank {result['rank']} | "
            f"Score: {result['score']:.4f} | "
            f"Accepted: {result['accepted']}"
        )
        print(
            f"Source: {result['source_file']}"
        )
        print(
            f"Page: {result['page_number']} | "
            f"Chunk: {result['chunk_id']}"
        )
        print(
            f"Best retrieval query: "
            f"{result['best_retrieval_query']}"
        )
        print(f"Text: {preview}...")
        print("-" * 80)

    wart_source_found = any(
        "wart" in result[
            "source_file"
        ].lower()
        for result in final_output[
            "results"
        ]
    )

    graph_finished = (
        output["trace"][-1]
        == "finalize_results"
    )

    results_exist = (
        len(final_output["results"]) > 0
    )

    test_passed = all(
        [
            output["route"] == "direct",
            wart_source_found,
            graph_finished,
            results_exist,
        ]
    )

    print()
    print("=" * 80)

    if test_passed:
        print(
            "TEST PASSED: The real LangGraph workflow "
            "retrieved the expected wart-treatment article."
        )
    else:
        print(
            "TEST WARNING: The real workflow did not "
            "produce the expected result."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()