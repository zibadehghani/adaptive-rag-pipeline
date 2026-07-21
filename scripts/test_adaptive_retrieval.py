from __future__ import annotations

from adaptive_rag.adaptive_retrieval import (
    adaptive_retrieve,
)
from adaptive_rag.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)
from adaptive_rag.embeddings import (
    Qwen3Embeddings,
)
from adaptive_rag.retrieval import (
    MedicalRetriever,
)


TEST_CASES = [
    {
        "query": "medical expert systems",
        "expected_route": "expansion",
        "expected_retrieval_query_count": 1,
        "expected_source_term": None,
    },
    {
        "query": (
            "How do medical expert systems diagnose diseases, "
            "recommend treatments, and classify patient risks?"
        ),
        "expected_route": "decomposition",
        "expected_retrieval_query_count": 3,
        "expected_source_term": None,
    },
    {
        "query": (
            "How does a fuzzy expert system select "
            "a wart treatment method?"
        ),
        "expected_route": "direct",
        "expected_retrieval_query_count": 1,
        "expected_source_term": "wart",
    },
]


def main() -> None:
    print("=" * 80)
    print("Adaptive RAG — Coverage-aware Retrieval Test")
    print("=" * 80)
    print("Loading Qwen3 model and FAISS index...")

    embedding_model = Qwen3Embeddings()

    retriever = MedicalRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH,
        embedding_model=embedding_model,
    )

    passed_tests = 0

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        output = adaptive_retrieve(
            query=test_case["query"],
            retriever=retriever,
            candidate_k=RETRIEVAL_CANDIDATE_K,
            top_k=RETRIEVAL_TOP_K,
        )

        result_ids = [
            result["faiss_id"]
            for result in output["results"]
        ]

        route_is_correct = (
            output["route"]
            == test_case["expected_route"]
        )

        query_count_is_correct = (
            len(output["retrieval_queries"])
            == test_case[
                "expected_retrieval_query_count"
            ]
        )

        results_exist = (
            0 < len(output["results"])
            <= RETRIEVAL_TOP_K
        )

        results_are_unique = (
            len(result_ids)
            == len(set(result_ids))
        )

        covered_queries = {
            query
            for result in output["results"]
            for query in result[
                "coverage_queries"
            ]
        }

        coverage_is_correct = (
            set(output["retrieval_queries"])
            <= covered_queries
        )

        source_term_is_correct = True

        expected_source_term = test_case[
            "expected_source_term"
        ]

        if expected_source_term is not None:
            source_term_is_correct = any(
                expected_source_term
                in result["source_file"].lower()
                for result in output["results"]
            )

        test_passed = all(
            [
                route_is_correct,
                query_count_is_correct,
                results_exist,
                results_are_unique,
                coverage_is_correct,
                source_term_is_correct,
            ]
        )

        if test_passed:
            passed_tests += 1

        print()
        print("-" * 80)
        print(f"Test {test_number}")
        print(f"Original query: {output['original_query']}")
        print(f"Route: {output['route']}")
        print(
            f"Retrieval queries: "
            f"{output['retrieval_queries']}"
        )
        print(
            f"Coverage: "
            f"{output['coverage_count']}/"
            f"{len(output['retrieval_queries'])}"
        )
        print(
            f"Retrieved candidates: "
            f"{output['retrieved_candidate_count']}"
        )
        print(
            f"Unique candidates: "
            f"{output['unique_candidate_count']}"
        )
        print(
            f"Final results: "
            f"{output['final_result_count']}"
        )

        for result in output["results"]:
            print(
                f"Rank {result['rank']} | "
                f"Score: {result['score']:.4f} | "
                f"Matches: {result['match_count']}"
            )
            print(
                f"Source: {result['source_file']}"
            )
            print(
                f"Page: {result['page_number']}"
            )
            print(
                f"Coverage queries: "
                f"{result['coverage_queries']}"
            )
            print(
                f"Best retrieval query: "
                f"{result['best_retrieval_query']}"
            )

        print(
            "Status: "
            + (
                "PASSED"
                if test_passed
                else "FAILED"
            )
        )

    print()
    print("=" * 80)
    print(
        f"Passed tests: "
        f"{passed_tests}/{len(TEST_CASES)}"
    )

    if passed_tests == len(TEST_CASES):
        print(
            "TEST PASSED: All retrieval paths preserved "
            "query coverage."
        )
    else:
        print(
            "TEST WARNING: One or more retrieval paths "
            "lost query coverage."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()