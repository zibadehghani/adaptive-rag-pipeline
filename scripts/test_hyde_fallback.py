from __future__ import annotations

from typing import Any

from adaptive_rag.hyde_fallback import (
    run_retrieval_with_fallback,
)


class FakeRetriever:
    """Return deterministic results without loading Qwen or FAISS."""

    def __init__(
        self,
        scenario: str,
    ) -> None:
        self.scenario = scenario
        self.search_call_count = 0

    def search(
        self,
        query: str,
        k: int,
    ) -> list[dict[str, Any]]:
        self.search_call_count += 1

        is_hyde_query = (
            "A medical decision-support study"
            in query
        )

        if self.scenario == "initial_pass":
            scores = [
                0.76,
                0.73,
                0.70,
                0.68,
                0.66,
            ]

        elif (
            self.scenario == "hyde_pass"
            and is_hyde_query
        ):
            scores = [
                0.75,
                0.72,
                0.69,
                0.67,
                0.65,
            ]

        else:
            scores = [
                0.61,
                0.59,
                0.57,
                0.55,
                0.53,
            ]

        return [
            {
                "rank": rank,
                "score": score,
                "faiss_id": (
                    self.search_call_count * 100
                    + rank
                ),
                "chunk_id": (
                    f"synthetic_chunk_{rank}"
                ),
                "text": (
                    f"Synthetic medical result {rank}"
                ),
                "source_file": (
                    "synthetic_article.pdf"
                ),
                "document_id": "synthetic_article",
                "page_number": rank,
            }
            for rank, score in enumerate(
                scores[:k],
                start=1,
            )
        ]


TEST_CASES = [
    {
        "name": (
            "Strong initial retrieval skips HyDE"
        ),
        "scenario": "initial_pass",
        "expected_hyde_used": False,
        "expected_final_stage": "initial",
        "expected_search_calls": 1,
        "expected_final_status": "pass",
    },
    {
        "name": (
            "Weak initial retrieval uses HyDE once"
        ),
        "scenario": "hyde_pass",
        "expected_hyde_used": True,
        "expected_final_stage": "hyde",
        "expected_search_calls": 2,
        "expected_final_status": "pass",
    },
    {
        "name": (
            "Weak HyDE still stops after one attempt"
        ),
        "scenario": "all_fail",
        "expected_hyde_used": True,
        "expected_final_stage": "hyde",
        "expected_search_calls": 2,
        "expected_final_status": "fail",
    },
]


def main() -> None:
    print("=" * 80)
    print("Adaptive RAG — One-time HyDE Fallback Test")
    print("=" * 80)

    passed_tests = 0

    query = (
        "How can clinicians estimate medication-related "
        "cardiac electrical recovery risk?"
    )

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        retriever = FakeRetriever(
            scenario=test_case["scenario"]
        )

        output = run_retrieval_with_fallback(
            query=query,
            retriever=retriever,
            candidate_k=5,
            top_k=5,
        )

        final_evaluation = output[
            "final_output"
        ]["evaluation"]

        hyde_usage_is_correct = (
            output["hyde_used"]
            == test_case[
                "expected_hyde_used"
            ]
        )

        final_stage_is_correct = (
            output["final_stage"]
            == test_case[
                "expected_final_stage"
            ]
        )

        search_calls_are_correct = (
            retriever.search_call_count
            == test_case[
                "expected_search_calls"
            ]
        )

        final_status_is_correct = (
            final_evaluation[
                "quality_status"
            ]
            == test_case[
                "expected_final_status"
            ]
        )

        attempt_count_is_correct = (
            output["hyde_attempt_count"]
            in {0, 1}
        )

        if output["hyde_used"]:
            attempt_count_is_correct = (
                output["hyde_attempt_count"]
                == 1
            )

        test_passed = all(
            [
                hyde_usage_is_correct,
                final_stage_is_correct,
                search_calls_are_correct,
                final_status_is_correct,
                attempt_count_is_correct,
            ]
        )

        if test_passed:
            passed_tests += 1

        print()
        print("-" * 80)
        print(
            f"Test {test_number}: "
            f"{test_case['name']}"
        )
        print(
            f"Initial quality: "
            f"{output['initial_output']['evaluation']['quality_status']}"
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
            f"Search calls: "
            f"{retriever.search_call_count}"
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
            f"Trace: "
            f"{output['trace']}"
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
            "TEST PASSED: HyDE fallback was applied "
            "at most one time."
        )
    else:
        print(
            "TEST WARNING: One or more HyDE fallback "
            "decisions were incorrect."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()