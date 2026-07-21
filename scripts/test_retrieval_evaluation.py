from __future__ import annotations

from typing import Any

from adaptive_rag.retrieval_evaluator import (
    evaluate_retrieval,
)


TEST_CASES = [
    {
        "name": "Strong retrieval with complete coverage",
        "scores": [
            0.74,
            0.70,
            0.68,
            0.66,
            0.64,
        ],
        "retrieval_queries": [
            "wart treatment selection",
        ],
        "covered_queries": [
            "wart treatment selection",
        ],
        "expected_status": "pass",
    },
    {
        "name": "Weak similarity scores",
        "scores": [
            0.61,
            0.59,
            0.57,
            0.55,
            0.53,
        ],
        "retrieval_queries": [
            "unfamiliar medical query",
        ],
        "covered_queries": [
            "unfamiliar medical query",
        ],
        "expected_status": "fail",
    },
    {
        "name": "Incomplete subquery coverage",
        "scores": [
            0.75,
            0.72,
            0.70,
            0.68,
            0.66,
        ],
        "retrieval_queries": [
            "diagnose diseases",
            "recommend treatments",
        ],
        "covered_queries": [
            "diagnose diseases",
        ],
        "expected_status": "fail",
    },
    {
        "name": "Insufficient accepted results",
        "scores": [
            0.70,
            0.64,
            0.63,
            0.62,
            0.61,
        ],
        "retrieval_queries": [
            "clinical risk assessment",
        ],
        "covered_queries": [
            "clinical risk assessment",
        ],
        "expected_status": "fail",
    },
]


def build_test_output(
    scores: list[float],
    retrieval_queries: list[str],
    covered_queries: list[str],
) -> dict[str, Any]:
    """Create a minimal adaptive-retrieval output for testing."""

    results = [
        {
            "rank": rank,
            "faiss_id": rank - 1,
            "score": score,
            "text": f"Synthetic result {rank}",
        }
        for rank, score in enumerate(
            scores,
            start=1,
        )
    ]

    return {
        "route": "direct",
        "retrieval_queries": retrieval_queries,
        "covered_retrieval_queries": covered_queries,
        "results": results,
    }


def main() -> None:
    print("=" * 80)
    print("Adaptive RAG — Retrieval Quality Evaluation Test")
    print("=" * 80)

    passed_tests = 0

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        retrieval_output = build_test_output(
            scores=test_case["scores"],
            retrieval_queries=test_case[
                "retrieval_queries"
            ],
            covered_queries=test_case[
                "covered_queries"
            ],
        )

        evaluated_output = evaluate_retrieval(
            retrieval_output
        )

        evaluation = evaluated_output[
            "evaluation"
        ]

        status_is_correct = (
            evaluation["quality_status"]
            == test_case["expected_status"]
        )

        accepted_flags_are_valid = all(
            result["accepted"]
            == (
                result["score"]
                >= evaluation["score_threshold"]
            )
            for result in evaluated_output[
                "results"
            ]
        )

        test_passed = (
            status_is_correct
            and accepted_flags_are_valid
        )

        if test_passed:
            passed_tests += 1

        print()
        print("-" * 80)
        print(f"Test {test_number}: {test_case['name']}")
        print(
            f"Expected status: "
            f"{test_case['expected_status']}"
        )
        print(
            f"Detected status: "
            f"{evaluation['quality_status']}"
        )
        print(
            f"Top score: "
            f"{evaluation['top_score']:.4f}"
        )
        print(
            f"Average score: "
            f"{evaluation['average_score']:.4f}"
        )
        print(
            f"Accepted results: "
            f"{evaluation['accepted_count']}/"
            f"{evaluation['result_count']}"
        )
        print(
            f"Coverage: "
            f"{evaluation['coverage_count']}/"
            f"{evaluation['coverage_total']}"
        )
        print(
            f"Checks: {evaluation['checks']}"
        )
        print(
            f"Reason: "
            f"{evaluation['quality_reason']}"
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
            "TEST PASSED: Retrieval quality decisions "
            "were calculated correctly."
        )
    else:
        print(
            "TEST WARNING: One or more quality "
            "decisions were incorrect."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()