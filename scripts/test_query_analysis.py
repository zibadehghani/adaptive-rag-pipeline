from __future__ import annotations

from adaptive_rag.query_analyzer import analyze_query


TEST_CASES = [
    {
        "query": "medical expert systems",
        "expected_route": "expansion",
    },
    {
        "query": (
            "How do medical expert systems diagnose diseases, "
            "recommend treatments, and classify patient risks?"
        ),
        "expected_route": "decomposition",
    },
    {
        "query": (
            "How does a fuzzy expert system select "
            "a wart treatment method?"
        ),
        "expected_route": "direct",
    },
]


def main() -> None:
    print("=" * 80)
    print("Adaptive RAG — Query Analysis Test")
    print("=" * 80)

    passed_tests = 0

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        result = analyze_query(
            test_case["query"]
        )

        route_is_correct = (
            result["route"]
            == test_case["expected_route"]
        )

        if route_is_correct:
            passed_tests += 1

        print(f"Test {test_number}")
        print(
            f"Query: {result['normalized_query']}"
        )
        print(
            f"Expected route: "
            f"{test_case['expected_route']}"
        )
        print(
            f"Detected route: {result['route']}"
        )
        print(f"Reason: {result['reason']}")
        print(
            f"Metrics: {result['metrics']}"
        )
        print(
            "Status: "
            + (
                "PASSED"
                if route_is_correct
                else "FAILED"
            )
        )
        print("-" * 80)

    print(
        f"Passed tests: "
        f"{passed_tests}/{len(TEST_CASES)}"
    )

    if passed_tests == len(TEST_CASES):
        print(
            "TEST PASSED: All queries were routed correctly."
        )
    else:
        print(
            "TEST WARNING: One or more routes were incorrect."
        )


if __name__ == "__main__":
    main()