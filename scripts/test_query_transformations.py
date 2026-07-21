from __future__ import annotations

import re

from adaptive_rag.query_transformer import (
    transform_query,
)


TEST_CASES = [
    {
        "query": "medical expert systems",
        "expected_route": "expansion",
        "expected_subquery_count": 0,
    },
    {
        "query": (
            "How do medical expert systems diagnose diseases, "
            "recommend treatments, and classify patient risks?"
        ),
        "expected_route": "decomposition",
        "expected_subquery_count": 3,
    },
    {
        "query": (
            "How does a fuzzy expert system select "
            "a wart treatment method?"
        ),
        "expected_route": "direct",
        "expected_subquery_count": 0,
    },
]


def has_invalid_connector(
    subquery: str,
) -> bool:
    """Detect a connector incorrectly placed before an action verb."""

    return bool(
        re.search(
            r"\b(?:and|or|also)\s+"
            r"(?:diagnose|recommend|classify|select|assess|predict)\b",
            subquery,
            flags=re.IGNORECASE,
        )
    )


def main() -> None:
    print("=" * 80)
    print("Adaptive RAG — Query Transformation Test")
    print("=" * 80)

    passed_tests = 0

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        result = transform_query(
            test_case["query"]
        )

        route_is_correct = (
            result["route"]
            == test_case["expected_route"]
        )

        subquery_count_is_correct = (
            len(result["subqueries"])
            == test_case[
                "expected_subquery_count"
            ]
        )

        subqueries_are_valid = not any(
            has_invalid_connector(subquery)
            for subquery in result["subqueries"]
        )

        transformation_is_valid = True

        if result["route"] == "expansion":
            transformation_is_valid = (
                len(result["added_terms"]) > 0
                and result["transformed_query"]
                != result["normalized_query"]
            )

        elif result["route"] == "direct":
            transformation_is_valid = (
                result["transformed_query"]
                == result["normalized_query"]
            )

        test_passed = (
            route_is_correct
            and subquery_count_is_correct
            and subqueries_are_valid
            and transformation_is_valid
        )

        if test_passed:
            passed_tests += 1

        print(f"Test {test_number}")
        print(
            f"Original query: "
            f"{result['original_query']}"
        )
        print(
            f"Route: {result['route']}"
        )
        print(
            f"Strategy: {result['strategy']}"
        )
        print(
            f"Transformed query: "
            f"{result['transformed_query']}"
        )
        print(
            f"Added terms: "
            f"{result['added_terms']}"
        )
        print(
            f"Subqueries: "
            f"{result['subqueries']}"
        )
        print(
            f"Valid subqueries: "
            f"{subqueries_are_valid}"
        )
        print(
            "Status: "
            + (
                "PASSED"
                if test_passed
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
            "TEST PASSED: All query transformations "
            "were created correctly."
        )
    else:
        print(
            "TEST WARNING: One or more query "
            "transformations were incorrect."
        )


if __name__ == "__main__":
    main()