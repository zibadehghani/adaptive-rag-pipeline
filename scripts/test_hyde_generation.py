from __future__ import annotations

from adaptive_rag.hyde import (
    generate_hyde_document,
)


TEST_CASES = [
    {
        "query": (
            "How can clinicians estimate the danger caused by "
            "medications that alter the heart's electrical "
            "recovery process?"
        ),
        "expected_route": "direct",
        "minimum_focus_items": 1,
    },
    {
        "query": (
            "How do medical expert systems diagnose diseases, "
            "recommend treatments, and classify patient risks?"
        ),
        "expected_route": "decomposition",
        "minimum_focus_items": 3,
    },
]


def contains_duplicate_punctuation(
    text: str,
) -> bool:
    """Detect punctuation combinations produced by formatting errors."""

    invalid_patterns = [
        "?.",
        "!.",
        ",.",
        ";.",
        ":.",
    ]

    return any(
        pattern in text
        for pattern in invalid_patterns
    )


def main() -> None:
    print("=" * 80)
    print("Adaptive RAG — Template-based HyDE Test")
    print("=" * 80)

    passed_tests = 0

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        output = generate_hyde_document(
            test_case["query"]
        )

        document = output[
            "hypothetical_document"
        ]

        route_is_correct = (
            output["route"]
            == test_case["expected_route"]
        )

        method_is_correct = (
            output["hyde_method"]
            == "template_based_hyde"
            and output["generation_source"]
            == "deterministic_template"
        )

        document_is_longer = (
            len(document.split())
            > len(
                output[
                    "normalized_query"
                ].split()
            )
        )

        focus_is_sufficient = (
            len(output["focus_items"])
            >= test_case[
                "minimum_focus_items"
            ]
        )

        query_is_represented = (
            output["normalized_query"].lower()
            in document.lower()
        )

        punctuation_is_valid = not (
            contains_duplicate_punctuation(
                document
            )
        )

        focus_items_are_clean = all(
            not item.endswith(
                (
                    "?",
                    ".",
                    "!",
                    ",",
                    ";",
                    ":",
                )
            )
            for item in output["focus_items"]
        )

        test_passed = all(
            [
                route_is_correct,
                method_is_correct,
                document_is_longer,
                focus_is_sufficient,
                query_is_represented,
                punctuation_is_valid,
                focus_items_are_clean,
            ]
        )

        if test_passed:
            passed_tests += 1

        print()
        print("-" * 80)
        print(f"Test {test_number}")
        print(
            f"Original query: "
            f"{output['original_query']}"
        )
        print(
            f"Detected route: "
            f"{output['route']}"
        )
        print(
            f"HyDE method: "
            f"{output['hyde_method']}"
        )
        print(
            f"Focus items: "
            f"{output['focus_items']}"
        )
        print(
            f"Document word count: "
            f"{output['hyde_document_word_count']}"
        )
        print(
            f"Valid punctuation: "
            f"{punctuation_is_valid}"
        )
        print(
            f"Hypothetical document: "
            f"{document}"
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
            "TEST PASSED: Template-based HyDE documents "
            "were generated correctly."
        )
    else:
        print(
            "TEST WARNING: One or more HyDE documents "
            "were generated incorrectly."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()