from __future__ import annotations

from typing import Any

from adaptive_rag.graph_workflow import (
    create_adaptive_rag_graph,
)


class FakeRetriever:
    """Provide deterministic results for LangGraph testing."""

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
                    f"synthetic_chunk_"
                    f"{self.search_call_count}_{rank}"
                ),
                "text": (
                    f"Synthetic medical result {rank}"
                ),
                "source_file": (
                    "synthetic_medical_article.pdf"
                ),
                "document_id": (
                    "synthetic_medical_article"
                ),
                "page_number": rank,
            }
            for rank, score in enumerate(
                scores[:k],
                start=1,
            )
        ]


TEST_CASES = [
    {
        "name": "Expansion route passes initially",
        "query": "medical expert systems",
        "scenario": "initial_pass",
        "expected_route": "expansion",
        "expected_route_node": "expand_query",
        "expected_hyde_used": False,
        "expected_final_stage": "initial",
        "expected_final_status": "pass",
        "expected_search_calls": 1,
    },
    {
        "name": "Decomposition route preserves coverage",
        "query": (
            "How do medical expert systems diagnose diseases, "
            "recommend treatments, and classify patient risks?"
        ),
        "scenario": "initial_pass",
        "expected_route": "decomposition",
        "expected_route_node": "decompose_query",
        "expected_hyde_used": False,
        "expected_final_stage": "initial",
        "expected_final_status": "pass",
        "expected_search_calls": 3,
    },
    {
        "name": "Direct route uses successful HyDE fallback",
        "query": (
            "How can clinicians estimate medication-related "
            "cardiac electrical recovery risk?"
        ),
        "scenario": "hyde_pass",
        "expected_route": "direct",
        "expected_route_node": "direct_query",
        "expected_hyde_used": True,
        "expected_final_stage": "hyde",
        "expected_final_status": "pass",
        "expected_search_calls": 2,
    },
    {
        "name": "Graph stops after failed HyDE fallback",
        "query": (
            "How can clinicians estimate medication-related "
            "cardiac electrical recovery risk?"
        ),
        "scenario": "all_fail",
        "expected_route": "direct",
        "expected_route_node": "direct_query",
        "expected_hyde_used": True,
        "expected_final_stage": "hyde",
        "expected_final_status": "fail",
        "expected_search_calls": 2,
    },
]


def main() -> None:
    print("=" * 80)
    print("Adaptive RAG — LangGraph Workflow Test")
    print("=" * 80)

    passed_tests = 0

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        retriever = FakeRetriever(
            scenario=test_case["scenario"]
        )

        graph = create_adaptive_rag_graph(
            retriever=retriever,
            candidate_k=5,
            top_k=5,
        )

        output = graph.invoke(
            {
                "original_query": test_case[
                    "query"
                ],
                "trace": [],
                "hyde_attempt_count": 0,
            }
        )

        final_evaluation = output[
            "final_output"
        ]["evaluation"]

        route_is_correct = (
            output["route"]
            == test_case["expected_route"]
        )

        route_node_is_present = (
            test_case["expected_route_node"]
            in output["trace"]
        )

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

        final_status_is_correct = (
            final_evaluation[
                "quality_status"
            ]
            == test_case[
                "expected_final_status"
            ]
        )

        search_calls_are_correct = (
            retriever.search_call_count
            == test_case[
                "expected_search_calls"
            ]
        )

        graph_finished_correctly = (
            output["trace"][-1]
            == "finalize_results"
        )

        hyde_attempt_is_valid = (
            output["hyde_attempt_count"]
            == (
                1
                if output["hyde_used"]
                else 0
            )
        )

        test_passed = all(
            [
                route_is_correct,
                route_node_is_present,
                hyde_usage_is_correct,
                final_stage_is_correct,
                final_status_is_correct,
                search_calls_are_correct,
                graph_finished_correctly,
                hyde_attempt_is_valid,
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
            f"Route: {output['route']}"
        )
        print(
            f"HyDE used: {output['hyde_used']}"
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
            f"Graph trace: "
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
            "TEST PASSED: The LangGraph workflow "
            "followed all Adaptive RAG routes correctly."
        )
    else:
        print(
            "TEST WARNING: One or more LangGraph "
            "routes were incorrect."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()