from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from adaptive_rag.adaptive_retrieval import (
    merge_retrieval_results,
    select_retrieval_queries,
)
from adaptive_rag.config import (
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)
from adaptive_rag.hyde import (
    generate_hyde_document,
)
from adaptive_rag.query_analyzer import (
    analyze_query,
)
from adaptive_rag.query_transformer import (
    decompose_query,
    expand_query,
)
from adaptive_rag.retrieval import (
    MedicalRetriever,
)
from adaptive_rag.retrieval_evaluator import (
    evaluate_retrieval,
)


class AdaptiveRAGState(TypedDict, total=False):
    """Shared state passed between Adaptive RAG graph nodes."""

    original_query: str
    normalized_query: str
    route: str
    reason: str
    metrics: dict[str, Any]

    strategy: str
    transformed_query: str
    added_terms: list[str]
    subqueries: list[str]

    initial_output: dict[str, Any]
    hyde_generation: dict[str, Any]
    hyde_output: dict[str, Any]
    final_output: dict[str, Any]

    fallback_triggered: bool
    hyde_used: bool
    hyde_attempt_count: int
    final_stage: str
    trace: list[str]


def _append_trace(
    state: AdaptiveRAGState,
    node_name: str,
) -> list[str]:
    """Append one node name to the execution trace."""

    return [
        *state.get("trace", []),
        node_name,
    ]


def _build_retrieval_output(
    state: AdaptiveRAGState,
    retrieval_queries: list[str],
    grouped_results: list[
        tuple[str, list[dict[str, Any]]]
    ],
    candidate_k: int,
    top_k: int,
    retrieval_mode: str,
) -> dict[str, Any]:
    """Create a standard output from one or more FAISS searches."""

    final_results = merge_retrieval_results(
        grouped_results=grouped_results,
        top_k=top_k,
    )

    covered_queries = list(
        dict.fromkeys(
            coverage_query
            for result in final_results
            for coverage_query in result.get(
                "coverage_queries",
                [],
            )
        )
    )

    unique_candidate_ids = {
        int(result["faiss_id"])
        for _, results in grouped_results
        for result in results
    }

    return {
        "original_query": state["original_query"],
        "normalized_query": state["normalized_query"],
        "route": state["route"],
        "reason": state["reason"],
        "metrics": state["metrics"],
        "strategy": state["strategy"],
        "transformed_query": state[
            "transformed_query"
        ],
        "added_terms": state.get(
            "added_terms",
            [],
        ),
        "subqueries": state.get(
            "subqueries",
            [],
        ),
        "retrieval_mode": retrieval_mode,
        "retrieval_queries": retrieval_queries,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "retrieved_candidate_count": sum(
            len(results)
            for _, results in grouped_results
        ),
        "unique_candidate_count": len(
            unique_candidate_ids
        ),
        "covered_retrieval_queries": covered_queries,
        "coverage_count": len(
            covered_queries
        ),
        "final_result_count": len(
            final_results
        ),
        "results": final_results,
    }


def create_adaptive_rag_graph(
    retriever: MedicalRetriever,
    candidate_k: int = RETRIEVAL_CANDIDATE_K,
    top_k: int = RETRIEVAL_TOP_K,
) -> Any:
    """Create and compile the Adaptive RAG LangGraph workflow."""

    if candidate_k < 1:
        raise ValueError(
            "candidate_k must be at least 1."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    def analyze_query_node(
        state: AdaptiveRAGState,
    ) -> dict[str, Any]:
        analysis = analyze_query(
            state["original_query"]
        )

        return {
            **analysis,
            "fallback_triggered": False,
            "hyde_used": False,
            "hyde_attempt_count": 0,
            "trace": _append_trace(
                state,
                "analyze_query",
            ),
        }

    def route_after_analysis(
        state: AdaptiveRAGState,
    ) -> Literal[
        "expansion",
        "decomposition",
        "direct",
    ]:
        return state["route"]

    def expand_query_node(
        state: AdaptiveRAGState,
    ) -> dict[str, Any]:
        transformation = expand_query(
            state["normalized_query"]
        )

        return {
            **transformation,
            "trace": _append_trace(
                state,
                "expand_query",
            ),
        }

    def decompose_query_node(
        state: AdaptiveRAGState,
    ) -> dict[str, Any]:
        transformation = decompose_query(
            state["normalized_query"]
        )

        return {
            **transformation,
            "trace": _append_trace(
                state,
                "decompose_query",
            ),
        }

    def direct_query_node(
        state: AdaptiveRAGState,
    ) -> dict[str, Any]:
        return {
            "strategy": "direct",
            "transformed_query": state[
                "normalized_query"
            ],
            "added_terms": [],
            "subqueries": [],
            "trace": _append_trace(
                state,
                "direct_query",
            ),
        }

    def initial_retrieval_node(
        state: AdaptiveRAGState,
    ) -> dict[str, Any]:
        transformation = {
            "route": state["route"],
            "transformed_query": state[
                "transformed_query"
            ],
            "subqueries": state.get(
                "subqueries",
                [],
            ),
        }

        retrieval_queries = select_retrieval_queries(
            transformation
        )

        grouped_results: list[
            tuple[str, list[dict[str, Any]]]
        ] = []

        for retrieval_query in retrieval_queries:
            results = retriever.search(
                query=retrieval_query,
                k=candidate_k,
            )

            grouped_results.append(
                (
                    retrieval_query,
                    results,
                )
            )

        initial_output = _build_retrieval_output(
            state=state,
            retrieval_queries=retrieval_queries,
            grouped_results=grouped_results,
            candidate_k=candidate_k,
            top_k=top_k,
            retrieval_mode="initial",
        )

        return {
            "initial_output": initial_output,
            "trace": _append_trace(
                state,
                "initial_retrieval",
            ),
        }

    def evaluate_initial_node(
        state: AdaptiveRAGState,
    ) -> dict[str, Any]:
        evaluated_output = evaluate_retrieval(
            state["initial_output"]
        )

        return {
            "initial_output": evaluated_output,
            "trace": _append_trace(
                state,
                "evaluate_initial_retrieval",
            ),
        }

    def route_after_initial_evaluation(
        state: AdaptiveRAGState,
    ) -> Literal["pass", "fail"]:
        quality_passed = state[
            "initial_output"
        ]["evaluation"]["quality_passed"]

        return (
            "pass"
            if quality_passed
            else "fail"
        )

    def generate_hyde_node(
        state: AdaptiveRAGState,
    ) -> dict[str, Any]:
        if state.get(
            "hyde_attempt_count",
            0,
        ) >= 1:
            raise RuntimeError(
                "HyDE fallback may only run once."
            )

        hyde_generation = generate_hyde_document(
            state["original_query"]
        )

        return {
            "hyde_generation": hyde_generation,
            "fallback_triggered": True,
            "hyde_used": True,
            "hyde_attempt_count": 1,
            "trace": _append_trace(
                state,
                "generate_template_based_hyde",
            ),
        }

    def hyde_retrieval_node(
        state: AdaptiveRAGState,
    ) -> dict[str, Any]:
        hypothetical_document = state[
            "hyde_generation"
        ]["hypothetical_document"]

        raw_results = retriever.search(
            query=hypothetical_document,
            k=candidate_k,
        )

        grouped_results = [
            (
                hypothetical_document,
                raw_results,
            )
        ]

        hyde_output = _build_retrieval_output(
            state=state,
            retrieval_queries=[
                hypothetical_document
            ],
            grouped_results=grouped_results,
            candidate_k=candidate_k,
            top_k=top_k,
            retrieval_mode="hyde",
        )

        hyde_output.update(
            {
                "hyde_method": state[
                    "hyde_generation"
                ]["hyde_method"],
                "generation_source": state[
                    "hyde_generation"
                ]["generation_source"],
                "focus_items": state[
                    "hyde_generation"
                ]["focus_items"],
                "hypothetical_document": (
                    hypothetical_document
                ),
                "hyde_document_word_count": state[
                    "hyde_generation"
                ]["hyde_document_word_count"],
            }
        )

        return {
            "hyde_output": hyde_output,
            "trace": _append_trace(
                state,
                "hyde_retrieval",
            ),
        }

    def evaluate_hyde_node(
        state: AdaptiveRAGState,
    ) -> dict[str, Any]:
        evaluated_output = evaluate_retrieval(
            state["hyde_output"]
        )

        return {
            "hyde_output": evaluated_output,
            "trace": _append_trace(
                state,
                "evaluate_hyde_retrieval",
            ),
        }

    def finalize_node(
        state: AdaptiveRAGState,
    ) -> dict[str, Any]:
        hyde_output = state.get(
            "hyde_output"
        )

        if hyde_output is not None:
            final_output = hyde_output
            final_stage = "hyde"
        else:
            final_output = state[
                "initial_output"
            ]
            final_stage = "initial"

        return {
            "final_output": final_output,
            "final_stage": final_stage,
            "trace": _append_trace(
                state,
                "finalize_results",
            ),
        }

    builder = StateGraph(
        AdaptiveRAGState
    )

    builder.add_node(
        "analyze_query",
        analyze_query_node,
    )

    builder.add_node(
        "expand_query",
        expand_query_node,
    )

    builder.add_node(
        "decompose_query",
        decompose_query_node,
    )

    builder.add_node(
        "direct_query",
        direct_query_node,
    )

    builder.add_node(
        "initial_retrieval",
        initial_retrieval_node,
    )

    builder.add_node(
        "evaluate_initial_retrieval",
        evaluate_initial_node,
    )

    builder.add_node(
        "generate_hyde",
        generate_hyde_node,
    )

    builder.add_node(
        "hyde_retrieval",
        hyde_retrieval_node,
    )

    builder.add_node(
        "evaluate_hyde_retrieval",
        evaluate_hyde_node,
    )

    builder.add_node(
        "finalize_results",
        finalize_node,
    )

    builder.add_edge(
        START,
        "analyze_query",
    )

    builder.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "expansion": "expand_query",
            "decomposition": "decompose_query",
            "direct": "direct_query",
        },
    )

    builder.add_edge(
        "expand_query",
        "initial_retrieval",
    )

    builder.add_edge(
        "decompose_query",
        "initial_retrieval",
    )

    builder.add_edge(
        "direct_query",
        "initial_retrieval",
    )

    builder.add_edge(
        "initial_retrieval",
        "evaluate_initial_retrieval",
    )

    builder.add_conditional_edges(
        "evaluate_initial_retrieval",
        route_after_initial_evaluation,
        {
            "pass": "finalize_results",
            "fail": "generate_hyde",
        },
    )

    builder.add_edge(
        "generate_hyde",
        "hyde_retrieval",
    )

    builder.add_edge(
        "hyde_retrieval",
        "evaluate_hyde_retrieval",
    )

    builder.add_edge(
        "evaluate_hyde_retrieval",
        "finalize_results",
    )

    builder.add_edge(
        "finalize_results",
        END,
    )

    return builder.compile()