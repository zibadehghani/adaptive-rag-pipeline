from __future__ import annotations

from typing import Any

from adaptive_rag.adaptive_retrieval import (
    adaptive_retrieve,
    merge_retrieval_results,
)
from adaptive_rag.config import (
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)
from adaptive_rag.hyde import (
    generate_hyde_document,
)
from adaptive_rag.retrieval import (
    MedicalRetriever,
)
from adaptive_rag.retrieval_evaluator import (
    evaluate_retrieval,
)


def retrieve_with_hyde(
    query: str,
    retriever: MedicalRetriever,
    candidate_k: int = RETRIEVAL_CANDIDATE_K,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    """Generate one HyDE document and retrieve its closest chunks."""

    hyde_output = generate_hyde_document(
        query
    )

    hypothetical_document = hyde_output[
        "hypothetical_document"
    ]

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

    final_results = merge_retrieval_results(
        grouped_results=grouped_results,
        top_k=top_k,
    )

    covered_queries = sorted(
        {
            coverage_query
            for result in final_results
            for coverage_query in result[
                "coverage_queries"
            ]
        }
    )

    return {
        **hyde_output,
        "retrieval_mode": "hyde",
        "retrieval_queries": [
            hypothetical_document
        ],
        "candidate_k": candidate_k,
        "top_k": top_k,
        "retrieved_candidate_count": len(
            raw_results
        ),
        "unique_candidate_count": len(
            {
                result["faiss_id"]
                for result in raw_results
            }
        ),
        "covered_retrieval_queries": (
            covered_queries
        ),
        "coverage_count": len(
            covered_queries
        ),
        "final_result_count": len(
            final_results
        ),
        "results": final_results,
    }


def run_retrieval_with_fallback(
    query: str,
    retriever: MedicalRetriever,
    candidate_k: int = RETRIEVAL_CANDIDATE_K,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    """Run initial retrieval and use HyDE at most one time."""

    initial_retrieval = adaptive_retrieve(
        query=query,
        retriever=retriever,
        candidate_k=candidate_k,
        top_k=top_k,
    )

    evaluated_initial = evaluate_retrieval(
        initial_retrieval
    )

    trace = [
        "analyze_and_transform_query",
        "initial_retrieval",
        "evaluate_initial_retrieval",
    ]

    initial_quality_passed = evaluated_initial[
        "evaluation"
    ]["quality_passed"]

    if initial_quality_passed:
        trace.append(
            "finalize_initial_results"
        )

        return {
            "original_query": query,
            "initial_route": evaluated_initial[
                "route"
            ],
            "fallback_triggered": False,
            "hyde_used": False,
            "hyde_attempt_count": 0,
            "final_stage": "initial",
            "initial_output": evaluated_initial,
            "hyde_output": None,
            "final_output": evaluated_initial,
            "trace": trace,
        }

    trace.extend(
        [
            "generate_template_based_hyde",
            "hyde_retrieval",
            "evaluate_hyde_retrieval",
            "finalize_hyde_results",
        ]
    )

    hyde_retrieval = retrieve_with_hyde(
        query=query,
        retriever=retriever,
        candidate_k=candidate_k,
        top_k=top_k,
    )

    evaluated_hyde = evaluate_retrieval(
        hyde_retrieval
    )

    return {
        "original_query": query,
        "initial_route": evaluated_initial[
            "route"
        ],
        "fallback_triggered": True,
        "hyde_used": True,
        "hyde_attempt_count": 1,
        "final_stage": "hyde",
        "initial_output": evaluated_initial,
        "hyde_output": evaluated_hyde,
        "final_output": evaluated_hyde,
        "trace": trace,
    }