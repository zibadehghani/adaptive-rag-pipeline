from __future__ import annotations

from typing import Any

from adaptive_rag.config import (
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)
from adaptive_rag.query_transformer import (
    transform_query,
)
from adaptive_rag.retrieval import MedicalRetriever


def select_retrieval_queries(
    transformation: dict[str, Any],
) -> list[str]:
    """Select the queries that must be sent to FAISS."""

    route = transformation["route"]

    if (
        route == "decomposition"
        and transformation["subqueries"]
    ):
        return list(
            transformation["subqueries"]
        )

    return [
        transformation["transformed_query"]
    ]


def merge_retrieval_results(
    grouped_results: list[
        tuple[str, list[dict[str, Any]]]
    ],
    top_k: int,
) -> list[dict[str, Any]]:
    """Merge duplicates while preserving subquery coverage."""

    merged_results: dict[
        int,
        dict[str, Any],
    ] = {}

    for retrieval_query, results in grouped_results:
        for result in results:
            faiss_id = int(
                result["faiss_id"]
            )

            existing_result = merged_results.get(
                faiss_id
            )

            if existing_result is None:
                new_result = dict(result)

                new_result[
                    "best_retrieval_query"
                ] = retrieval_query

                new_result[
                    "matched_queries"
                ] = [retrieval_query]

                new_result["match_count"] = 1
                new_result["coverage_queries"] = []

                merged_results[
                    faiss_id
                ] = new_result

                continue

            matched_queries = list(
                existing_result[
                    "matched_queries"
                ]
            )

            if retrieval_query not in matched_queries:
                matched_queries.append(
                    retrieval_query
                )

            if result["score"] > existing_result["score"]:
                existing_result.update(
                    result
                )

                existing_result[
                    "best_retrieval_query"
                ] = retrieval_query

            existing_result[
                "matched_queries"
            ] = matched_queries

            existing_result[
                "match_count"
            ] = len(matched_queries)

    globally_sorted_results = sorted(
        merged_results.values(),
        key=lambda item: (
            item["score"],
            item["match_count"],
        ),
        reverse=True,
    )

    selected_results: list[
        dict[str, Any]
    ] = []

    selected_ids: set[int] = set()

    for retrieval_query, query_results in grouped_results:
        if len(selected_results) >= top_k:
            break

        query_results_sorted = sorted(
            query_results,
            key=lambda item: item["score"],
            reverse=True,
        )

        for query_result in query_results_sorted:
            faiss_id = int(
                query_result["faiss_id"]
            )

            if faiss_id in selected_ids:
                continue

            selected_result = dict(
                merged_results[faiss_id]
            )

            selected_result[
                "coverage_queries"
            ] = [retrieval_query]

            selected_results.append(
                selected_result
            )

            selected_ids.add(
                faiss_id
            )

            break

    for merged_result in globally_sorted_results:
        if len(selected_results) >= top_k:
            break

        faiss_id = int(
            merged_result["faiss_id"]
        )

        if faiss_id in selected_ids:
            continue

        selected_result = dict(
            merged_result
        )

        selected_result[
            "coverage_queries"
        ] = []

        selected_results.append(
            selected_result
        )

        selected_ids.add(
            faiss_id
        )

    selected_results.sort(
        key=lambda item: (
            item["score"],
            item["match_count"],
        ),
        reverse=True,
    )

    for rank, result in enumerate(
        selected_results,
        start=1,
    ):
        result["rank"] = rank

    return selected_results


def adaptive_retrieve(
    query: str,
    retriever: MedicalRetriever,
    candidate_k: int = RETRIEVAL_CANDIDATE_K,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    """Transform a query and retrieve its best unique chunks."""

    if candidate_k < 1:
        raise ValueError(
            "candidate_k must be at least 1."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    transformation = transform_query(
        query
    )

    retrieval_queries = select_retrieval_queries(
        transformation
    )

    grouped_results: list[
        tuple[str, list[dict[str, Any]]]
    ] = []

    for retrieval_query in retrieval_queries:
        query_results = retriever.search(
            query=retrieval_query,
            k=candidate_k,
        )

        grouped_results.append(
            (
                retrieval_query,
                query_results,
            )
        )

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
        **transformation,
        "retrieval_queries": retrieval_queries,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "retrieved_candidate_count": sum(
            len(results)
            for _, results in grouped_results
        ),
        "unique_candidate_count": len(
            {
                result["faiss_id"]
                for _, results in grouped_results
                for result in results
            }
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