from __future__ import annotations

from typing import Any

from adaptive_rag.config import (
    RETRIEVAL_MIN_ACCEPTED_RESULTS,
    RETRIEVAL_MIN_AVERAGE_SCORE,
    RETRIEVAL_SCORE_THRESHOLD,
)


def evaluate_retrieval(
    retrieval_output: dict[str, Any],
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    min_accepted_results: int = RETRIEVAL_MIN_ACCEPTED_RESULTS,
    min_average_score: float = RETRIEVAL_MIN_AVERAGE_SCORE,
) -> dict[str, Any]:
    """Evaluate retrieval quality using scores and query coverage."""

    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError(
            "score_threshold must be between 0 and 1."
        )

    if min_accepted_results < 1:
        raise ValueError(
            "min_accepted_results must be at least 1."
        )

    if not 0.0 <= min_average_score <= 1.0:
        raise ValueError(
            "min_average_score must be between 0 and 1."
        )

    original_results = retrieval_output.get(
        "results",
        [],
    )

    evaluated_results: list[dict[str, Any]] = []

    for result in original_results:
        evaluated_result = dict(result)

        evaluated_result["accepted"] = (
            float(result["score"])
            >= score_threshold
        )

        evaluated_results.append(
            evaluated_result
        )

    scores = [
        float(result["score"])
        for result in evaluated_results
    ]

    accepted_results = [
        result
        for result in evaluated_results
        if result["accepted"]
    ]

    result_count = len(
        evaluated_results
    )

    accepted_count = len(
        accepted_results
    )

    top_score = (
        max(scores)
        if scores
        else 0.0
    )

    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    retrieval_queries = list(
        dict.fromkeys(
            retrieval_output.get(
                "retrieval_queries",
                [],
            )
        )
    )

    covered_queries = set(
        retrieval_output.get(
            "covered_retrieval_queries",
            [],
        )
    )

    coverage_count = sum(
        query in covered_queries
        for query in retrieval_queries
    )

    coverage_total = len(
        retrieval_queries
    )

    coverage_ratio = (
        coverage_count / coverage_total
        if coverage_total > 0
        else 0.0
    )

    has_results = result_count > 0

    accepted_count_passed = (
        accepted_count
        >= min_accepted_results
    )

    average_score_passed = (
        average_score
        >= min_average_score
    )

    coverage_passed = (
        coverage_total > 0
        and coverage_count
        == coverage_total
    )

    quality_passed = all(
        [
            has_results,
            accepted_count_passed,
            average_score_passed,
            coverage_passed,
        ]
    )

    failed_reasons: list[str] = []

    if not has_results:
        failed_reasons.append(
            "No retrieval results were returned."
        )

    if not accepted_count_passed:
        failed_reasons.append(
            "The number of accepted results is below "
            "the configured minimum."
        )

    if not average_score_passed:
        failed_reasons.append(
            "The average similarity score is below "
            "the configured minimum."
        )

    if not coverage_passed:
        failed_reasons.append(
            "One or more retrieval queries are not "
            "represented in the final results."
        )

    if quality_passed:
        quality_status = "pass"
        quality_reason = (
            "Retrieval quality passed all configured checks."
        )
    else:
        quality_status = "fail"
        quality_reason = " ".join(
            failed_reasons
        )

    return {
        **retrieval_output,
        "results": evaluated_results,
        "evaluation": {
            "quality_status": quality_status,
            "quality_passed": quality_passed,
            "quality_reason": quality_reason,
            "result_count": result_count,
            "accepted_count": accepted_count,
            "rejected_count": (
                result_count - accepted_count
            ),
            "top_score": top_score,
            "average_score": average_score,
            "coverage_count": coverage_count,
            "coverage_total": coverage_total,
            "coverage_ratio": coverage_ratio,
            "score_threshold": score_threshold,
            "min_accepted_results": min_accepted_results,
            "min_average_score": min_average_score,
            "checks": {
                "has_results": has_results,
                "accepted_count_passed": (
                    accepted_count_passed
                ),
                "average_score_passed": (
                    average_score_passed
                ),
                "coverage_passed": coverage_passed,
            },
        },
    }