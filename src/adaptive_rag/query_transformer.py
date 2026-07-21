from __future__ import annotations

import re
from typing import Any

from adaptive_rag.query_analyzer import (
    ACTION_VERBS,
    analyze_query,
    normalize_query,
)


EXPANSION_RULES = {
    "medical expert systems": [
        "clinical decision support",
        "disease diagnosis",
        "treatment recommendation",
        "patient risk classification",
    ],
    "heart": [
        "cardiac diagnosis",
        "cardiovascular disease",
        "arrhythmia risk assessment",
    ],
    "wart": [
        "HPV",
        "cryotherapy",
        "immunotherapy",
        "treatment selection",
    ],
    "migraine": [
        "headache diagnosis",
        "migraine symptoms",
        "treatment recommendation",
    ],
    "diabetes": [
        "type 2 diabetes",
        "nutrition recommendation",
        "clinical decision support",
    ],
}

DEFAULT_EXPANSION_TERMS = [
    "clinical diagnosis",
    "treatment recommendation",
    "medical decision support",
]


def _remove_trailing_punctuation(
    text: str,
) -> str:
    """Remove punctuation from the end of a query."""

    return re.sub(
        r"[?.!,;:]+$",
        "",
        text,
    ).strip()


def _remove_leading_connector(
    text: str,
) -> str:
    """Remove a leading connector such as and, or, or also."""

    return re.sub(
        r"^(?:and|or|also)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def expand_query(
    query: str,
) -> dict[str, Any]:
    """Expand a short query with deterministic medical terms."""

    normalized_query = normalize_query(query)
    lowercase_query = normalized_query.lower()

    added_terms: list[str] = []

    for phrase, related_terms in EXPANSION_RULES.items():
        if phrase not in lowercase_query:
            continue

        for term in related_terms:
            term_already_present = (
                term.lower() in lowercase_query
                or term in added_terms
            )

            if not term_already_present:
                added_terms.append(term)

    if not added_terms:
        added_terms.extend(
            DEFAULT_EXPANSION_TERMS
        )

    transformed_query = " ".join(
        [
            normalized_query,
            *added_terms,
        ]
    )

    return {
        "strategy": "expansion",
        "transformed_query": transformed_query,
        "added_terms": added_terms,
        "subqueries": [],
    }


def decompose_query(
    query: str,
) -> dict[str, Any]:
    """Split a multi-action query into focused subqueries."""

    normalized_query = normalize_query(query)

    cleaned_query = _remove_trailing_punctuation(
        normalized_query
    )

    action_pattern = (
        r"\b("
        + "|".join(
            sorted(
                ACTION_VERBS,
                key=len,
                reverse=True,
            )
        )
        + r")\b"
    )

    first_action = re.search(
        action_pattern,
        cleaned_query,
        flags=re.IGNORECASE,
    )

    if first_action is None:
        return {
            "strategy": "decomposition",
            "transformed_query": normalized_query,
            "added_terms": [],
            "subqueries": [normalized_query],
        }

    shared_prefix = cleaned_query[
        : first_action.start()
    ].strip()

    action_section = cleaned_query[
        first_action.start() :
    ].strip()

    action_parts = re.split(
        r"\s*,\s*|\s+(?:and|or|also)\s+",
        action_section,
        flags=re.IGNORECASE,
    )

    subqueries: list[str] = []

    for action_part in action_parts:
        cleaned_part = _remove_trailing_punctuation(
            action_part
        )

        cleaned_part = _remove_leading_connector(
            cleaned_part
        )

        if not cleaned_part:
            continue

        subquery = " ".join(
            part
            for part in [
                shared_prefix,
                cleaned_part,
            ]
            if part
        )

        subquery = f"{subquery}?"

        if subquery not in subqueries:
            subqueries.append(subquery)

    return {
        "strategy": "decomposition",
        "transformed_query": normalized_query,
        "added_terms": [],
        "subqueries": subqueries,
    }


def transform_query(
    query: str,
) -> dict[str, Any]:
    """Analyze a query and apply its selected transformation."""

    analysis = analyze_query(query)

    route = analysis["route"]

    if route == "expansion":
        transformation = expand_query(
            analysis["normalized_query"]
        )

    elif route == "decomposition":
        transformation = decompose_query(
            analysis["normalized_query"]
        )

    else:
        transformation = {
            "strategy": "direct",
            "transformed_query": analysis[
                "normalized_query"
            ],
            "added_terms": [],
            "subqueries": [],
        }

    return {
        **analysis,
        **transformation,
    }