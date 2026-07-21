from __future__ import annotations

import re
from typing import Any

from adaptive_rag.config import (
    QUERY_COMPLEX_MIN_ACTIONS,
    QUERY_COMPLEX_MIN_CONNECTORS,
    QUERY_SHORT_MAX_WORDS,
)


ACTION_VERBS = {
    "diagnose",
    "diagnoses",
    "detect",
    "detects",
    "recommend",
    "recommends",
    "select",
    "selects",
    "classify",
    "classifies",
    "assess",
    "assesses",
    "estimate",
    "estimates",
    "treat",
    "treats",
    "manage",
    "manages",
    "predict",
    "predicts",
}

DOMAIN_NOUNS = {
    "diagnosis",
    "detection",
    "recommendation",
    "selection",
    "classification",
    "assessment",
    "estimation",
    "treatment",
    "management",
    "prediction",
}

CONNECTOR_TERMS = {
    "and",
    "or",
    "also",
    "while",
    "versus",
    "vs",
}


def normalize_query(query: str) -> str:
    """Remove unnecessary whitespace from an English query."""

    return re.sub(
        r"\s+",
        " ",
        query,
    ).strip()


def tokenize_query(query: str) -> list[str]:
    """Extract lowercase English words from a query."""

    return re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?",
        query.lower(),
    )


def analyze_query(
    query: str,
) -> dict[str, Any]:
    """Classify a query as expansion, decomposition, or direct."""

    normalized_query = normalize_query(query)

    if not normalized_query:
        raise ValueError(
            "The query cannot be empty."
        )

    words = tokenize_query(
        normalized_query
    )

    word_count = len(words)

    action_verbs_found = sorted(
        {
            word
            for word in words
            if word in ACTION_VERBS
        }
    )

    domain_nouns_found = sorted(
        {
            word
            for word in words
            if word in DOMAIN_NOUNS
        }
    )

    connector_terms_found = [
        word
        for word in words
        if word in CONNECTOR_TERMS
    ]

    action_count = len(
        action_verbs_found
    )

    connector_count = len(
        connector_terms_found
    )

    has_multiple_question_parts = (
        normalized_query.count("?") > 1
        or ";" in normalized_query
    )

    if word_count <= QUERY_SHORT_MAX_WORDS:
        route = "expansion"
        reason = (
            "The query is short and may not contain enough "
            "retrieval context."
        )

    elif (
        action_count >= QUERY_COMPLEX_MIN_ACTIONS
        or connector_count >= QUERY_COMPLEX_MIN_CONNECTORS
        or has_multiple_question_parts
    ):
        route = "decomposition"
        reason = (
            "The query contains multiple action verbs or "
            "connected information needs."
        )

    else:
        route = "direct"
        reason = (
            "The query is sufficiently specific and contains "
            "one primary information need."
        )

    return {
        "original_query": query,
        "normalized_query": normalized_query,
        "route": route,
        "reason": reason,
        "metrics": {
            "word_count": word_count,
            "action_count": action_count,
            "connector_count": connector_count,
            "action_verbs": action_verbs_found,
            "domain_nouns": domain_nouns_found,
            "connector_terms": connector_terms_found,
            "multiple_question_parts": has_multiple_question_parts,
        },
    }