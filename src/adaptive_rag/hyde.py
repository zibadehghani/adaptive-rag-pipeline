from __future__ import annotations

from typing import Any

from adaptive_rag.query_transformer import (
    transform_query,
)


def _clean_focus_item(
    text: str,
) -> str:
    """Remove surrounding whitespace and trailing punctuation."""

    return text.strip().rstrip(
        "?.!,;:"
    )


def _select_focus_items(
    transformation: dict[str, Any],
) -> list[str]:
    """Select and clean concepts for the hypothetical passage."""

    if transformation["subqueries"]:
        raw_focus_items = transformation[
            "subqueries"
        ]

    elif transformation["added_terms"]:
        raw_focus_items = [
            transformation["normalized_query"],
            *transformation["added_terms"],
        ]

    else:
        raw_focus_items = [
            transformation["normalized_query"]
        ]

    cleaned_focus_items: list[str] = []

    for item in raw_focus_items:
        cleaned_item = _clean_focus_item(
            item
        )

        if (
            cleaned_item
            and cleaned_item
            not in cleaned_focus_items
        ):
            cleaned_focus_items.append(
                cleaned_item
            )

    return cleaned_focus_items


def generate_hyde_document(
    query: str,
) -> dict[str, Any]:
    """Generate a deterministic template-based HyDE passage."""

    transformation = transform_query(
        query
    )

    focus_items = _select_focus_items(
        transformation
    )

    focus_text = "; ".join(
        focus_items
    )

    hypothetical_document = (
        "A medical decision-support study addresses the "
        f"clinical information need: "
        f"{transformation['normalized_query']} "
        f"The study focuses on {focus_text}. "
        "A medical expert system evaluates patient symptoms, "
        "clinical findings, measurements, medications, and "
        "risk factors. It compares the available evidence with "
        "encoded medical knowledge, decision rules, fuzzy logic, "
        "or trained predictive models. The system identifies "
        "relevant clinical patterns and supports diagnosis, "
        "treatment recommendation, or patient risk classification. "
        "The resulting decision is intended to be interpretable "
        "and supported by the most relevant clinical evidence."
    )

    return {
        **transformation,
        "hyde_method": "template_based_hyde",
        "generation_source": "deterministic_template",
        "focus_items": focus_items,
        "hypothetical_document": hypothetical_document,
        "hyde_document_word_count": len(
            hypothetical_document.split()
        ),
    }