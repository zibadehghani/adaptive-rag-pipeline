from __future__ import annotations

import json
from typing import Any

import streamlit as st

from adaptive_rag.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_MIN_ACCEPTED_RESULTS,
    RETRIEVAL_MIN_AVERAGE_SCORE,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from adaptive_rag.embeddings import (
    Qwen3Embeddings,
)
from adaptive_rag.graph_workflow import (
    create_adaptive_rag_graph,
)
from adaptive_rag.retrieval import (
    MedicalRetriever,
)


st.set_page_config(
    page_title="Adaptive RAG Explorer",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #102a43;
        --navy-hover: #163b65;
        --app-bg: #d6dbe5;
        --sidebar-bg: #c7cfdb;
        --input-bg: #dde3ec;
        --code-bg: #cfd8e3;
        --divider: #8b98ab;
        --text-dark: #0f172a;
        --white: #ffffff;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: var(--app-bg) !important;
    }

    header[data-testid="stHeader"],
    .stAppHeader,
    [data-testid="stToolbar"] {
        background-color: var(--app-bg) !important;
    }

    [data-testid="stMain"] {
        background-color: var(--app-bg) !important;
    }


    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background-color: var(--sidebar-bg) !important;
    }

    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="input"] input {
        background-color: var(--input-bg) !important;
        color: var(--text-dark) !important;
    }

    div[data-baseweb="textarea"] {
        border-color: transparent !important;
        box-shadow: none !important;
    }

    div[data-baseweb="textarea"]:focus-within {
        border-color: var(--navy) !important;
        box-shadow: 0 0 0 1px var(--navy) !important;
    }

    .stTextArea textarea:focus,
    .stTextArea textarea:focus-visible {
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div.stButton > button {
        background-color: var(--navy) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        background-color: var(--navy-hover) !important;
        color: var(--white) !important;
        border-color: var(--navy-hover) !important;
    }

    .stNumberInput button {
        background-color: var(--white) !important;
        color: var(--text-dark) !important;
        border-color: transparent !important;
        transition: all 0.2s ease !important;
    }

    .stNumberInput button:hover {
        background-color: var(--navy) !important;
        color: var(--white) !important;
        border-color: var(--navy) !important;
    }

    .stNumberInput button:active,
    .stNumberInput button:focus,
    .stNumberInput button:focus-visible {
        background-color: var(--navy) !important;
        color: var(--white) !important;
        border-color: var(--navy) !important;
        box-shadow: none !important;
        outline: none !important;
    }

    .stNumberInput button:hover svg,
    .stNumberInput button:active svg,
    .stNumberInput button:focus svg {
        fill: var(--white) !important;
        color: var(--white) !important;
    }

    hr {
        border-color: var(--divider) !important;
    }

    code {
        background-color: var(--code-bg) !important;
        color: #0f2747 !important;
    }
    
    div[data-testid="stTextAreaRootElement"] {
    border: 1px solid #8b98ab !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

    div[data-testid="stTextAreaRootElement"]:hover {
    border-color: #102a43 !important;
}

    div[data-testid="stTextAreaRootElement"]:focus-within {
    border: 1px solid #102a43 !important;
    box-shadow: 0 0 0 1px #102a43 !important;
    outline: none !important;
}

    div[data-testid="stTextAreaRootElement"] textarea,
    div[data-testid="stTextAreaRootElement"] textarea:hover,
    div[data-testid="stTextAreaRootElement"] textarea:focus,
    div[data-testid="stTextAreaRootElement"] textarea:focus-visible {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_retriever() -> MedicalRetriever:
    """Load the embedding model and FAISS index once."""

    embedding_model = Qwen3Embeddings()

    return MedicalRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH,
        embedding_model=embedding_model,
    )


def format_score(
    score: float,
) -> str:
    """Format a similarity score for display."""

    return f"{float(score):.4f}"


def render_query_transformation(
    output: dict[str, Any],
) -> None:
    """Display query analysis and transformation information."""

    st.subheader("Query Analysis and Transformation")

    st.write(
        f"**Original query:** "
        f"{output['original_query']}"
    )

    st.write(
        f"**Selected route:** "
        f"`{output['route']}`"
    )

    st.write(
        f"**Routing reason:** "
        f"{output['reason']}"
    )

    st.write(
        f"**Strategy:** "
        f"`{output['strategy']}`"
    )

    st.write(
        f"**Transformed query:** "
        f"{output['transformed_query']}"
    )

    added_terms = output.get(
        "added_terms",
        [],
    )

    if added_terms:
        st.write(
            "**Expansion terms:**",
            added_terms,
        )

    subqueries = output.get(
        "subqueries",
        [],
    )

    if subqueries:
        st.write("**Generated subqueries:**")

        for index, subquery in enumerate(
            subqueries,
            start=1,
        ):
            st.write(
                f"{index}. {subquery}"
            )

    with st.expander(
        "Query analysis metrics"
    ):
        st.json(
            output.get(
                "metrics",
                {},
            )
        )


def render_quality_comparison(
    output: dict[str, Any],
) -> None:
    """Compare initial and final retrieval quality."""

    initial_evaluation = output[
        "initial_output"
    ]["evaluation"]

    final_evaluation = output[
        "final_output"
    ]["evaluation"]

    st.subheader("Retrieval Quality")

    initial_column, final_column = st.columns(
        2
    )

    with initial_column:
        st.markdown("#### Initial Retrieval")

        st.metric(
            "Quality",
            initial_evaluation[
                "quality_status"
            ].upper(),
        )

        st.metric(
            "Average score",
            format_score(
                initial_evaluation[
                    "average_score"
                ]
            ),
        )

        st.write(
            f"Accepted results: "
            f"{initial_evaluation['accepted_count']}/"
            f"{initial_evaluation['result_count']}"
        )

        st.write(
            f"Coverage: "
            f"{initial_evaluation['coverage_count']}/"
            f"{initial_evaluation['coverage_total']}"
        )

        st.caption(
            initial_evaluation[
                "quality_reason"
            ]
        )

    with final_column:
        st.markdown("#### Final Retrieval")

        st.metric(
            "Quality",
            final_evaluation[
                "quality_status"
            ].upper(),
        )

        st.metric(
            "Average score",
            format_score(
                final_evaluation[
                    "average_score"
                ]
            ),
        )

        st.write(
            f"Accepted results: "
            f"{final_evaluation['accepted_count']}/"
            f"{final_evaluation['result_count']}"
        )

        st.write(
            f"Coverage: "
            f"{final_evaluation['coverage_count']}/"
            f"{final_evaluation['coverage_total']}"
        )

        st.caption(
            final_evaluation[
                "quality_reason"
            ]
        )


def render_hyde_information(
    output: dict[str, Any],
) -> None:
    """Display template-based HyDE information when used."""

    if not output["hyde_used"]:
        st.info(
            "Initial retrieval passed. "
            "The HyDE fallback was not required."
        )
        return

    hyde_generation = output[
        "hyde_generation"
    ]

    st.warning(
        "Initial retrieval failed the quality checks. "
        "Template-based HyDE was used once."
    )

    with st.expander(
        "Template-based HyDE details",
        expanded=True,
    ):
        st.write(
            f"**Method:** "
            f"`{hyde_generation['hyde_method']}`"
        )

        st.write(
            f"**Generation source:** "
            f"`{hyde_generation['generation_source']}`"
        )

        st.write(
            f"**Focus items:** "
            f"{hyde_generation['focus_items']}"
        )

        st.write(
            f"**Document word count:** "
            f"{hyde_generation['hyde_document_word_count']}"
        )

        st.text_area(
            "Hypothetical document",
            value=hyde_generation[
                "hypothetical_document"
            ],
            height=240,
            disabled=True,
        )


def render_result_table(
    final_output: dict[str, Any],
) -> None:
    """Display final retrieval results as a table and detailed cards."""

    results = final_output.get(
        "results",
        [],
    )

    st.subheader("Final Retrieved Chunks")

    if not results:
        st.warning(
            "No retrieval results were returned."
        )
        return

    table_rows = [
        {
            "Rank": result["rank"],
            "Score": round(
                float(result["score"]),
                4,
            ),
            "Accepted": result["accepted"],
            "Source": result["source_file"],
            "Page": result["page_number"],
            "Chunk": result["chunk_id"],
            "Matches": result.get(
                "match_count",
                1,
            ),
        }
        for result in results
    ]

    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Retrieved Text Details")

    for result in results:
        title = (
            f"Rank {result['rank']} — "
            f"{result['source_file']} — "
            f"Score {result['score']:.4f}"
        )

        with st.expander(title):
            st.write(
                f"**Accepted:** "
                f"{result['accepted']}"
            )

            st.write(
                f"**Page:** "
                f"{result['page_number']}"
            )

            st.write(
                f"**Chunk ID:** "
                f"`{result['chunk_id']}`"
            )

            st.write(
                f"**Best retrieval query:** "
                f"{result['best_retrieval_query']}"
            )

            matched_queries = result.get(
                "matched_queries",
                [],
            )

            if matched_queries:
                st.write(
                    "**Matched queries:**",
                    matched_queries,
                )

            coverage_queries = result.get(
                "coverage_queries",
                [],
            )

            if coverage_queries:
                st.write(
                    "**Coverage representative for:**",
                    coverage_queries,
                )

            st.markdown("**Retrieved text:**")

            st.write(
                result["text"]
            )


def render_trace_and_export(
    output: dict[str, Any],
) -> None:
    """Display the graph path and provide JSON export."""

    st.subheader("LangGraph Execution Trace")

    trace_text = " → ".join(
        output.get(
            "trace",
            [],
        )
    )

    st.code(
        trace_text,
        language=None,
    )

    export_data = json.dumps(
        output,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    st.download_button(
        label="Download run as JSON",
        data=export_data,
        file_name="adaptive_rag_run.json",
        mime="application/json",
        use_container_width=True,
    )


st.title("Adaptive RAG Query Enhancement Explorer")

st.caption(
    "Explainable retrieval over medical expert-system articles. "
    "This application displays retrieved evidence and graph decisions; "
    "it does not generate a final medical answer."
)

with st.sidebar:
    st.header("Retrieval Configuration")

    candidate_k = st.number_input(
        "Candidate K",
        min_value=1,
        max_value=50,
        value=RETRIEVAL_CANDIDATE_K,
        step=1,
    )

    top_k = st.number_input(
        "Final Top K",
        min_value=1,
        max_value=int(candidate_k),
        value=min(
            RETRIEVAL_TOP_K,
            int(candidate_k),
        ),
        step=1,
    )

    st.divider()

    st.write(
        f"Score threshold: "
        f"`{RETRIEVAL_SCORE_THRESHOLD}`"
    )

    st.write(
        f"Minimum accepted results: "
        f"`{RETRIEVAL_MIN_ACCEPTED_RESULTS}`"
    )

    st.write(
        f"Minimum average score: "
        f"`{RETRIEVAL_MIN_AVERAGE_SCORE}`"
    )

    st.divider()

    st.write(
        f"FAISS index: "
        f"`{FAISS_INDEX_PATH.name}`"
    )

    st.write(
        f"Metadata: "
        f"`{FAISS_METADATA_PATH.name}`"
    )


def clear_current_query() -> None:
    """Clear the current query and its displayed results."""

    st.session_state["query_input"] = ""

    st.session_state.pop(
        "adaptive_rag_output",
        None,
    )

    st.session_state.pop(
        "adaptive_rag_query",
        None,
    )


if "query_input" not in st.session_state:
    st.session_state["query_input"] = (
        "How does a fuzzy expert system select "
        "a wart treatment method?"
    )


query = st.text_area(
    "Enter an English medical expert-system query",
    height=110,
    key="query_input",
)


run_column, clear_column = st.columns(
    [4, 1]
)

with run_column:
    run_button = st.button(
        "Run Adaptive Retrieval",
        type="primary",
        use_container_width=True,
    )

with clear_column:
    st.button(
        "New Query",
        on_click=clear_current_query,
        use_container_width=True,
    )

if run_button:
    if not query.strip():
        st.warning(
            "Please enter a non-empty query."
        )

    else:
        try:
            with st.spinner(
                "Loading the model and running LangGraph..."
            ):
                retriever = load_retriever()

                graph = create_adaptive_rag_graph(
                    retriever=retriever,
                    candidate_k=int(candidate_k),
                    top_k=int(top_k),
                )

                graph_output = graph.invoke(
                    {
                        "original_query": query,
                        "trace": [],
                        "hyde_attempt_count": 0,
                    }
                )

            st.session_state[
                "adaptive_rag_output"
            ] = graph_output

            st.session_state[
                "adaptive_rag_query"
            ] = query

        except Exception as error:
            st.exception(error)


output = st.session_state.get(
    "adaptive_rag_output"
)

if output is not None:
    st.success(
        "Adaptive retrieval completed successfully."
    )

    summary_columns = st.columns(
        4
    )

    with summary_columns[0]:
        st.metric(
            "Route",
            output["route"].upper(),
        )

    with summary_columns[1]:
        st.metric(
            "Final stage",
            output[
                "final_stage"
            ].upper(),
        )

    with summary_columns[2]:
        st.metric(
            "Final quality",
            output[
                "final_output"
            ]["evaluation"][
                "quality_status"
            ].upper(),
        )

    with summary_columns[3]:
        st.metric(
            "HyDE used",
            (
                "YES"
                if output["hyde_used"]
                else "NO"
            ),
        )

    st.divider()

    render_query_transformation(
        output
    )

    st.divider()

    render_quality_comparison(
        output
    )

    st.divider()

    render_hyde_information(
        output
    )

    st.divider()

    render_result_table(
        output["final_output"]
    )

    st.divider()

    render_trace_and_export(
        output
    )