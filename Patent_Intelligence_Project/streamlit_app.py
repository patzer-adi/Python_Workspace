"""Streamlit app for inspecting the patent retrieval pipeline."""

from pathlib import Path

import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.rag_search import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_PATH,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    open_collection,
    search,
)


st.set_page_config(page_title="Patent Intelligence", page_icon="🔎", layout="wide")
st.title("Patent Intelligence")
st.caption("Inspectable patent search over the existing ChromaDB index")


@st.cache_resource(show_spinner="Opening the patent collection…")
def load_collection(path: str, collection_name: str):
    return open_collection(path, collection_name)


@st.cache_resource(show_spinner="Loading MiniLM query encoder…")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading cross-encoder reranker…")
def load_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def _chunk_label(chunk: dict) -> str:
    chunk_index = chunk.get("chunk_index", "?")
    embedding_rank = chunk.get("embedding_rank", "?")
    base = f"Chunk {chunk_index} · embedding rank {embedding_rank} · similarity {chunk.get('embedding_similarity', 0.0):.4f}"
    if "reranker_score" in chunk:
        base += f" · reranker rank {chunk.get('reranker_rank', '?')} · score {chunk.get('reranker_score', 0.0):.4f}"
    return base


with st.sidebar:
    st.header("Search settings")
    db_path = st.text_input("ChromaDB directory", value=str(DEFAULT_DB_PATH))
    collection_name = st.text_input("Collection name", value=DEFAULT_COLLECTION_NAME)
    result_count = st.slider("Patent results", min_value=3, max_value=20, value=10)
    candidate_count = st.slider("Candidate chunks", min_value=10, max_value=100, value=50, step=10)
    use_reranker = st.toggle("Use cross-encoder reranking", value=False)
    st.caption("Reranker scores are raw logits, not probabilities.")

query = st.text_area(
    "Describe the invention or feature you are looking for",
    placeholder="For example: a device for holding cylindrical components during surface treatment while minimizing contact area",
    height=120,
)

search_clicked = st.button("Search patents", type="primary", disabled=not query.strip())

if search_clicked:
    st.session_state.pop("search_response", None)
    st.session_state.pop("search_query", None)
    try:
        collection = load_collection(str(Path(db_path).expanduser().resolve()), collection_name)
        embedding_model = load_embedding_model()
        reranker = load_reranker() if use_reranker else None
        with st.spinner("Searching patent evidence…"):
            response = search(
                query=query,
                collection=collection,
                embedding_model=embedding_model,
                top_k=result_count,
                candidate_count=candidate_count,
                reranker=reranker,
            )
        st.session_state["search_response"] = response
        st.session_state["search_query"] = query
    except Exception as exc:
        st.session_state.pop("search_response", None)
        st.error(f"Search could not be completed: {exc}")

response = st.session_state.get("search_response")
if response:
    st.subheader("Results")
    st.caption(f"Query: {st.session_state.get('search_query', '')}")

    timings = response["timings"]
    metric_cols = st.columns(5)
    metric_cols[0].metric("Collection chunks", f"{response['collection_count']:,}")
    metric_cols[1].metric("Candidates", response["candidate_count"])
    metric_cols[2].metric("Patents shown", len(response["results"]))
    metric_cols[3].metric("Embedding", _format_seconds(timings.get("embedding")))
    metric_cols[4].metric("Vector search", _format_seconds(timings.get("vector_search")))
    if "reranking" in timings:
        st.caption(f"Cross-encoder reranking: {_format_seconds(timings.get('reranking'))}")

    pipeline = response.get("pipeline", {})
    flow_cols = st.columns(4)
    flow_cols[0].metric("Retrieved chunks", pipeline.get("retrieved_candidates", 0))
    flow_cols[1].metric("Unique patents", pipeline.get("unique_patents", 0))
    flow_cols[2].metric("Returned patents", pipeline.get("returned_patents", 0))
    flow_cols[3].metric("Reranked", "Yes" if pipeline.get("reranked") else "No")

    if not response["results"]:
        st.info("The selected collection has no indexed chunks.")
    else:
        result_rows = []
        for patent in response["results"]:
            row = {
                "rank": patent["rank"],
                "patent_number": patent["patent_number"],
                "title": patent["title"],
                "score": round(float(patent["score"]), 4),
                "evidence_chunks": patent["retrieved_chunk_count"],
                "max_similarity": round(float(patent.get("max_similarity", 0.0)), 4),
                "top3_mean_similarity": round(float(patent.get("top3_mean_similarity", 0.0)), 4),
            }
            if "max_reranker_score" in patent:
                row["max_reranker_score"] = round(float(patent["max_reranker_score"]), 4)
            result_rows.append(row)

        st.dataframe(result_rows, use_container_width=True, hide_index=True)

        for patent in response["results"]:
            label = f"{patent['rank']}. {patent['patent_number']} — {patent['title']}"
            with st.expander(label):
                summary_cols = st.columns(5)
                summary_cols[0].metric("Patent score", f"{patent['score']:.4f}")
                summary_cols[1].metric("Max similarity", f"{patent.get('max_similarity', 0.0):.4f}")
                summary_cols[2].metric("Top-3 mean", f"{patent.get('top3_mean_similarity', 0.0):.4f}")
                summary_cols[3].metric("Evidence chunks", patent["retrieved_chunk_count"])
                summary_cols[4].metric("Decision", patent.get("decision") or "Not recorded")

                if "max_reranker_score" in patent:
                    rerank_cols = st.columns(2)
                    rerank_cols[0].metric("Max reranker score", f"{patent['max_reranker_score']:.4f}")
                    rerank_cols[1].metric("Top-3 mean reranker score", f"{patent.get('top3_mean_reranker_score', 0.0):.4f}")

                st.write(f"Retrieved {patent['retrieved_chunk_count']} chunk(s) for this patent; showing the top 3 evidence chunks.")
                for chunk in patent["chunks"]:
                    with st.container(border=True):
                        st.markdown(f"**{_chunk_label(chunk)}**")
                        st.write(chunk["document"])

    with st.expander("Pipeline inspection"):
        inspection_cols = st.columns(3)
        inspection_cols[0].write(f"**Collection:** {response['collection_name']}")
        inspection_cols[1].write(f"**Index path:** {response['collection_path']}")
        inspection_cols[2].write(f"**Embedding model:** {response['embedding_model']}")
        st.write(f"**Reranker model:** {response.get('reranker_model') or 'Disabled'}")
        st.write("**Stage timings**")
        st.json(timings)

        st.write("**Top retrieved chunks**")
        top_chunks = []
        for chunk in response["candidate_chunks"]:
            row = {
                "embedding_rank": chunk.get("embedding_rank"),
                "patent_number": chunk.get("patent_number"),
                "chunk_index": chunk.get("chunk_index"),
                "embedding_similarity": round(float(chunk.get("embedding_similarity", 0.0)), 4),
            }
            if "reranker_score" in chunk:
                row["reranker_rank"] = chunk.get("reranker_rank")
                row["reranker_score"] = round(float(chunk.get("reranker_score", 0.0)), 4)
            top_chunks.append(row)
        st.dataframe(top_chunks, use_container_width=True, hide_index=True)

else:
    st.info("Enter a query and search to inspect ranked patents and their source chunks.")
