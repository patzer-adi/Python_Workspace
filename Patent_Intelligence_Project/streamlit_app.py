"""Patent Intelligence — AI-assisted patent discovery over the existing ChromaDB index.

Stage 1: clean search UI, TXT upload, evidence-backed result cards.
Retrieval is delegated to ``src.rag_search.search`` (unchanged).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.document_processor import (
    MAX_SECTIONS_PER_SEARCH,
    DocumentError,
    chunk_text,
    extract_text,
    select_sections,
)

st.set_page_config(page_title="Patent Intelligence", page_icon="🔎", layout="centered")

from src.theme import apply_theme, render_nav

apply_theme()
render_nav("search")

# --------------------------------------------------------------------------- #
# Configuration (env vars or Streamlit secrets; never hardcoded)
# --------------------------------------------------------------------------- #

def _config(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets file configured
        return default


try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from src.rag_search import (
        DEFAULT_COLLECTION_NAME,
        DEFAULT_DB_PATH,
        EMBEDDING_MODEL_NAME,
        RERANKER_MODEL_NAME,
        open_collection,
        search,
    )
    from src.patent_search import search_document
    from src.llm_service import ACTIONS, LLMError, followup, load_config, run_action
except Exception as exc:  # surface import problems in the UI instead of a stack trace
    st.title("Patent Intelligence")
    st.error(f"The retrieval backend could not be imported: {exc}")
    st.stop()

DB_PATH = _config("PATENT_DB_PATH", str(DEFAULT_DB_PATH))
COLLECTION = _config("PATENT_COLLECTION", DEFAULT_COLLECTION_NAME)
LLM_CFG = load_config(_config)
MAX_LLM_CALLS = int(_config("LLM_MAX_CALLS_PER_SESSION", "25") or 25)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 820px;}
      .pi-sub {color: #5b6470; font-size: 1.05rem; margin-top: -0.6rem;}
      .pi-why {color: #5b6470; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Opening the patent collection…")
def load_collection(path: str, collection_name: str):
    return open_collection(path, collection_name)


@st.cache_resource(show_spinner="Loading the query encoder…")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading the reranker…")
def load_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


# --------------------------------------------------------------------------- #
# Search execution
# --------------------------------------------------------------------------- #

def run_search(mode: str, text: str, top_k: int, candidates: int, use_reranker: bool) -> None:
    for key in ("response", "query", "mode", "chat"):
        st.session_state.pop(f"pi_{key}", None)
    db = Path(str(DB_PATH)).expanduser().resolve()
    if not db.exists():
        st.error(
            f"The patent index was not found at `{db}`. Set `PATENT_DB_PATH` "
            "(environment variable or Streamlit secret) to your ChromaDB folder."
        )
        return
    try:
        collection = load_collection(str(db), str(COLLECTION))
        encoder = load_embedding_model()
        reranker = load_reranker() if use_reranker else None
        with st.spinner("Searching patent evidence…"):
            if mode == "document":
                sections = select_sections(chunk_text(text), MAX_SECTIONS_PER_SEARCH)
                response = search_document(
                    sections,
                    collection=collection,
                    embedding_model=encoder,
                    top_k=top_k,
                    candidate_count=candidates,
                    reranker=reranker,
                    search_fn=search,
                )
            else:
                response = search(
                    query=text,
                    collection=collection,
                    embedding_model=encoder,
                    top_k=top_k,
                    candidate_count=candidates,
                    reranker=reranker,
                )
                response["mode"] = "query"
        st.session_state.update(pi_response=response, pi_query=text, pi_mode=mode)
    except Exception as exc:
        st.error(f"Search could not be completed: {exc}")


# --------------------------------------------------------------------------- #
# Result presentation helpers
# --------------------------------------------------------------------------- #

def _strengths(results: list[dict]) -> dict[str, tuple[str, float]]:
    """Relative (not absolute) match strength per patent, versus the other results."""
    scores = [float(p.get("score", 0.0)) for p in results]
    if not scores:
        return {}
    lo, hi = min(scores), max(scores)
    out: dict[str, tuple[str, float]] = {}
    for patent, score in zip(results, scores):
        if lo > 0:
            ratio = score / hi
        else:  # reranker logits can be negative: fall back to min-max
            ratio = (score - lo) / (hi - lo) if hi > lo else 1.0
        ratio = max(0.05, min(ratio, 1.0))
        label = "Strongest match" if ratio >= 0.999 else "Strong" if ratio >= 0.85 else "Good" if ratio >= 0.65 else "Moderate"
        out[str(patent.get("patent_number"))] = (label, ratio)
    return out


def _excerpt(text: str, limit: int = 420) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _why(patent: dict, mode: str, sections_total: int) -> str:
    n = patent.get("retrieved_chunk_count", 0)
    parts = [f"{n} matching passage{'s' if n != 1 else ''} retrieved from this patent"]
    if mode == "document":
        parts.append(f"relevant to {patent.get('matched_sections', 1)} of {sections_total} sections of your document")
    return "; ".join(parts) + "."


def render_card(patent: dict, strength: tuple[str, float], mode: str, sections_total: int) -> None:
    number = patent.get("patent_number", "?")
    chunks = patent.get("chunks", [])
    with st.container(border=True):
        head, pick = st.columns([5, 2])
        head.markdown(f"**{patent['rank']}. {patent.get('title') or 'Untitled'}**")
        head.caption(
            f"Patent {number}" + (f" · Decision/class: {patent['decision']}" if patent.get("decision") else "")
        )
        pick.checkbox("Select to compare", key=f"pi_cmp_{number}")
        if pick.button("Open in Patent Graph", key=f"pi_graph_{number}"):
            from src.patent_sbert_embeddings import normalize_id

            st.session_state["pg_action"] = ("patent", normalize_id(number))
            st.switch_page("pages/1_Patent_Graph.py")

        label, ratio = strength
        st.progress(ratio, text=f"Relative match: {label}")

        if chunks:
            st.markdown(f"> {_excerpt(chunks[0].get('document', ''))}")
        st.markdown(f"<div class='pi-why'>Why retrieved: {_why(patent, mode, sections_total)}</div>", unsafe_allow_html=True)

        if chunks:
            with st.expander("Inspect evidence"):
                for chunk in chunks[:3]:
                    meta = f"Passage {chunk.get('chunk_index', '?')}"
                    if mode == "document" and chunk.get("matched_section"):
                        meta += f" · matched your section {chunk['matched_section']}"
                    st.caption(meta)
                    st.write(chunk.get("document", ""))


def render_assistant(results: list[dict], selected: list[str]) -> None:
    st.subheader("Go deeper")
    if LLM_CFG is None:
        st.info(
            "AI explanations are off. Add `GEMINI_API_KEY` or `OPENROUTER_API_KEY` to the app's secrets "
            "to enable them. Search and evidence browsing work without it."
        )
        return

    chat: list[dict] = st.session_state.setdefault("pi_chat", [])
    description = st.session_state.get("pi_query", "")
    pending: tuple[str, str] | None = None  # (kind, key-or-question)

    st.caption(
        "Answers are generated from the retrieved passages only and labelled as hypotheses where they go beyond them. "
        "Your text and the passages are sent to the configured AI provider "
        f"({LLM_CFG.provider.capitalize()}). Tick cards to focus the analysis on specific patents."
    )
    keys = list(ACTIONS)
    for row in (keys[:3], keys[3:]):
        for col, key in zip(st.columns(len(row)), row):
            if col.button(ACTIONS[key].label, key=f"act_{key}", use_container_width=True):
                pending = ("action", key)
    question = st.chat_input("Ask a follow-up about these results…")
    if question:
        pending = ("followup", question)

    if pending:
        if st.session_state.get("pi_llm_calls", 0) >= MAX_LLM_CALLS:
            st.warning("This session has reached its AI request limit. Reload the page to start a new session.")
        else:
            kind, value = pending
            history = [{"role": m["role"], "content": m["content"]} for m in chat]
            try:
                with st.spinner("Analysing the retrieved evidence…"):
                    if kind == "action":
                        answer = run_action(LLM_CFG, value, description, results, selected, history)
                        user_text = ACTIONS[value].label
                    else:
                        answer = followup(LLM_CFG, value, description, results, selected, history)
                        user_text = value
                st.session_state["pi_llm_calls"] = st.session_state.get("pi_llm_calls", 0) + 1
                chat.append({"role": "user", "content": user_text})
                chat.append({"role": "assistant", "content": answer.text, "patents": answer.patents_used})
            except LLMError as exc:
                st.error(str(exc))

    for message in chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("patents"):
                st.caption("Evidence used: " + ", ".join(f"#{n}" for n in message["patents"]))


def render_developer_details(response: dict) -> None:
    with st.expander("Developer details"):
        st.write(f"**Collection:** {response.get('collection_name')}  \n**Index path:** {response.get('collection_path')}")
        st.write(
            f"**Embedding model:** {response.get('embedding_model')}  \n"
            f"**Reranker:** {response.get('reranker_model') or 'Disabled'}"
        )
        if response.get("mode") == "document":
            st.write(
                f"**Mode:** document search — {response.get('sections_searched')} section(s) searched, "
                "rankings merged with reciprocal rank fusion (patent `score` = RRF value, not a similarity)."
            )
            for err in response.get("section_errors", []):
                st.warning(err)
        st.write(f"**Collection chunks:** {response.get('collection_count', 0):,}")
        st.write("**Pipeline**")
        st.json(response.get("pipeline", {}))
        st.write("**Stage timings (s)**")
        st.json(response.get("timings", {}))
        rows = []
        for chunk in response.get("candidate_chunks", [])[:100]:
            row = {
                "patent_number": chunk.get("patent_number"),
                "chunk_index": chunk.get("chunk_index"),
                "embedding_rank": chunk.get("embedding_rank"),
                "embedding_similarity": round(float(chunk.get("embedding_similarity", 0.0)), 4),
            }
            if "reranker_score" in chunk:
                row["reranker_score (logit)"] = round(float(chunk["reranker_score"]), 4)
            rows.append(row)
        st.write("**Retrieved chunks** (first 100)")
        st.dataframe(rows, hide_index=True)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

st.title("Patent Intelligence")
st.markdown(
    "<p class='pi-sub'>Describe an invention in plain language, or upload a short technical "
    "description, and discover the most relevant patents with the passages that support each match.</p>",
    unsafe_allow_html=True,
)

with st.expander("Search options"):
    top_k = st.slider("Patents to show", 3, 20, 10)
    candidates = st.slider("Candidate passages to scan", 10, 100, 50, step=10)
    use_reranker = st.toggle("Refine ranking with the cross-encoder (slower, uses more memory)", value=False)

tab_text, tab_file = st.tabs(["Describe your invention", "Upload a TXT file"])

with tab_text:
    query = st.text_area(
        "What are you looking for?",
        placeholder="e.g. A device for holding cylindrical components during surface treatment while minimizing contact area",
        height=130,
        label_visibility="collapsed",
    )
    if st.button("Search patents", type="primary", disabled=not query.strip(), key="go_text"):
        run_search("query", query.strip(), top_k, candidates, use_reranker)

with tab_file:
    st.caption(
        "Your file is read in memory for this session only. It is **not** stored, "
        "and it is **not** added to the patent index. Text sections are sent to the search "
        "engine running in this app; no external service is used at this stage."
    )
    upload = st.file_uploader("TXT file", type=["txt"], label_visibility="collapsed")
    if upload is not None:
        signature = (upload.name, upload.size)
        if st.session_state.get("pi_upload_sig") != signature:
            st.session_state["pi_upload_sig"] = signature
            try:
                st.session_state["pi_doc_text"] = extract_text(upload.name, upload.getvalue())
                st.session_state.pop("pi_upload_error", None)
            except DocumentError as exc:
                st.session_state["pi_doc_text"] = ""
                st.session_state["pi_upload_error"] = str(exc)
        if st.session_state.get("pi_upload_error"):
            st.error(st.session_state["pi_upload_error"])
        else:
            doc_text = st.text_area(
                "Preview — edit before searching",
                key="pi_doc_text",
                height=220,
            )
            n_chunks = len(chunk_text(doc_text)) if doc_text.strip() else 0
            used = min(n_chunks, MAX_SECTIONS_PER_SEARCH)
            st.caption(f"{len(doc_text):,} characters · {n_chunks} section(s)" + (f" · {used} sampled for search" if n_chunks > used else ""))
            if st.button("Search with this document", type="primary", disabled=not doc_text.strip(), key="go_doc"):
                run_search("document", doc_text.strip(), top_k, candidates, use_reranker)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

response = st.session_state.get("pi_response")
if response:
    mode = st.session_state.get("pi_mode", "query")
    results = response.get("results", [])
    st.divider()
    st.subheader("Relevant patents")
    if not results:
        st.info("No patents were retrieved. Try describing the invention in more technical detail.")
    else:
        sections_total = response.get("sections_searched", 1)
        strengths = _strengths(results)
        selected = [
            str(p.get("patent_number")) for p in results if st.session_state.get(f"pi_cmp_{p.get('patent_number')}")
        ]
        st.caption(
            "Ranked by semantic similarity of retrieved passages. Match strength is relative to the best result "
            "and is not a probability."
            + (f" · {len(selected)} selected for comparison" if selected else "")
        )
        for patent in results:
            render_card(patent, strengths[str(patent.get("patent_number"))], mode, sections_total)
        render_assistant(results, selected)
    render_developer_details(response)
