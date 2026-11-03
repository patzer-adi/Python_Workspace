"""
Patent Intelligence - unified Streamlit app.

Views (one session-state key, st.session_state["view"], controls which is shown)
  Home / Search             query box and number of patents -> 'Analyse patents'
  Research Brief            LLM brief grounded in the retrieved evidence, top patents, next-step buttons
  Relevant patents to you   compact ranked list, CSV export, technical details
  Patent Details            abstract, scores, supporting chunks, graph link
  Advanced Settings (Tools) embedding model, candidate chunks, reranking (default: Nomic + reranking)
  Model Comparison          several models side by side + Reciprocal Rank Fusion, timings,
                            optional relevance metrics for one query, CSV export
  Graph Explorer            saved 2016 patent similarity graph (click a node)
  Model Evaluation          ONE button runs every model (+ RRF) and shows a summary, metrics, latency and
                            downloads. Two modes: automatic CPC-subclass proxy benchmark (no files),
                            or your own labelled CSV.
  Diagnostics               index health, encoder/index dimension check, saved report browser

Research brief LLM (default: local Ollama, model llama3.2). Environment variables:
  PATENT_APP_LLM_PROVIDER = ollama | openai (any OpenAI-compatible endpoint)
  PATENT_APP_LLM_MODEL, PATENT_APP_OLLAMA_URL, PATENT_APP_LLM_BASE_URL, PATENT_APP_LLM_API_KEY

Place this file in the project root and run:
    ./.Patent_Intelligence_virtual_env_final/bin/python -m streamlit run streamlit_app.py

Read-only: never builds embeddings, never writes to ChromaDB.
Requires streamlit >= 1.35 (native plotly click selection).

Reliability changes vs. the first draft (no features removed):
  - project-root validation, empty-collection / empty-query guards
  - distance-metric check before treating 1 - distance as cosine similarity
  - validated ChromaDB result arrays and cross-encoder output
  - missing scores are never silently turned into 0
  - RRF input validation, failed-model reporting
  - stale-result notices on Search and Comparison, CSV export of results
  - richer index health (missing / empty / error), edge-weight diagnostics
  - evaluation page split into System health vs Retrieval evaluation

Second round of fixes:
  - an 'unspecified' distance metric is no longer assumed to be cosine: it is verified against
    the stored vector of the top hit, and the search stops if it cannot be verified
  - comparison stale-check covers query, models, depth, candidates and RRF k
  - empty ChromaDB responses give a friendly message
  - graph edge weights are called 'edge weights' (not cosine similarity) until verified
  - dimension check validates that a stored embedding was returned
  - show_df / show_plotly pick the width API by Streamlit version instead of catching every error

Third round (evaluation):
  - 'Evaluation & Diagnostics' split into 'Diagnostics' (health checks, saved reports) and
    'Model Evaluation' (labelled-query benchmark with P@K, R@K, MRR, nDCG@K, latency, throughput)
  - Comparison page records embedding / vector-search timings and can compute the same relevance
    metrics for one query from a pasted list of known relevant patents

Fourth round (one-click evaluation):
  - Model Evaluation has an automatic mode: samples indexed patents as queries, treats patents in the
    same CPC subclass as relevant, removes the query patent from every ranking. One button runs all
    models and shows a summary, charts and downloads. The labelled-CSV flow remains as a second mode.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "data").is_dir() else _HERE.parent

RUN_ID = os.environ.get("PATENT_APP_RUN_ID", "20261022_190054")
DB_ROOT = ROOT / "data" / "patent_subset_runs" / RUN_ID / "chroma"
LEGACY_DB = ROOT / "data" / "chroma_db"
GRAPH_DIR = ROOT / "data" / "patent_graph_2016"
GRAPH_PATH = GRAPH_DIR / "patent_similarity_graph.graphml"
SAMPLE_PATH = GRAPH_DIR / "samples" / "patents_2016_sample_2500.parquet"
RESULTS_DIR = ROOT / "results"
DEVICE = os.environ.get("PATENT_APP_DEVICE", "cpu")
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# One authoritative record per model. Query prefixes must match the
# document-side prefixes used at indexing time (see dynamic_chunker.py).
MODELS = {
    "minilm": dict(label="MiniLM (all-MiniLM-L6-v2)", model_id="sentence-transformers/all-MiniLM-L6-v2",
                   collection="patent_chunks_minilm", max_length=256, query_prefix=""),
    "mpnet": dict(label="MPNet (all-mpnet-base-v2)", model_id="sentence-transformers/all-mpnet-base-v2",
                  collection="patent_chunks_mpnet", max_length=384, query_prefix=""),
    "bge": dict(label="BGE (bge-base-en-v1.5)", model_id="BAAI/bge-base-en-v1.5",
                collection="patent_chunks_bge", max_length=512, query_prefix=""),
    "e5": dict(label="E5 (e5-base-v2)", model_id="intfloat/e5-base-v2",
               collection="patent_chunks_e5", max_length=512, query_prefix="query: "),
    "patentsberta": dict(label="PatentSBERTa", model_id="AI-Growth-Lab/PatentSBERTa",
                         collection="patent_chunks_patentsberta", max_length=512, query_prefix=""),
    "paecter": dict(label="PaECTER", model_id="mpi-inno-comp/paecter",
                    collection="patent_chunks_paecter", max_length=512, query_prefix=""),
    "nomic": dict(label="Nomic (embed-text-v1.5)", model_id="nomic-ai/nomic-embed-text-v1.5",
                  collection="patent_chunks_nomic", max_length=8192, query_prefix="search_query: ",
                  trust_remote_code=True),
    # Original single-collection index, kept separate from the seven-model run.
    "legacy": dict(label="Legacy MiniLM index (data/chroma_db)",
                   model_id="sentence-transformers/all-MiniLM-L6-v2",
                   collection="patent_rag_chunks", max_length=256, query_prefix="",
                   db_path=LEGACY_DB),
}

st.set_page_config(page_title="Patent Intelligence", page_icon="🔎", layout="wide")

if not (ROOT / "data").is_dir():
    st.error(f"Could not find the project 'data' directory (looked in {_HERE} and its parent). "
             "Place this file in the Patent_Intelligence_Project root.")
    st.stop()


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _streamlit_version() -> tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0)
    except Exception:
        return (0, 0)


# width="stretch" replaced use_container_width in Streamlit 1.50.
WIDTH_KW = {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}


def show_df(data, **kwargs):
    st.dataframe(data, hide_index=True, **WIDTH_KW, **kwargs)


def show_plotly(fig, **kwargs):
    return st.plotly_chart(fig, **WIDTH_KW, **kwargs)


def fmt_s(value):
    return "n/a" if value is None else f"{value:.2f}s"


def db_for(key: str) -> tuple[Path, str]:
    cfg = MODELS[key]
    return Path(cfg.get("db_path", DB_ROOT / key)), cfg["collection"]


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def open_collection(db_path: str, name: str):
    import chromadb

    if not Path(db_path).is_dir():
        raise FileNotFoundError(f"ChromaDB directory not found: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    names = {c.name if hasattr(c, "name") else str(c) for c in client.list_collections()}
    if name not in names:
        raise ValueError(f"Collection '{name}' not in {db_path}. Found: {', '.join(sorted(names)) or '<none>'}")
    return client.get_collection(name)


def get_collection(key: str):
    path, name = db_for(key)
    return open_collection(str(path), name)


def get_distance_metric(col) -> str:
    """Best-effort read of the collection's distance metric ('cosine', 'l2', 'ip' or 'unspecified')."""
    meta = getattr(col, "metadata", None) or {}
    metric = meta.get("hnsw:space")
    if not metric:
        try:  # newer ChromaDB versions expose the space via collection configuration
            cfg = getattr(col, "configuration", None) or {}
            metric = (cfg.get("hnsw") or {}).get("space") or (cfg.get("spann") or {}).get("space")
        except Exception:
            metric = None
    return str(metric).lower() if metric else "unspecified"


_METRIC_CHECKS: dict[str, str] = {}
_DIM_CHECKS: dict[str, int] = {}


def collection_identity(col, total: int) -> str:
    """Changes if the collection is replaced or its size changes, so cached checks are not reused."""
    return f"{getattr(col, 'id', col.name)}::{total}"


def verify_metric_from_hits(col, query_vec, ids, dists, max_hits: int = 5) -> str:
    """Fallback used only when the collection does not report its distance metric.

    Recomputes cosine distance, squared L2 and inner-product distance (1 - dot) from the stored
    vectors of the top hits and compares them with the distances ChromaDB returned. Cosine is
    accepted only if EVERY checked hit matches cosine distance and at least one hit separates
    cosine from squared L2. Anything else stops the search.
    """
    n = min(max_hits, len(ids))
    got = col.get(ids=list(ids[:n]), include=["embeddings"])
    emb = got.get("embeddings")
    if emb is None or len(emb) == 0:
        raise ValueError("Cannot verify the distance metric: no stored embeddings were returned.")
    by_id = dict(zip(got.get("ids") or [], emb))
    q = np.asarray(query_vec, dtype=np.float64)
    tol = 1e-3
    all_cos, any_l2_separates, stored_norms = True, False, []
    for cid, dist in zip(ids[:n], dists[:n]):
        if cid not in by_id:
            raise ValueError(f"Cannot verify the distance metric: no stored vector for chunk {cid!r}.")
        v = np.asarray(by_id[cid], dtype=np.float64)
        stored_norms.append(float(np.linalg.norm(v)))
        cos_d = 1.0 - float(np.dot(v, q) / (np.linalg.norm(v) * np.linalg.norm(q)))
        l2_sq = float(np.sum((v - q) ** 2))
        if abs(float(dist) - cos_d) >= tol:
            all_cos = False
        if abs(l2_sq - cos_d) > 10 * tol:
            any_l2_separates = True
    if not all_cos:
        raise ValueError("Returned distances do not match cosine distance recomputed from the stored vectors "
                         "(the collection may use L2 or inner product). Not treating 1 - distance as cosine "
                         "similarity. Set the metric explicitly (hnsw:space) or rebuild the index metadata.")
    if not any_l2_separates:
        raise ValueError("Cannot tell cosine from squared L2 with these hits. Try a different query.")
    note = "" if all(abs(x - 1.0) < 1e-2 for x in stored_norms) else " (stored vectors not unit-norm)"
    return "cosine" + note


@st.cache_resource(show_spinner="Loading query encoder...", max_entries=3)
def load_encoder(model_id: str, max_length: int, trust_remote_code: bool):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=DEVICE, trust_remote_code=trust_remote_code)
    model.max_seq_length = max_length
    return model


def get_encoder(key: str):
    cfg = MODELS[key]
    return load_encoder(cfg["model_id"], cfg["max_length"], bool(cfg.get("trust_remote_code", False)))


@st.cache_resource(show_spinner="Loading cross-encoder reranker...")
def load_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_ID, device=DEVICE)


@st.cache_data(show_spinner="Reading patent titles from the index...")
def title_map() -> dict[str, str]:
    """The indexes store titles as chunks with section == 'title', not as metadata."""
    for key in ("bge", "patentsberta", "mpnet", "minilm"):
        try:
            col = get_collection(key)
            mapping: dict[str, str] = {}
            for section_value in ("title", "Title", "TITLE"):  # tolerate capitalisation differences
                got = col.get(where={"section": section_value}, include=["documents", "metadatas"])
                for doc, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
                    pn = str((meta or {}).get("patent_number", "")).strip()
                    if pn and doc and pn not in mapping:
                        mapping[pn] = doc.strip()
                if mapping:
                    break
            if mapping:
                return mapping
        except Exception:
            continue
    return {}


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------
def _score(chunks: list[dict], key: str) -> dict:
    """Patent score = 0.5*best chunk + 0.5*mean(top-3 chunks). Missing scores give None, never 0."""
    values = [float(c[key]) for c in chunks if c.get(key) is not None and np.isfinite(float(c[key]))]
    if not values:
        return dict(max=None, top3=None, score=None)
    values.sort(reverse=True)
    top3 = mean(values[:3])
    return dict(max=values[0], top3=top3, score=0.5 * values[0] + 0.5 * top3)


def search_model(key: str, query: str, top_k: int, candidates: int, rerank: bool) -> dict:
    """Chunk retrieval -> patent grouping (0.5*max + 0.5*top3-mean), optional rerank."""
    if key not in MODELS:
        raise ValueError(f"Unknown model: {key}")
    query = (query or "").strip()
    if not query:
        raise ValueError("Enter a non-empty patent search query.")
    top_k, candidates = int(top_k), int(candidates)
    if top_k < 1 or candidates < 1:
        raise ValueError("top_k and candidates must both be at least 1.")

    t_start = time.perf_counter()
    cfg = MODELS[key]
    col = get_collection(key)
    total = int(col.count())
    if total == 0:
        raise ValueError(f"The collection {cfg['collection']!r} is empty.")
    metric = get_distance_metric(col)
    if metric not in ("cosine", "unspecified"):
        raise ValueError(f"Collection {col.name!r} uses distance metric {metric!r}, not cosine; "
                         "1 - distance would not be a cosine similarity.")
    encoder = get_encoder(key)
    titles = title_map()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    vec = encoder.encode(cfg["query_prefix"] + query, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    timings["embedding"] = time.perf_counter() - t0

    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim != 1 or vec.size == 0:
        raise RuntimeError(f"Query encoder returned an unexpected shape: {vec.shape}")
    if not np.isfinite(vec).all():
        raise RuntimeError("Query encoder returned non-finite values.")
    dim_key = f"{db_for(key)[0]}::{col.name}::{collection_identity(col, total)}"
    if dim_key not in _DIM_CHECKS:
        sample = col.get(limit=1, include=["embeddings"]).get("embeddings")
        if sample is None or len(sample) == 0:
            raise RuntimeError("Could not read a stored embedding to check the vector dimension.")
        _DIM_CHECKS[dim_key] = len(sample[0])
    if vec.size != _DIM_CHECKS[dim_key]:
        raise RuntimeError(f"Dimension mismatch: encoder output is {vec.size}-d but the index stores "
                           f"{_DIM_CHECKS[dim_key]}-d vectors; {cfg['model_id']} cannot query this collection.")

    t1 = time.perf_counter()
    n = min(max(1, int(candidates)), total)
    raw = col.query(query_embeddings=[vec.tolist()], n_results=n,
                    include=["documents", "metadatas", "distances"])
    timings["vector_search"] = time.perf_counter() - t1

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    if not ids:
        timings["total"] = time.perf_counter() - t_start
        return dict(
            model=key, label=cfg["label"], query=query, collection=col.name,
            collection_path=str(db_for(key)[0]), collection_count=total, candidates=0,
            reranked=False, score_kind="cosine similarity", distance_metric=metric,
            model_id=cfg["model_id"], top_k=top_k, candidates_requested=candidates,
            patents=[], results=[], candidate_chunks=[], timings=timings)
    if not (len(ids) == len(docs) == len(metas) == len(dists)):
        raise RuntimeError("ChromaDB returned inconsistent result-array lengths.")

    if metric == "unspecified":
        cache_key = f"{db_for(key)[0]}::{col.name}::{collection_identity(col, total)}"
        if cache_key not in _METRIC_CHECKS:
            _METRIC_CHECKS[cache_key] = verify_metric_from_hits(col, vec, ids, dists)
        metric = f"{_METRIC_CHECKS[cache_key]} (verified from stored vectors)"

    chunks = []
    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        meta = meta or {}
        pn = str(meta.get("patent_number", "Unknown"))
        chunks.append(dict(
            chunk_id=cid, document=doc or "", patent_number=pn,
            title=str(meta.get("title") or titles.get(pn) or "Title unavailable"),
            section=meta.get("section", ""), chunk_index=meta.get("chunk_index", ""),
            cpc_label=meta.get("cpc_label", ""), ipc_label=meta.get("ipc_label", ""),
            decision=meta.get("decision", ""), date_published=meta.get("date_published", ""),
            embedding_rank=rank, distance=float(dist), embedding_similarity=1.0 - float(dist),
        ))

    score_key = "embedding_similarity"
    ordered = chunks
    if rerank and chunks:
        t2 = time.perf_counter()
        scores = np.asarray(load_reranker().predict([(query, c["document"]) for c in chunks],
                                                    show_progress_bar=False), dtype=np.float32)
        if scores.ndim != 1 or len(scores) != len(chunks):
            raise RuntimeError(f"Unexpected cross-encoder output shape: {scores.shape}")
        if not np.isfinite(scores).all():
            raise RuntimeError("Cross-encoder produced non-finite scores.")
        for c, s in zip(chunks, scores):
            c["reranker_score"] = float(s)
        ordered = sorted(chunks, key=lambda c: c["reranker_score"], reverse=True)
        for r, c in enumerate(ordered, start=1):
            c["reranker_rank"] = r
        timings["reranking"] = time.perf_counter() - t2
        score_key = "reranker_score"

    groups: dict[str, list[dict]] = defaultdict(list)
    for c in ordered:
        groups[c["patent_number"]].append(c)

    patents = []
    for pn, cs in groups.items():
        cs = sorted(cs, key=lambda c: c[score_key], reverse=True)
        sim = _score(cs, "embedding_similarity")
        main = _score(cs, score_key)
        if main["score"] is None or sim["score"] is None:
            continue  # unscored patents are excluded, never ranked as zero relevance
        patents.append(dict(
            patent_number=pn, title=cs[0]["title"], cpc_label=cs[0]["cpc_label"],
            ipc_label=cs[0]["ipc_label"], decision=cs[0]["decision"],
            date_published=cs[0]["date_published"], score=main["score"],
            max_similarity=sim["max"], top3_similarity=sim["top3"],
            chunk_count=len(cs), chunks=cs[:3],
        ))
    patents.sort(key=lambda p: p["score"], reverse=True)
    for r, p in enumerate(patents, start=1):
        p["rank"] = r

    timings["total"] = time.perf_counter() - t_start
    return dict(
        model=key, label=cfg["label"], query=query, collection=col.name,
        collection_path=str(db_for(key)[0]), collection_count=total, candidates=len(chunks),
        reranked=rerank, score_kind="reranker logit" if rerank else "cosine similarity",
        distance_metric=metric, model_id=cfg["model_id"], top_k=top_k, candidates_requested=candidates,
        patents=patents, results=patents[: int(top_k)], candidate_chunks=ordered, timings=timings,
    )


def rrf_fuse(rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over patent ranks (never over raw scores)."""
    if k < 1:
        raise ValueError("RRF constant k must be at least 1.")
    fused: dict[str, float] = defaultdict(float)
    for ranked in rankings.values():
        seen: set[str] = set()
        for rank, pn in enumerate(ranked, start=1):
            pn = str(pn).strip()
            if not pn or pn == "Unknown" or pn in seen:
                continue
            seen.add(pn)
            fused[pn] += 1.0 / (k + rank)
    return dict(fused)


# ----------------------------------------------------------------------
# Relevance metrics and model-to-model benchmark
# ----------------------------------------------------------------------
EVAL_KS = (1, 3, 5, 10, 20)


def parse_relevance_text(text: str) -> dict[str, int]:
    """One patent per line: 'number' (grade 1) or 'number,grade' (grade 0 = judged not relevant)."""
    rel: dict[str, int] = {}
    for n, line in enumerate((text or "").splitlines(), start=1):
        parts = [p for p in re.split(r"[,;\s]+", line.strip()) if p]
        if not parts:
            continue
        if len(parts) > 2:
            raise ValueError(f"Line {n}: expected 'patent_number' or 'patent_number,grade'.")
        try:
            grade = int(parts[1]) if len(parts) == 2 else 1
        except ValueError:
            raise ValueError(f"Line {n}: grade {parts[1]!r} is not an integer.") from None
        if grade < 0:
            raise ValueError(f"Line {n}: grade must be >= 0.")
        if parts[0] in rel and rel[parts[0]] != grade:
            raise ValueError(f"Line {n}: patent {parts[0]} is given two different grades.")
        rel[parts[0]] = grade
    return rel


def ranking_metrics(ranked: list[str], rel: dict[str, int], ks) -> dict:
    """P@K, R@K, nDCG@K and MRR for one ranked patent list.

    rel maps patent_number -> grade (>0 relevant). Patents missing from rel count as not relevant.
    P@K always divides by K, so a ranking shorter than K is penalised rather than flattered.
    R@K is relative to the labelled relevant set, not to every relevant patent in the corpus.
    nDCG uses gain 2^grade - 1 (binary labels reduce to standard binary nDCG).
    MRR is taken over the supplied list (the retrieval depth).
    """
    ranked = list(dict.fromkeys(str(p) for p in ranked))
    relevant = {p for p, g in rel.items() if g > 0}
    names = ["MRR"] + [f"{m}@{k}" for k in ks for m in ("P", "R", "nDCG")]
    if not relevant:
        return {n: None for n in names}
    first = next((i for i, p in enumerate(ranked, 1) if p in relevant), None)
    out = {"MRR": 1.0 / first if first else 0.0}
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)
    for k in ks:
        top = ranked[:k]
        hits = sum(p in relevant for p in top)
        dcg = sum((2 ** rel.get(p, 0) - 1) / math.log2(i + 1) for i, p in enumerate(top, 1))
        idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal[:k], 1))
        out[f"P@{k}"] = hits / k
        out[f"R@{k}"] = hits / len(relevant)
        out[f"nDCG@{k}"] = dcg / idcg if idcg else 0.0
    return out


def parse_label_table(df: pd.DataFrame) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Labels CSV -> (queries, labels). Columns: query, patent_number, [relevance], [query_id]."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"query", "patent_number"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}. "
                         "Expected: query, patent_number, relevance (optional), query_id (optional).")
    df = df.dropna(subset=["query", "patent_number"])
    df["query"] = df["query"].astype(str).str.strip()
    df["patent_number"] = df["patent_number"].astype(str).str.strip()
    df = df[(df["query"] != "") & (df["patent_number"] != "")]
    if df.empty:
        raise ValueError("The label file has no usable rows.")
    if "relevance" not in df.columns:
        df["relevance"] = 1
    df["relevance"] = pd.to_numeric(df["relevance"], errors="raise")
    if (df["relevance"] < 0).any() or (df["relevance"] % 1 != 0).any():
        raise ValueError("relevance must be a non-negative integer (0 = not relevant, 1+ = relevant / graded).")
    df["relevance"] = df["relevance"].astype(int)

    if "query_id" in df.columns:
        df["query_id"] = df["query_id"].astype(str).str.strip()
    else:
        order = {q: f"q{i}" for i, q in enumerate(dict.fromkeys(df["query"]), start=1)}
        df["query_id"] = df["query"].map(order)
    if (df.groupby("query_id")["query"].nunique() > 1).any():
        raise ValueError("A query_id is used with more than one query text.")
    if (df.groupby(["query_id", "patent_number"])["relevance"].nunique() > 1).any():
        raise ValueError("The same patent has conflicting relevance grades for one query.")

    queries = [dict(query_id=qid, query=g["query"].iloc[0]) for qid, g in df.groupby("query_id", sort=False)]
    labels = {qid: dict(zip(g["patent_number"], g["relevance"])) for qid, g in df.groupby("query_id", sort=False)}
    return queries, labels


def run_model_evaluation(model_keys, queries, labels, depth, candidates, rerank, rrf_k, ks, on_progress=None) -> dict:
    """Run every model on every labelled query; return per-query metrics, timings and rankings.

    Each model gets one untimed warm-up query first, so timings exclude one-time encoder/index warm-up.
    A model that fails on any query is dropped entirely, so all systems are averaged over the same queries.
    RRF is added (over the surviving models) when at least two models survive.
    A query may carry an 'exclude' patent number (automatic benchmark): that patent is removed from every ranking.
    """
    ks = [k for k in ks if k <= depth]
    kmax = max(ks)
    rows, rank_rows, errors, per_model = [], [], {}, {}

    for i, key in enumerate(model_keys):
        m_rows, m_rank, m_rankings = [], [], {}
        try:
            search_model(key, queries[0]["query"], depth, candidates, rerank)  # warm-up, not recorded
            for qi, q in enumerate(queries):
                resp = search_model(key, q["query"], depth, candidates, rerank)
                excl = q.get("exclude")  # the query patent itself (automatic benchmark)
                kept = [p for p in resp["patents"] if p["patent_number"] != excl][:depth]
                ranked = [p["patent_number"] for p in kept]
                rel, t = labels[q["query_id"]], resp["timings"]
                m_rankings[q["query_id"]] = ranked
                m_rows.append(dict(
                    system=key, query_id=q["query_id"], patents_returned=len(kept),
                    candidate_chunks=resp["candidates"], embedding_s=t.get("embedding"),
                    vector_search_s=t.get("vector_search"), reranking_s=t.get("reranking"), total_s=t.get("total"),
                    unjudged_frac=sum(p not in rel for p in ranked[:kmax]) / kmax,
                    **ranking_metrics(ranked, rel, ks)))
                for r, p in enumerate(kept, start=1):
                    m_rank.append(dict(system=key, query_id=q["query_id"], rank=r, patent_number=p["patent_number"],
                                       score=p["score"], relevance=rel.get(p["patent_number"], "")))
                if on_progress:
                    on_progress((i + (qi + 1) / len(queries)) / len(model_keys))
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            continue
        rows += m_rows
        rank_rows += m_rank
        per_model[key] = m_rankings
        gc.collect()

    if len(per_model) >= 2:
        lookup = {(r["system"], r["query_id"]): r for r in rows}
        ok = list(per_model)
        for q in queries:
            qid, rel = q["query_id"], labels[q["query_id"]]
            fused = rrf_fuse({m: per_model[m][qid] for m in ok}, rrf_k)
            ranked = [pn for pn, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))][:depth]
            comp = [lookup[(m, qid)] for m in ok]
            rerank_sum = [c["reranking_s"] for c in comp if c["reranking_s"] is not None]
            rows.append(dict(
                system="rrf", query_id=qid, patents_returned=len(fused),
                candidate_chunks=sum(c["candidate_chunks"] for c in comp),
                embedding_s=sum(c["embedding_s"] or 0 for c in comp),
                vector_search_s=sum(c["vector_search_s"] or 0 for c in comp),
                reranking_s=sum(rerank_sum) if rerank_sum else None,
                total_s=sum(c["total_s"] or 0 for c in comp),   # sum of component latencies (sequential)
                unjudged_frac=sum(p not in rel for p in ranked[:kmax]) / kmax,
                **ranking_metrics(ranked, rel, ks)))
            for r, pn in enumerate(ranked, start=1):
                rank_rows.append(dict(system="rrf", query_id=qid, rank=r, patent_number=pn,
                                      score=fused[pn], relevance=rel.get(pn, "")))
    return dict(rows=rows, rankings=rank_rows, errors=errors, ks=ks,
                systems=(list(per_model) + (["rrf"] if len(per_model) >= 2 else [])))


# ----------------------------------------------------------------------
# Automatic benchmark (no label file needed)
# ----------------------------------------------------------------------
def cpc_subclass(label) -> str:
    """First CPC subclass found in a label, e.g. 'H04L' (empty string if none)."""
    m = re.search(r"[A-HY]\d{2}[A-Z]", str(label or "").upper())
    return m.group(0) if m else ""


@st.cache_data(show_spinner="Reading patents from the index...")
def evaluation_universe() -> tuple[pd.DataFrame, str]:
    """All indexed patents with title and CPC label, read from the title chunks of one collection."""
    for key in ("bge", "patentsberta", "mpnet", "minilm"):
        try:
            col = get_collection(key)
            rows = []
            for section_value in ("title", "Title", "TITLE"):
                got = col.get(where={"section": section_value}, include=["documents", "metadatas"])
                for doc, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
                    meta = meta or {}
                    pn = str(meta.get("patent_number", "")).strip()
                    if pn:
                        rows.append(dict(patent_number=pn, title=(doc or "").strip(),
                                         cpc_label=str(meta.get("cpc_label", "") or "")))
                if rows:
                    break
            if rows:
                df = pd.DataFrame(rows).drop_duplicates("patent_number").reset_index(drop=True)
                df["subclass"] = df["cpc_label"].map(cpc_subclass)
                return df, key
        except Exception:
            continue
    raise RuntimeError("Could not read patent metadata from any index.")


def build_auto_benchmark(n_queries: int, seed: int, query_source: str):
    """Self-supervised benchmark: query = a patent's abstract/title; relevant = other patents in its CPC subclass.

    Only subclasses with 4-60 indexed patents are used (so recall is meaningful and not diluted).
    Every indexed patent gets a grade (1 same subclass, 0 otherwise), so nothing is 'unjudged'.
    """
    uni, src_key = evaluation_universe()
    sizes = uni[uni["subclass"] != ""].groupby("subclass")["patent_number"].transform("size")
    eligible = uni[uni["subclass"] != ""][(sizes >= 4) & (sizes <= 60)]
    if eligible.empty:
        raise ValueError("No CPC subclass with 4-60 patents was found, so no automatic benchmark can be built. "
                         "Use 'Upload my own labels' instead.")
    pool = eligible.sample(n=min(n_queries * 3, len(eligible)), random_state=int(seed))  # oversample, some lack text
    members = uni.groupby("subclass")["patent_number"].apply(set).to_dict()
    all_pns = uni["patent_number"].tolist()

    queries, labels = [], {}
    for _, row in pool.iterrows():
        if len(queries) >= n_queries:
            break
        text = ""
        if query_source == "Abstract":
            try:
                text = patent_abstract(src_key, row["patent_number"])
            except Exception:
                text = ""
        text = " ".join((text or row["title"]).split())[:400]
        if not text:
            continue
        qid = f"auto_{row['patent_number']}"
        rel_set = members[row["subclass"]] - {row["patent_number"]}
        labels[qid] = {pn: (1 if pn in rel_set else 0) for pn in all_pns}
        queries.append(dict(query_id=qid, query=text, exclude=row["patent_number"]))
    if not queries:
        raise ValueError("Could not build any query text from the index.")
    info = dict(universe=len(uni), eligible=len(eligible), queries=len(queries),
                avg_relevant=float(np.mean([sum(g > 0 for g in l.values()) for l in labels.values()])))
    return queries, labels, info


def system_label(system: str) -> str:
    return "RRF fusion" if system == "rrf" else MODELS[system]["label"].split(" (")[0]


def latency_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for system, g in df.groupby("system", sort=False):
        row = {"system": system_label(system), "queries": len(g)}
        for col in ("embedding_s", "vector_search_s", "reranking_s", "total_s"):
            v = g[col].dropna().astype(float)
            if len(v):
                name = col[:-2]
                row[f"{name}_mean_s"] = v.mean()
                row[f"{name}_p50_s"] = float(np.percentile(v, 50))
                row[f"{name}_p95_s"] = float(np.percentile(v, 95))
        row["throughput_qps"] = len(g) / g["total_s"].sum() if g["total_s"].sum() > 0 else None
        out.append(row)
    return pd.DataFrame(out).round(4)


# ----------------------------------------------------------------------
# Abstracts and LLM research brief
# ----------------------------------------------------------------------
def _chunk_order(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0


@st.cache_data(show_spinner=False, max_entries=512)
def patent_abstract(model_key: str, patent_number: str) -> str:
    """Abstract text stored in the index as chunks with section == 'abstract'. Empty string if none found."""
    col = get_collection(model_key)
    candidates = [[patent_number]]
    if patent_number.isdigit():
        candidates.append([int(patent_number)])  # tolerate patent_number stored as an integer
    for values in candidates:
        try:
            got = col.get(where={"$and": [{"patent_number": {"$in": values}},
                                          {"section": {"$in": ["abstract", "Abstract", "ABSTRACT"]}}]},
                          include=["documents", "metadatas"])
        except Exception:
            continue
        docs = got.get("documents") or []
        metas = got.get("metadatas") or [{}] * len(docs)
        pairs = sorted(zip(metas, docs), key=lambda t: _chunk_order((t[0] or {}).get("chunk_index")))
        text = "\n\n".join(d.strip() for _, d in pairs if d and d.strip())
        if text:
            return text
    return ""


def excerpt(text: str, n: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


# LLM provider settings. Ollama is the default; any OpenAI-compatible endpoint can be used instead:
#   PATENT_APP_LLM_PROVIDER=openai PATENT_APP_LLM_MODEL=<model> PATENT_APP_LLM_API_KEY=<key>
LLM_PROVIDER = os.environ.get("PATENT_APP_LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.environ.get("PATENT_APP_LLM_MODEL", "llama3.2:3b")
OLLAMA_URL = os.environ.get("PATENT_APP_OLLAMA_URL", "http://localhost:11434")
LLM_BASE_URL = os.environ.get("PATENT_APP_LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("PATENT_APP_LLM_API_KEY", "")
LLM_TIMEOUT = int(os.environ.get("PATENT_APP_LLM_TIMEOUT", "300"))

# Evidence budget for the brief (a small local model has a small context window).
BRIEF_PATENTS = 6
BRIEF_CHUNKS = 3
BRIEF_CHUNK_CHARS = 500
BRIEF_ABSTRACT_CHARS = 700

BRIEF_SYSTEM = """You are a careful patent research assistant. You receive a research question and evidence \
retrieved from a patent collection by embedding similarity.

Rules:
- Make patent-specific statements only from the supplied evidence. Do not use outside knowledge about these patents.
- Cite every substantive finding with the patent number in square brackets, e.g. [12345678]. Only cite patent \
numbers that appear in the evidence.
- Separate what the passages explicitly disclose from your own interpretation. Start interpretation with "Interpretation:".
- If the evidence is thin, off-topic or inconsistent, say so plainly.
- Never invent titles, patent numbers, claims or technical details.
- Retrieval scores show textual similarity only. Never say a patent is novel, invalid, infringed or legally relevant \
on that basis.
- Be concise."""


def llm_label() -> str:
    return f"{LLM_PROVIDER}:{LLM_MODEL}"


def _post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LLM server returned HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the LLM server at {url} ({exc.reason}). For Ollama, make sure it is "
                           f"running and the model is installed (ollama pull {LLM_MODEL}).") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"The LLM did not answer within {LLM_TIMEOUT}s.") from exc


def _chat_ollama(system: str, user: str) -> str:
    data = _post_json(f"{OLLAMA_URL.rstrip('/')}/api/chat", dict(
        model=LLM_MODEL, stream=False, options=dict(temperature=0.2, num_ctx=8192),
        messages=[dict(role="system", content=system), dict(role="user", content=user)]))
    return ((data.get("message") or {}).get("content") or "").strip()


def _chat_openai(system: str, user: str) -> str:
    if not LLM_API_KEY:
        raise RuntimeError("PATENT_APP_LLM_API_KEY is not set.")
    data = _post_json(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", dict(
        model=LLM_MODEL, temperature=0.2,
        messages=[dict(role="system", content=system), dict(role="user", content=user)]),
        headers={"Authorization": f"Bearer {LLM_API_KEY}"})
    choices = data.get("choices") or [{}]
    return ((choices[0].get("message") or {}).get("content") or "").strip()


LLM_PROVIDERS = {"ollama": _chat_ollama, "openai": _chat_openai}


def prepare_evidence(resp: dict) -> list[dict]:
    """Compact evidence bundle: top unique patents, abstract and best chunks only (never all candidate chunks)."""
    evidence = []
    for p in resp["results"][:BRIEF_PATENTS]:
        try:
            abstract = patent_abstract(resp["model"], p["patent_number"])[:BRIEF_ABSTRACT_CHARS]
        except Exception:
            abstract = ""  # the brief can still be written from the retrieved passages
        evidence.append(dict(
            patent_number=p["patent_number"], title=p["title"], rank=p["rank"],
            score=p["score"], score_kind=resp["score_kind"],
            abstract=abstract,
            passages=[dict(chunk_id=c["chunk_id"], section=c["section"] or "?", text=c["document"][:BRIEF_CHUNK_CHARS])
                      for c in p["chunks"][:BRIEF_CHUNKS]],
        ))
    return evidence


def generate_research_brief(query: str, evidence: list[dict]) -> str:
    """Evidence-grounded brief from the configured LLM provider. Raises on any failure."""
    if LLM_PROVIDER not in LLM_PROVIDERS:
        raise ValueError(f"Unknown LLM provider {LLM_PROVIDER!r}. Use one of: {', '.join(LLM_PROVIDERS)}.")
    if not evidence:
        raise ValueError("No evidence to summarise.")
    blocks = []
    for e in evidence:
        lines = [f"[{e['patent_number']}] Title: {e['title']}",
                 f"Retrieval rank {e['rank']} ({e['score_kind']} {e['score']:.3f}; similarity only)"]
        if e["abstract"]:
            lines.append(f"Abstract: {e['abstract']}")
        for i, ps in enumerate(e["passages"], start=1):
            lines.append(f"Passage {i} (section: {ps['section']}, chunk {ps['chunk_id']}): {ps['text']}")
        blocks.append("\n".join(lines))
    user = (f"Research question:\n{query}\n\nRetrieved evidence ({len(evidence)} patents):\n\n" + "\n\n".join(blocks) +
            "\n\nWrite the brief in Markdown with exactly these sections, using ### headings:\n"
            "### Executive summary (2-4 sentences)\n"
            "### Key technical themes (bullets, each with citations)\n"
            "### How the patents differ (bullets, each with citations)\n"
            "### Limitations and confidence")
    text = LLM_PROVIDERS[LLM_PROVIDER](BRIEF_SYSTEM, user)
    if not text:
        raise RuntimeError("The LLM returned an empty response.")
    return text


# ----------------------------------------------------------------------
# Session state and navigation (one authoritative key: st.session_state["view"])
# ----------------------------------------------------------------------
STATE_DEFAULTS = dict(
    view="Home", search_resp=None, research_brief=None, brief_error=None, active_query="",
    selected_patent=None, show_all_patents=False,
    settings=dict(model="nomic", top_k=10, candidates=50, rerank=True),
)


def initialise_state():
    for k, v in STATE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = dict(v) if isinstance(v, dict) else v


def navigate_to(view: str):
    st.session_state["view"] = view


def set_state(key: str, value):
    st.session_state[key] = value


def select_patent(patent_number: str):
    st.session_state["selected_patent"] = patent_number
    st.session_state["view"] = "Patent Details"


def goto_graph(patent_number: str):
    st.session_state["graph_selected"] = patent_number
    st.session_state["view"] = "Graph Explorer"


def new_search():
    for k in ("search_resp", "research_brief", "brief_error", "selected_patent"):
        st.session_state[k] = None
    st.session_state["show_all_patents"] = False
    st.session_state["view"] = "Home"


def build_brief(resp: dict):
    """Generate the brief from an existing response. A failure is stored; the retrieval is never discarded."""
    ss = st.session_state
    ss["research_brief"], ss["brief_error"] = None, None
    try:
        with st.spinner("Preparing research brief..."):
            evidence = prepare_evidence(resp)
            text = generate_research_brief(resp["query"], evidence)
        ss["research_brief"] = dict(text=text, evidence_patents=[e["patent_number"] for e in evidence], provider=llm_label())
    except Exception as exc:
        ss["brief_error"] = f"{type(exc).__name__}: {exc}"


def run_analysis(key: str, query: str, top_k: int, candidates: int, rerank: bool):
    ss = st.session_state
    for k in ("search_resp", "research_brief", "brief_error", "selected_patent"):  # never show a stale answer
        ss[k] = None
    ss["show_all_patents"] = False
    ss["settings"] = dict(model=key, top_k=int(top_k), candidates=int(candidates), rerank=bool(rerank))
    try:
        with st.spinner("Searching the patent collection..."):
            resp = search_model(key, query, top_k, candidates, rerank)
    except Exception as exc:
        st.error(f"Patent retrieval failed: {type(exc).__name__}: {exc}")
        return
    ss["search_resp"] = resp
    ss["active_query"] = query.strip()
    if resp["patents"]:
        build_brief(resp)
    ss["view"] = "Research Brief"
    st.rerun()


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------
def model_options() -> list[str]:
    return list(MODELS.keys())


def score_label(resp: dict) -> str:
    return "Reranker logit" if resp["reranked"] else "Embedding similarity (cosine)"


def render_patent_card(p: dict, resp: dict, prefix: str):
    with st.container(border=True):
        st.markdown(f"**{p['rank']}. {p['title']}**")
        st.caption(f"Patent {p['patent_number']} | {score_label(resp)} (patent score): {p['score']:.3f} "
                   f"| {p['chunk_count']} evidence chunks")
        if p["chunks"]:
            best = p["chunks"][0]
            st.write(f"*{best['section'] or 'passage'}:* {excerpt(best['document'])}")
        c = st.columns([1, 1, 4])
        c[0].button("View details", key=f"{prefix}_d_{p['patent_number']}", on_click=select_patent,
                    args=(p["patent_number"],), **WIDTH_KW)
        c[1].button("Explore in graph", key=f"{prefix}_g_{p['patent_number']}", on_click=goto_graph,
                    args=(p["patent_number"],), **WIDTH_KW)


def render_pipeline_inspection(resp: dict):
    t = resp["timings"]
    cols = st.columns(5)
    cols[0].metric("Collection chunks", f"{resp['collection_count']:,}")
    cols[1].metric("Candidate chunks", resp["candidates"])
    cols[2].metric("Unique patents", len(resp["patents"]))
    cols[3].metric("Embedding", fmt_s(t.get("embedding")))
    cols[4].metric("Vector search", fmt_s(t.get("vector_search")))
    if "reranking" in t:
        st.caption(f"Cross-encoder reranking: {fmt_s(t['reranking'])}. Reranker scores are raw logits, not probabilities.")
    st.caption(f"Patent score = 0.5 x best chunk + 0.5 x mean of top-3 chunks ({resp['score_kind']}). "
               "This is a heuristic ranking signal, not a calibrated probability of relevance. "
               "Scores are only comparable within one model.")
    st.caption(f"Model ID: {resp.get('model_id', '?')} | distance metric: {resp.get('distance_metric', '?')}"
               + f" | total search time {fmt_s(t.get('total'))}"
               + f" | candidate chunks are chunks, not patents ({resp['candidates']} chunks -> {len(resp['patents'])} patents)")
    st.write(f"**Model:** {resp['label']}  \n**Collection:** {resp['collection']}  \n**Path:** {resp['collection_path']}")
    st.json(t)
    show_df([{
        "embedding_rank": c["embedding_rank"], "patent": c["patent_number"], "section": c["section"],
        "similarity": round(c["embedding_similarity"], 4),
        **({"reranker_rank": c["reranker_rank"], "reranker_score": round(c["reranker_score"], 4)}
           if "reranker_score" in c else {}),
    } for c in resp["candidate_chunks"]])


# ----------------------------------------------------------------------
# View: Home / Search
# ----------------------------------------------------------------------
def page_home():
    s = st.session_state["settings"]
    st.title("Patent Intelligence")
    st.caption("Discover technical approaches, related inventions, and supporting patent evidence.")

    query = st.text_area("What invention or technical problem are you investigating?", height=140,
                         value=st.session_state.get("active_query", ""),
                         placeholder="Describe the mechanism, application, or technical problem in your own words...")
    st.caption("Tip: include the input signals, mechanism, algorithm, or desired outcome where possible.")

    top_k = st.slider("How many patents do you want to retrieve?", 3, 20, s["top_k"])

    if st.button("Analyse patents", type="primary", disabled=not query.strip()):
        run_analysis(s["model"], query, top_k, s["candidates"], s["rerank"])
    st.caption("Retrieves evidence from the patent index, ranks patents, then asks the configured LLM "
               f"({llm_label()}) for an evidence-based research brief.")


# ----------------------------------------------------------------------
# View: Advanced Settings (Tools)
# ----------------------------------------------------------------------
def page_settings():
    s = st.session_state["settings"]
    st.title("Advanced Settings")
    st.caption("Retrieval settings used by 'Analyse patents' on the Home page. "
               "Defaults: Nomic embeddings with cross-encoder reranking.")
    opts = model_options()
    s["model"] = st.selectbox("Embedding model", opts, index=opts.index(s["model"]) if s["model"] in opts else 0,
                              format_func=lambda k: MODELS[k]["label"])
    s["candidates"] = st.slider("Candidate chunks", 10, 200, s["candidates"], step=10)
    s["rerank"] = st.toggle("Cross-encoder reranking", value=s["rerank"])
    st.caption("Changes apply to the next search; the current results are not re-run.")


# ----------------------------------------------------------------------
# View: Research Brief
# ----------------------------------------------------------------------
def page_brief():
    ss = st.session_state
    resp, brief, err = ss.get("search_resp"), ss.get("research_brief"), ss.get("brief_error")
    st.title("Research Brief")
    if not resp:
        st.info("Run a patent search first.")
        st.button("Go to search", on_click=navigate_to, args=("Home",))
        return
    st.caption(f"Query: {resp['query']} | Model: {resp['label']}")

    if not resp["patents"]:
        st.info("No patents were retrieved for this query.")
    elif brief:
        st.markdown(brief["text"])
        unknown = sorted(set(re.findall(r"\[(\d{5,})\]", brief["text"])) - set(brief["evidence_patents"]))
        if unknown:
            st.warning("The brief cites patent numbers that were not in the supplied evidence: "
                       + ", ".join(unknown) + ". Treat those citations as unreliable.")
        cited = [pn for pn in brief["evidence_patents"] if pn in brief["text"]]
        if cited:
            st.caption("Open a cited patent:")
            cols = st.columns(min(len(cited), 6))
            for i, pn in enumerate(cited):
                cols[i % len(cols)].button(pn, key=f"cite_{pn}", on_click=select_patent, args=(pn,), **WIDTH_KW)
        st.caption(f"Generated by {brief['provider']} from the top {len(brief['evidence_patents'])} retrieved patents. "
                   "Similarity ranking does not establish novelty, patentability or legal relevance; "
                   "check the claims and source passages.")
    else:
        if err:
            st.error(f"The research brief could not be generated: {err}")
            st.caption("Your retrieved patents are still available below and under 'Relevant patents to you'.")
        else:
            st.info("No brief has been generated for this search.")
        if st.button("Retry brief" if err else "Generate brief", type="primary"):
            build_brief(resp)
            st.rerun()

    if resp["results"]:
        st.subheader("Top retrieved patents")
        for p in resp["results"][:5]:
            render_patent_card(p, resp, "brief")

    st.subheader("Continue your investigation")
    c = st.columns(5)
    c[0].button("View all patents", on_click=navigate_to, args=("Relevant patents to you",), key="b_all", **WIDTH_KW)
    c[1].button("Compare models", on_click=navigate_to, args=("Model Comparison",), key="b_cmp", **WIDTH_KW)
    c[2].button("Open graph", on_click=navigate_to, args=("Graph Explorer",), key="b_graph", **WIDTH_KW)
    c[3].button("Evaluation", on_click=navigate_to, args=("Model Evaluation",), key="b_eval", **WIDTH_KW)
    c[4].button("New search", on_click=new_search, key="b_new", **WIDTH_KW)


# ----------------------------------------------------------------------
# View: Relevant patents to you
# ----------------------------------------------------------------------
def render_patent_list():
    ss = st.session_state
    resp = ss.get("search_resp")
    st.title("Relevant patents to you")
    st.info("These are the patents whose text best matches your description, ranked by similarity. "
            "Open View details for the abstract and the matching passages, or Explore in graph to see related patents. "
            "A high rank means similar wording, not that the invention is novel or legally relevant.")
    if not resp:
        st.info("Run a patent search first.")
        st.button("Go to search", on_click=navigate_to, args=("Home",))
        return
    st.caption(f"Model: {resp['label']} | Query: {resp['query']} | {len(resp['patents'])} unique patents "
               f"from {resp['candidates']} candidate chunks")
    st.caption(f"Patent score = 0.5 x best chunk + 0.5 x mean of top-3 chunks ({resp['score_kind']}); a heuristic "
               "ranking signal, not a probability. Scores are only comparable within one model.")
    if resp["reranked"]:
        st.warning("Reranking is on: the patent score is built from cross-encoder logits, while similarity values in "
                   "the detail view are embedding cosine similarities. They are different quantities.")
    if not resp["results"]:
        st.info("No results returned.")
        return

    export = pd.DataFrame([{
        "rank": p["rank"], "patent_number": p["patent_number"], "title": p["title"], "score": p["score"], "score_kind": resp["score_kind"],
        "max_similarity": p["max_similarity"], "top3_similarity": p["top3_similarity"],
        "evidence_chunks": p["chunk_count"], "cpc": p["cpc_label"], "ipc": p["ipc_label"],
        "decision": p["decision"], "date_published": p["date_published"], "model": resp["model"],
    } for p in resp["results"]])
    st.download_button("Download ranked results (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name=f"patent_results_{resp['model']}.csv", mime="text/csv", key="list_dl")

    show_all = ss.get("show_all_patents", False)
    rows = resp["patents"][:50] if show_all else resp["results"][:10]
    for p in rows:
        render_patent_card(p, resp, "list")
    if show_all:
        st.caption("Showing up to 50 patents.")
        st.button("Show top results only", on_click=set_state, args=("show_all_patents", False), key="list_less")
    elif len(resp["patents"]) > len(rows):
        st.button(f"View all retrieved patents ({len(resp['patents'])})", on_click=set_state,
                  args=("show_all_patents", True), key="list_more")

    with st.expander("Technical details"):
        render_pipeline_inspection(resp)


# ----------------------------------------------------------------------
# View: Patent Details
# ----------------------------------------------------------------------
def render_patent_details():
    ss = st.session_state
    resp, pn = ss.get("search_resp"), ss.get("selected_patent")
    if not resp or not pn:
        st.info("Select a patent from the retrieved results.")
        st.button("Go to relevant patents", on_click=navigate_to, args=("Relevant patents to you",))
        return
    p = next((x for x in resp["patents"] if x["patent_number"] == pn), None)
    if p is None:
        st.info(f"Patent {pn} is not in the current search results.")
        st.button("Back to results", on_click=navigate_to, args=("Relevant patents to you",), key="d_back0")
        return

    st.header(p["title"] or "Title unavailable")
    st.caption(f"Patent number: {pn} | CPC {p['cpc_label'] or 'n/a'} | IPC {p['ipc_label'] or 'n/a'} "
               f"| published {p['date_published'] or 'n/a'}")
    c = st.columns([1, 1, 1, 3])
    c[0].button("Back to results", on_click=navigate_to, args=("Relevant patents to you",), key="d_back", **WIDTH_KW)
    c[1].button("Back to brief", on_click=navigate_to, args=("Research Brief",), key="d_brief", **WIDTH_KW)
    c[2].button("Explore in graph", on_click=goto_graph, args=(pn,), key="d_graph", **WIDTH_KW)
    st.caption("The graph is the saved 2016 patent graph; a retrieved patent outside that dataset has no node.")

    with st.expander("Abstract", expanded=True):
        try:
            abstract = patent_abstract(resp["model"], pn)
        except Exception as exc:
            abstract = ""
            st.warning(f"Could not read the abstract from the index: {type(exc).__name__}: {exc}")
        st.write(abstract or "No abstract chunk was found for this patent in the index.")

    m = st.columns(5)
    m[0].metric("Patent score (reranker)" if resp["reranked"] else "Patent score (cosine)", f"{p['score']:.4f}")
    m[1].metric("Max similarity (embedding)", f"{p['max_similarity']:.4f}")
    m[2].metric("Top-3 mean (embedding)", f"{p['top3_similarity']:.4f}")
    m[3].metric("Evidence chunks", p["chunk_count"])
    m[4].metric("Decision", p["decision"] or "n/a")

    if p["chunks"]:
        best = p["chunks"][0]
        st.subheader("Why it was retrieved")
        st.write(f"Highest-scoring passage ({best['section'] or 'section unknown'}): {excerpt(best['document'], 400)}")
        st.caption("This is the passage that matched your query best. Ranking reflects textual similarity, "
                   "not a verified explanation of relevance.")

    st.subheader("Supporting chunks")
    chunks = [ch for ch in resp["candidate_chunks"] if ch["patent_number"] == pn]
    for ch in chunks[:8]:
        with st.container(border=True):
            label = (f"Section: {ch['section'] or '?'} | chunk {ch['chunk_index']} | embedding rank {ch['embedding_rank']} | "
                     f"similarity {ch['embedding_similarity']:.4f}")
            if "reranker_score" in ch:
                label += f" | reranker rank {ch['reranker_rank']} | logit {ch['reranker_score']:.3f}"
            st.markdown(f"**{label}**")
            st.write(ch["document"])
    if len(chunks) > 8:
        st.caption(f"Showing 8 of {len(chunks)} retrieved chunks for this patent.")


# ----------------------------------------------------------------------
# Page 2: Model comparison + RRF
# ----------------------------------------------------------------------
def page_compare():
    st.title("Model Comparison and RRF")
    st.caption("Each model searches its own index. Patent ranks are compared, then fused with Reciprocal Rank Fusion.")

    with st.sidebar:
        st.header("Comparison settings")
        keys = st.multiselect("Models", [k for k in MODELS if k != "legacy"],
                              default=["minilm", "bge", "e5"], format_func=lambda k: MODELS[k]["label"])
        depth = st.slider("Patents per model ranking", 5, 30, 10)
        candidates = st.slider("Candidate chunks per model", 20, 200, 100, step=10)
        rrf_k = st.number_input("RRF constant k", 1, 200, 60)
        st.caption("Loading several encoders is slow on CPU; models load one at a time, at most 3 stay cached.")

    query = st.text_area("Query", height=100, key="cmp_query", value=st.session_state.get("active_query", ""),
                         placeholder="A wireless communication system that adapts transmission parameters to channel conditions.")
    st.text_area("Known relevant patents (optional, for metrics)", height=90, key="cmp_relevance",
                 placeholder="One per line: patent_number  or  patent_number,grade   (grade 0 = judged not relevant)")
    if st.button("Run comparison", type="primary", disabled=not (query.strip() and len(keys) >= 2)):
        responses, errors = {}, {}
        bar = st.progress(0.0)
        for i, key in enumerate(keys):
            try:
                with st.spinner(f"Searching {MODELS[key]['label']}..."):
                    responses[key] = search_model(key, query, depth, candidates, False)
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"
            bar.progress((i + 1) / len(keys))
            gc.collect()
        bar.empty()
        st.session_state["cmp"] = dict(query=query, responses=responses, errors=errors, depth=depth, k=int(rrf_k),
                                       candidates=candidates, requested=list(keys))
    elif len(keys) < 2:
        st.info("Select at least two models.")

    cmp = st.session_state.get("cmp")
    if not cmp:
        return
    for key, msg in cmp["errors"].items():
        st.error(f"{MODELS[key]['label']} unavailable: {msg}")
    responses = cmp["responses"]
    if len(responses) < 1:
        return
    st.caption(f"Query: {cmp['query']} | models succeeded: {len(responses)} of {len(cmp.get('requested', responses))} "
               f"| candidates per model: {cmp.get('candidates', '?')} | RRF k = {cmp['k']}")
    if len(responses) < 2:
        st.warning("Only one model succeeded, so the 'fusion' below is just that model's ranking.")
    current = (query.strip(), sorted(keys), int(depth), int(candidates), int(rrf_k))
    ran = (cmp["query"].strip(), sorted(cmp["requested"]), int(cmp["depth"]), int(cmp["candidates"]), int(cmp["k"]))
    if current != ran:
        st.warning("The query, models, depth, candidate count or RRF k changed since this comparison ran. "
                   "Press 'Run comparison' to refresh.")

    depth_used, k_used = cmp["depth"], cmp["k"]
    rankings = {k: [p["patent_number"] for p in r["patents"][:depth_used]] for k, r in responses.items()}
    fused = rrf_fuse(rankings, k_used)
    info = {}
    for r in responses.values():
        for p in r["patents"]:
            info.setdefault(p["patent_number"], p)

    rows = []
    for pn, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        row = {"RRF rank": 0, "patent_number": pn, "title": info[pn]["title"], "RRF score": round(score, 5)}
        hits = 0
        for k, ranked in rankings.items():
            if pn in ranked:
                row[MODELS[k]["label"].split(" (")[0]] = ranked.index(pn) + 1
                hits += 1
            else:
                row[MODELS[k]["label"].split(" (")[0]] = None
        row["models_retrieved"] = hits
        rows.append(row)
    for i, row in enumerate(rows, start=1):
        row["RRF rank"] = i

    st.subheader(f"RRF ranking (k = {k_used}, top {depth_used} patents per model, {len(responses)} models)")
    st.caption("Blank = not in that model's top list (no score invented). RRF rewards patents ranked near the top by "
               "several models; its maximum depends on the number of models and the depth. RRF uses patent ranks only, "
               "so it depends on the depth chosen. Fusion is a method, not evidence of better retrieval; "
               "that needs a labelled evaluation.")
    show_df(rows[:50])

    st.subheader("Top-list overlap (Jaccard)")
    ks = list(rankings)
    overlap = pd.DataFrame(
        [[len(set(rankings[a]) & set(rankings[b])) / max(1, len(set(rankings[a]) | set(rankings[b]))) for b in ks] for a in ks],
        index=[MODELS[k]["label"].split(" (")[0] for k in ks], columns=[MODELS[k]["label"].split(" (")[0] for k in ks],
    ).round(2)
    st.dataframe(overlap)

    st.subheader("Per-model results")
    for k, r in responses.items():
        with st.expander(f"{r['label']}  |  embed {fmt_s(r['timings'].get('embedding'))}, search {fmt_s(r['timings'].get('vector_search'))}"):
            show_df([{"rank": p["rank"], "patent_number": p["patent_number"], "title": p["title"],
                      "score": round(p["score"], 4), "evidence_chunks": p["chunk_count"]} for p in r["patents"][:depth_used]])

    # ---- timings (this run) ----
    st.subheader("Timings (this run)")
    timing_rows = [{
        "system": r["label"].split(" (")[0], "embedding_s": r["timings"].get("embedding"),
        "vector_search_s": r["timings"].get("vector_search"), "total_s": r["timings"].get("total"),
        "candidate_chunks": r["candidates"], "patents_returned": len(r["patents"]),
    } for r in responses.values()]
    timing_df = pd.DataFrame(timing_rows).round(4)
    show_df(timing_df)
    st.caption("One query, no warm-up: the first call per model can include one-time cost. "
               "Use Model Evaluation for warmed multi-query latency and throughput.")

    # ---- relevance metrics (this query) ----
    out_df = timing_df
    try:
        rel = parse_relevance_text(st.session_state.get("cmp_relevance", ""))
    except ValueError as exc:
        st.error(f"Relevance list: {exc}")
        rel = {}
    if any(g > 0 for g in rel.values()):
        k_values = [k for k in EVAL_KS if k <= depth_used]
        systems = {MODELS[k]["label"].split(" (")[0]: v for k, v in rankings.items()}
        if len(responses) >= 2:
            systems["RRF"] = [pn for pn, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))][:depth_used]
        met = pd.DataFrame([dict(system=s, **ranking_metrics(r, rel, k_values)) for s, r in systems.items()]).round(4)
        st.subheader("Relevance metrics (this query)")
        show_df(met)
        st.caption("Unlisted patents count as not relevant; recall is relative to the patents you listed. "
                   "One query is an anecdote, not an evaluation: use Model Evaluation for several queries.")
        out_df = timing_df.merge(met, on="system", how="outer")
    st.download_button("Download comparison (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                       file_name="model_comparison.csv", mime="text/csv", key="cmp_dl")


# ----------------------------------------------------------------------
# Page 3: Graph explorer
# ----------------------------------------------------------------------
SECTION_DOMAINS = {
    "A": "Healthcare, Agriculture & Consumer Products", "B": "Transport, Manufacturing & Packaging",
    "C": "Chemistry, Pharmaceuticals & Materials", "D": "Textiles, Fabrics & Paper",
    "E": "Construction, Buildings & Mining", "F": "Mechanical Engineering, Engines & Heating",
    "G": "Computing, Optics & Measurement", "H": "Electronics, Electrical Systems & Communications",
}
EDGE_BANDS = [
    (0.00, 0.80, "Weight < 0.80", "#64748b"), (0.80, 0.85, "0.80-0.85", "#38bdf8"),
    (0.85, 0.90, "0.85-0.90", "#34d399"), (0.90, 0.95, "0.90-0.95", "#fbbf24"), (0.95, 1.01, "0.95+", "#f472b6"),
]


@st.cache_resource(show_spinner="Loading saved patent graph and computing layout (first time only)...")
def load_graph():
    import networkx as nx
    import pyarrow.parquet as pq

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Patent sample not found: {SAMPLE_PATH}")

    full = nx.relabel_nodes(nx.Graph(nx.read_graphml(GRAPH_PATH)), str)
    names = pq.read_schema(SAMPLE_PATH).names
    cols = [c for c in ("patent_number", "title", "abstract", "ipc_label", "cpc_label") if c in names]
    patents = pd.read_parquet(SAMPLE_PATH, columns=cols)  # skip the large full-text columns
    patents["patent_number"] = patents["patent_number"].astype(str)
    patents = patents.drop_duplicates("patent_number").set_index("patent_number", drop=False)
    missing = len(set(full.nodes()) - set(patents.index))
    patents = patents.reindex(list(full.nodes()))
    patents["patent_number"] = patents.index.astype(str)
    for c in ("title", "abstract", "ipc_label", "cpc_label"):
        patents[c] = patents[c].fillna("").astype(str) if c in patents.columns else ""
    patents["technology_domain"] = patents["ipc_label"].str[:1].map(SECTION_DOMAINS).fillna("Other / Unclassified")

    # Union of each patent's five strongest neighbours: a less cluttered display graph.
    reduced = nx.Graph()
    reduced.add_nodes_from(full.nodes())
    for node in full.nodes():
        strongest = sorted(full[node].items(), key=lambda it: float(it[1].get("weight", 0)), reverse=True)[:5]
        for nb, attrs in strongest:
            if nb != node:
                reduced.add_edge(node, nb, weight=float(attrs.get("weight", 0)))

    communities = nx.community.louvain_communities(reduced, weight="weight", resolution=1.0, seed=42)
    cluster_of = {str(n): cid for cid, members in enumerate(communities) for n in members}
    pos = nx.spring_layout(reduced, seed=42, k=0.12, iterations=100, weight="weight")
    pos = {str(n): (float(x), float(y)) for n, (x, y) in pos.items()}
    weights = [float(a["weight"]) for _, _, a in full.edges(data=True) if "weight" in a]
    return dict(full=full, reduced=reduced, patents=patents, cluster_of=cluster_of, pos=pos,
                n_comm=len(communities), missing_meta=missing,
                w_stats=(min(weights), max(weights), len(weights)) if weights else None)


def page_graph():
    import networkx as nx
    import plotly.graph_objects as go

    st.title("Patent Graph Explorer")
    st.caption("Saved 2016 patent similarity graph. Click a node, or search in the sidebar.")

    try:
        g = load_graph()
    except Exception as exc:
        st.error(f"Could not load the graph: {type(exc).__name__}: {exc}")
        return
    G, full, patents, cluster_of, pos = g["reduced"], g["full"], g["patents"], g["cluster_of"], g["pos"]
    ws = g.get("w_stats")
    if ws is None:
        st.warning("No numeric edge weights found in the GraphML; similarity bands and the slider are not meaningful.")
    else:
        st.caption(f"Saved edge weights: min {ws[0]:.3f}, max {ws[1]:.3f} across {ws[2]:,} edges "
                   "(bands and the slider treat weights as similarity scores in [0, 1]; "
                   "confirm how 'weight' was defined in the graph-building code). "
                   "Display graph = union of each patent's 5 strongest neighbours; the table below lists all saved edges.")
        if ws[0] < 0 or ws[1] > 1.0001:
            st.warning("Edge weights fall outside [0, 1]; verify what they represent before trusting the bands.")
    if g["missing_meta"]:
        st.warning(f"{g['missing_meta']} graph nodes have no row in the patent sample; they show as 'Title unavailable'.")

    # ---- selection: sidebar search + clicks (read previous chart event from state) ----
    with st.sidebar:
        st.header("Explore patents")
        q = st.text_input("Search by patent number or title", placeholder="e.g. 15006410 or wireless").strip().lower()
        domains = ["All domains"] + sorted(patents["technology_domain"].unique().tolist())
        domain = st.selectbox("Technology domain", domains)
        min_sim = st.slider("Minimum displayed edge weight", 0.75, 0.99, 0.88, 0.01,
                            help="Changes which edges are drawn, not the saved graph.")
        show_all = st.checkbox("Ignore weight threshold (show every edge among the patents currently shown)", value=False)
        options: list[str] = []
        if q:
            mask = patents["patent_number"].str.lower().str.contains(q, na=False, regex=False)
            mask |= patents["title"].str.lower().str.contains(q, na=False, regex=False)
            options = patents.loc[mask, "patent_number"].head(100).tolist()
        sb_choice = None
        if options:
            sb_choice = st.selectbox("Matching patents", options,
                                     format_func=lambda p: f"{p} - {patents.loc[p, 'title'][:80]}")
        elif q:
            st.warning("No matching patents found.")

    if sb_choice and sb_choice != st.session_state.get("_last_sb"):
        st.session_state["graph_selected"] = sb_choice
    st.session_state["_last_sb"] = sb_choice

    prev_event = st.session_state.get("graph_chart") or {}
    try:
        pts = prev_event["selection"]["points"]
    except Exception:
        pts = []
    clicked = None
    if pts:
        cd = pts[0].get("customdata")
        clicked = str(cd[0]) if cd else None
    if clicked and clicked != st.session_state.get("_last_click"):
        st.session_state["graph_selected"] = clicked
    st.session_state["_last_click"] = clicked

    selected = st.session_state.get("graph_selected")
    if selected and selected not in G:
        st.info(f"Patent {selected} is not represented in this graph sample. "
                "Absence here does not mean the patent has no related patents.")
        selected = None

    # ---- visible nodes and edges ----
    visible = set(G.nodes())
    if domain != "All domains":
        visible = set(patents.index[patents["technology_domain"] == domain])
    if selected:
        if selected not in visible:
            st.info(f"Selected patent {selected} is outside the '{domain}' filter; it is kept visible so the selection stays consistent.")
        visible.add(selected)
    edges = [(u, v, float(a.get("weight", 0))) for u, v, a in G.edges(data=True)
             if u in visible and v in visible and (show_all or float(a.get("weight", 0)) >= min_sim)]

    m = st.columns(4)
    m[0].metric("Patents shown", f"{len(visible):,}")
    m[1].metric("Displayed edges", f"{len(edges):,}")
    m[2].metric("Communities", g["n_comm"])
    m[3].metric("Connected components", nx.number_connected_components(G.subgraph(visible)) if visible else 0)

    hl_nodes, hl_edges = set(), set()
    if selected:
        hl_nodes.add(selected)
        for nb in G.neighbors(selected):
            if nb in visible:
                hl_nodes.add(nb)
                hl_edges.add(tuple(sorted((selected, nb))))

    fig = go.Figure()
    for low, high, label, color in EDGE_BANDS:
        xs, ys, hx, hy = [], [], [], []
        for u, v, s in edges:
            if not (low <= s < high):
                continue
            (x0, y0), (x1, y1) = pos[u], pos[v]
            if selected and tuple(sorted((u, v))) in hl_edges:
                hx += [x0, x1, None]; hy += [y0, y1, None]
            else:
                xs += [x0, x1, None]; ys += [y0, y1, None]
        if xs:
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=label, line=dict(color=color, width=0.7),
                                       opacity=0.2 if selected else 0.45, hoverinfo="skip"))
        if hx:
            fig.add_trace(go.Scattergl(x=hx, y=hy, mode="lines", line=dict(color=color, width=3),
                                       hoverinfo="skip", showlegend=False))

    ids = sorted(visible)
    nx_, ny_, col_, size_, op_, text_, cd_ = [], [], [], [], [], [], []
    for pid in ids:
        x, y = pos[pid]
        cid = cluster_of.get(pid, -1)
        base = f"hsl({(cid * 137.508) % 360:.1f}, 70%, 58%)"
        if selected:
            if pid == selected:
                c, s, o = "#ffffff", 17, 1.0
            elif pid in hl_nodes:
                c, s, o = base, 11, 1.0
            else:
                c, s, o = "#374151", 4, 0.18
        else:
            c, s, o = base, 6, 0.9
        nx_.append(x); ny_.append(y); col_.append(c); size_.append(s); op_.append(o)
        text_.append(patents.loc[pid, "title"] or "Title unavailable")
        cd_.append([pid, patents.loc[pid, "ipc_label"], cid, G.degree(pid)])
    fig.add_trace(go.Scatter(
        x=nx_, y=ny_, mode="markers", name="Patents", customdata=cd_, text=text_,
        marker=dict(size=size_, color=col_, opacity=op_, line=dict(width=0.5, color="#111827")),
        hovertemplate="<b>%{text}</b><br>Patent: %{customdata[0]}<br>IPC: %{customdata[1]}"
                      "<br>Community: %{customdata[2]}<br>Graph degree: %{customdata[3]}<extra></extra>",
    ))
    fig.update_layout(template="plotly_dark", title="Patent Similarity Network", height=760,
                      margin=dict(l=5, r=5, t=55, b=5), hovermode="closest",
                      legend=dict(orientation="h", y=1.02, x=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))
    show_plotly(fig, key="graph_chart", on_select="rerun", selection_mode="points",
                config={"scrollZoom": True, "displaylogo": False})

    # ---- details ----
    if not selected:
        st.info("Click a node or search in the sidebar to inspect a patent and highlight its neighbours.")
        return
    row = patents.loc[selected]
    st.divider()
    st.subheader(row["title"] or "Title unavailable")
    st.caption(f"Patent number: {selected}")
    d = st.columns(3)
    d[0].write("**Technology domain**"); d[0].write(row["technology_domain"])
    d[1].write("**IPC**"); d[1].write(row["ipc_label"] or "n/a")
    d[2].write("**CPC**"); d[2].write(row["cpc_label"] or "n/a")
    with st.expander("Abstract", expanded=True):
        st.write(row["abstract"].strip() if row["abstract"].strip().lower() not in ("", "nan") else "No abstract available.")

    st.subheader("Connected patents (all saved edges)")
    nbs = sorted(((str(n), float(a.get("weight", 0))) for n, a in full[selected].items()), key=lambda t: t[1], reverse=True)
    if not nbs:
        st.info("No connected patents found.")
        return
    show_df([{"Patent number": n, "Title": patents.loc[n, "title"] or "Title unavailable",
              "Edge weight": round(s, 4), "Technology domain": patents.loc[n, "technology_domain"]} for n, s in nbs])
    c1, c2 = st.columns([4, 1])
    nxt = c1.selectbox("Inspect a connected patent", [n for n, _ in nbs],
                       format_func=lambda p: f"{p} - {patents.loc[p, 'title'][:70]}")
    if c2.button("Select", **WIDTH_KW):
        st.session_state["graph_selected"] = nxt
        st.rerun()


# ----------------------------------------------------------------------
# Page 4: Diagnostics (system health + saved reports)
# ----------------------------------------------------------------------
def page_diagnostics():
    st.title("Diagnostics")
    st.caption("Index health, encoder/index compatibility, resource paths and a browser for saved reports. "
               "Model-versus-model metrics are on the Model Evaluation page.")

    st.header("1. System health")
    st.subheader("Index health")
    rows = []
    for key, cfg in MODELS.items():
        path, name = db_for(key)
        row = dict(model=key, collection=name, path=str(path), path_exists=path.is_dir(),
                   count=None, distance_metric=None, status="")
        if not path.is_dir():
            row["status"] = "DATABASE MISSING"
        else:
            try:
                col = open_collection(str(path), name)
                row.update(count=int(col.count()), distance_metric=get_distance_metric(col))
                row["status"] = "OK" if row["count"] > 0 else "EMPTY"
            except Exception as exc:
                row["status"] = f"{type(exc).__name__}: {exc}"[:160]
        rows.append(row)
    show_df(rows)
    if any(r["status"] != "OK" for r in rows):
        st.warning("One or more indexes are missing, empty or unreadable. Search those models only after fixing the reported issue.")
    else:
        st.success("All configured collections are accessible and non-empty.")
    if any(r["distance_metric"] not in (None, "cosine") for r in rows):
        st.warning("A collection reports a non-cosine (or unspecified) distance metric; 1 - distance is only a cosine "
                   "similarity for cosine collections.")
    st.caption(f"Seven-model run: {RUN_ID}  |  Chunk counts differ per model because each uses its own token budget.")

    st.subheader("Encoder / index compatibility")
    key = st.selectbox("Model to verify", list(MODELS.keys()), format_func=lambda k: MODELS[k]["label"], key="verify_model")
    if st.button("Verify embedding dimension (loads the encoder)"):
        try:
            col = get_collection(key)
            enc = get_encoder(key)
            got = col.get(limit=1, include=["embeddings"])
            emb = got.get("embeddings")
            if emb is None or len(emb) == 0:
                raise ValueError("The collection returned no stored embedding to compare against.")
            stored = len(emb[0])
            live = int(enc.get_sentence_embedding_dimension())
            (st.success if stored == live else st.error)(
                f"Stored vectors: {stored}-d | encoder output: {live}-d. "
                + ("Match." if stored == live else "MISMATCH: this encoder cannot query this index."))
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
        st.caption("A matching dimension does not prove the same model produced the index; it only rules out gross mismatches.")

    st.subheader("Other resources")
    show_df([
        dict(resource="GraphML graph", path=str(GRAPH_PATH), exists=GRAPH_PATH.exists()),
        dict(resource="Graph Parquet sample", path=str(SAMPLE_PATH), exists=SAMPLE_PATH.exists()),
        dict(resource="Legacy ChromaDB", path=str(LEGACY_DB), exists=LEGACY_DB.is_dir()),
        dict(resource="Results directory", path=str(RESULTS_DIR), exists=RESULTS_DIR.is_dir()),
    ])
    st.write(f"Titles resolvable from the index: **{len(title_map()):,}** patents")

    st.header("2. Saved reports")
    st.warning("Reports based on CPC subclass co-membership are a weak proxy for relevance, not verified prior-art accuracy. "
               "Check each report's relevance signal before quoting its numbers.")
    files = sorted([p for p in RESULTS_DIR.rglob("*") if p.suffix in (".csv", ".json") and p.stat().st_size < 50_000_000]) if RESULTS_DIR.is_dir() else []
    if not files:
        st.info("No CSV/JSON reports found under results/.")
        return
    chosen = st.selectbox("Report", files, format_func=lambda p: str(p.relative_to(RESULTS_DIR)))
    st.caption(f"File: {chosen} | {chosen.stat().st_size / 1024:,.1f} KB | type: {chosen.suffix.lstrip('.').upper()}")
    try:
        if chosen.suffix == ".csv":
            df = pd.read_csv(chosen)
            st.caption(f"{len(df):,} rows x {len(df.columns)} columns (showing up to 2,000 rows). "
                       "Columns: " + ", ".join(map(str, df.columns[:30])) + (" ..." if len(df.columns) > 30 else ""))
            show_df(df.head(2000))
            st.download_button("Download full report (CSV)", chosen.read_bytes(), file_name=chosen.name, mime="text/csv")
        else:
            data = json.loads(chosen.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                st.caption("Top-level keys: " + ", ".join(map(str, list(data)[:30])))
            st.json(data)
    except Exception as exc:
        st.error(f"Could not read report: {type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------
# Page 5: Model Evaluation (one-click automatic benchmark, or uploaded labels)
# ----------------------------------------------------------------------
def render_evaluation_results(run: dict, stale: bool):
    res = run["result"]
    if stale:
        st.warning("Settings changed since this run. Press 'Run full evaluation' to refresh.")
    for key, msg in res["errors"].items():
        st.error(f"{MODELS[key]['label']} was dropped (failed on at least one query): {msg}")
    if not res["rows"]:
        st.error("No model completed the evaluation.")
        return
    df = pd.DataFrame(res["rows"])
    rks = res["ks"]
    short = df[df["patents_returned"] < min(max(rks), run["depth"])]
    if len(short):
        st.warning(f"{len(short)} (system, query) pair(s) returned fewer unique patents than the largest K. "
                   "Recall is capped there; raise 'Candidate chunks per model'.")

    metric_cols = ["MRR"] + [f"{m}@{k}" for k in rks for m in ("P", "R", "nDCG")]
    summary = df.groupby("system", sort=False).agg(
        queries=("query_id", "nunique"), **{c: (c, "mean") for c in metric_cols}, unjudged_frac=("unjudged_frac", "mean")
    ).reset_index()
    summary.insert(0, "model", summary.pop("system").map(system_label))
    summary = summary.round(4)
    lat = latency_table(df)

    # ---- headline summary ----
    kref = 10 if 10 in rks else max(rks)
    st.subheader("Summary")
    st.caption(f"{summary['queries'].max()} queries | {len(summary)} systems | mode: {run['mode']}")
    c = st.columns(4)
    for col, metric in zip(c, ["MRR", f"nDCG@{kref}", f"P@{kref}", f"R@{kref}"]):
        best = summary.loc[summary[metric].idxmax()]
        col.metric(f"Best {metric}", best["model"], f"{best[metric]:.3f}", delta_color="off")
    fastest = lat.loc[lat["total_mean_s"].idxmin()]
    st.info(f"Fastest system: **{fastest['system']}** ({fastest['total_mean_s']:.2f}s per query on average). "
            f"Highest nDCG@{kref}: **{summary.loc[summary[f'nDCG@{kref}'].idxmax(), 'model']}**.")
    chart = summary.set_index("model")[[f"nDCG@{kref}", f"P@{kref}", f"R@{kref}", "MRR"]]
    st.bar_chart(chart)

    st.subheader("Relevance metrics (mean over queries)")
    show_df(summary)
    st.caption("P@K divides by K; R@K is relative to the relevant set; nDCG uses gain 2^grade - 1; "
               "MRR is over the chosen retrieval depth.")
    st.subheader("Latency and throughput")
    show_df(lat)
    st.caption("Warmed, single-query, sequential timings. Compare models on the same machine only. "
               "RRF latency is the sum of its component models.")
    with st.expander("Per-query results"):
        per_q = df.copy()
        per_q["system"] = per_q["system"].map(system_label)
        show_df(per_q.round(4))

    st.subheader("Downloads")
    full_report = summary.merge(lat.rename(columns={"system": "model"}), on="model", how="left", suffixes=("", "_lat"))
    d = st.columns(5)
    d[0].download_button("Full report (CSV)", full_report.to_csv(index=False).encode("utf-8"),
                         file_name="eval_full_report.csv", mime="text/csv", key="dl_full")
    d[1].download_button("Relevance summary (CSV)", summary.to_csv(index=False).encode("utf-8"),
                         file_name="eval_summary.csv", mime="text/csv", key="dl_sum")
    d[2].download_button("Latency summary (CSV)", lat.to_csv(index=False).encode("utf-8"),
                         file_name="eval_latency.csv", mime="text/csv", key="dl_lat")
    d[3].download_button("Per-query metrics (CSV)", df.to_csv(index=False).encode("utf-8"),
                         file_name="eval_per_query.csv", mime="text/csv", key="dl_pq")
    d[4].download_button("Ranked lists (CSV)", pd.DataFrame(res["rankings"]).to_csv(index=False).encode("utf-8"),
                         file_name="eval_rankings.csv", mime="text/csv", key="dl_rk")


def page_evaluation():
    st.title("Model Evaluation")
    st.caption("Press one button to run every model on a set of queries and get Precision@K, Recall@K, MRR, nDCG@K, "
               "latency and throughput, plus Reciprocal Rank Fusion. Summary and downloads appear when it finishes.")

    mode = st.radio("Relevance labels", ["Automatic benchmark (no files needed)", "Upload my own labels"],
                    horizontal=True, key="eval_mode")
    auto = mode.startswith("Automatic")

    queries, labels, auto_params = [], {}, None
    if auto:
        with st.expander("How the automatic benchmark works (read before quoting numbers)", expanded=True):
            st.markdown(
                "- **Queries:** randomly sampled indexed patents; the query text is that patent's abstract (or title).\n"
                "- **Relevant patents:** the other indexed patents in the same **CPC subclass** (e.g. H04L). "
                "Only subclasses with 4-60 patents are used.\n"
                "- **Self-match removed:** the query patent is excluded from every ranking.\n"
                "- **Weak proxy:** same CPC subclass is not verified prior-art relevance. Use these numbers to "
                "**compare models with each other**, not as absolute accuracy. For real accuracy, upload expert labels.\n"
                "- **Latency:** one untimed warm-up query per model, then sequential timed queries on `" + DEVICE + "`.")
        c1, c2, c3 = st.columns(3)
        n_q = c1.slider("Number of queries", 10, 100, 30, step=5, key="eval_nq")
        qsrc = c2.selectbox("Query text", ["Abstract", "Title"], key="eval_qsrc")
        seed = int(c3.number_input("Random seed", 0, 9999, 42, key="eval_seed"))
        auto_params = (n_q, seed, qsrc)
    else:
        template = pd.DataFrame([
            dict(query_id="q1", query="wireless system that adapts transmission to channel conditions", patent_number="0000001", relevance=2),
            dict(query_id="q1", query="wireless system that adapts transmission to channel conditions", patent_number="0000002", relevance=1),
            dict(query_id="q1", query="wireless system that adapts transmission to channel conditions", patent_number="0000003", relevance=0),
        ])
        st.download_button("Download label template (CSV)", template.to_csv(index=False).encode("utf-8"),
                           file_name="relevance_labels_template.csv", mime="text/csv")
        st.caption("Columns: query, patent_number, relevance (optional, default 1), query_id (optional).")
        upload = st.file_uploader("Relevance labels (CSV)", type="csv", key="eval_labels_file")
        if upload is None:
            st.info("Upload a label file to run an evaluation.")
            return
        try:
            all_q, labels = parse_label_table(pd.read_csv(upload, dtype=str))
        except Exception as exc:
            st.error(f"Could not read the label file: {type(exc).__name__}: {exc}")
            return
        queries = [q for q in all_q if any(g > 0 for g in labels[q["query_id"]].values())]
        if len(queries) < len(all_q):
            st.warning(f"{len(all_q) - len(queries)} query(ies) have no relevant patent and are skipped.")
        if not queries:
            st.error("No query has a relevant patent, so no metric can be computed.")
            return
        st.caption(f"{len(queries)} usable queries.")

    with st.expander("Settings (defaults run everything)"):
        c1, c2 = st.columns(2)
        keys = c1.multiselect("Models", [k for k in MODELS if k != "legacy"],
                              default=[k for k in MODELS if k != "legacy"],
                              format_func=lambda k: MODELS[k]["label"], key="eval_models")
        ks_sel = c2.multiselect("K values", list(EVAL_KS), default=[1, 3, 5, 10], key="eval_ks")
        c3, c4, c5, c6 = st.columns(4)
        depth = c3.slider("Patents per ranking (depth)", 5, 50, 20, key="eval_depth")
        candidates = c4.slider("Candidate chunks per model", 50, 300, 150, step=10, key="eval_cands")
        rerank = c5.toggle("Cross-encoder reranking", value=False, key="eval_rerank")
        rrf_k = int(c6.number_input("RRF constant k", 1, 200, 60, key="eval_rrfk"))
    ks = sorted(k for k in ks_sel if k <= depth)
    if ks_sel and not ks:
        st.warning("Every selected K is larger than the depth. Increase the depth or pick smaller K values.")

    label_sig = auto_params if auto else json.dumps(labels, sort_keys=True)
    sig = (mode, label_sig, tuple(sorted(keys)), depth, candidates, rerank, rrf_k, tuple(ks))
    nq = auto_params[0] if auto else len(queries)
    st.caption(f"This will run {len(keys)} models x {nq} queries (plus one warm-up each). "
               "On CPU, large models such as Nomic can take a while.")

    if st.button("Run full evaluation", type="primary", disabled=not (keys and ks)):
        try:
            if auto:
                with st.spinner("Building the benchmark from the index..."):
                    queries, labels, info = build_auto_benchmark(*auto_params)
                st.caption(f"Benchmark: {info['queries']} queries from {info['eligible']:,} eligible patents "
                           f"(universe {info['universe']:,}); average {info['avg_relevant']:.1f} relevant patents per query.")
        except Exception as exc:
            st.error(f"Could not build the benchmark: {type(exc).__name__}: {exc}")
            return
        bar = st.progress(0.0, text="Running evaluation...")
        result = run_model_evaluation(keys, queries, labels, depth, candidates, rerank, rrf_k, ks,
                                      on_progress=lambda f: bar.progress(min(f, 1.0), text="Running evaluation..."))
        bar.empty()
        st.session_state["eval_run"] = dict(sig=sig, result=result, queries=queries, depth=depth,
                                            mode="automatic (CPC-subclass proxy)" if auto else "uploaded labels")

    run = st.session_state.get("eval_run")
    if run:
        render_evaluation_results(run, stale=(run["sig"] != sig))


# ----------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------
VIEW_PAGES = {
    "Home": page_home,
    "Research Brief": page_brief,
    "Relevant patents to you": render_patent_list,
    "Patent Details": render_patent_details,
    "Advanced Settings": page_settings,
    "Model Comparison": page_compare,
    "Graph Explorer": page_graph,
    "Model Evaluation": page_evaluation,
    "Diagnostics": page_diagnostics,
}
RESEARCH_VIEWS = ["Home", "Research Brief", "Relevant patents to you", "Graph Explorer"]
TOOL_VIEWS = ["Advanced Settings", "Model Comparison", "Model Evaluation", "Diagnostics"]

initialise_state()
if st.session_state["view"] not in VIEW_PAGES:
    st.session_state["view"] = "Home"
view = st.session_state["view"]

st.sidebar.title("Patent Intelligence")


def nav_button(container, name):
    active = view == name or (view == "Patent Details" and name == "Relevant patents to you")
    container.button(name, key=f"nav_{name}", on_click=navigate_to, args=(name,),
                     type="primary" if active else "secondary", **WIDTH_KW)


for name in RESEARCH_VIEWS:
    nav_button(st.sidebar, name)
with st.sidebar.expander("Tools", expanded=view in TOOL_VIEWS):
    for name in TOOL_VIEWS:
        nav_button(st, name)
st.sidebar.divider()
VIEW_PAGES[view]()
