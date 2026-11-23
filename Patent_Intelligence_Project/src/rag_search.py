"""Chroma-backed retrieval helpers for the Streamlit patent app."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_DIR / "data" / "chroma_db"
DEFAULT_COLLECTION_NAME = "patent_rag_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class PatentCollection:
    """Wrapper for a persistent Chroma collection."""

    path: Path
    name: str
    collection: Any

    def count(self) -> int:
        return int(self.collection.count())


def _ensure_embedding_model(embedding_model):
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedding_model


def _resolve_db_path(requested: str | Path | None) -> Path:
    if requested is None:
        return DEFAULT_DB_PATH
    return Path(requested).expanduser().resolve()


def open_collection(db_path: str | Path | None = None, collection_name: str = DEFAULT_COLLECTION_NAME) -> PatentCollection:
    """Open the existing ChromaDB collection without rebuilding it."""

    from chromadb import PersistentClient

    resolved_path = _resolve_db_path(db_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"ChromaDB directory not found: {resolved_path}")

    client = PersistentClient(path=str(resolved_path))
    available_names = [collection.name for collection in client.list_collections()]
    if collection_name not in available_names:
        available = ", ".join(available_names) if available_names else "<none>"
        raise ValueError(
            f"Collection '{collection_name}' was not found in {resolved_path}. "
            f"Available collections: {available}"
        )

    return PatentCollection(
        path=resolved_path,
        name=collection_name,
        collection=client.get_collection(name=collection_name),
    )


def _similarity_from_distance(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return float(1.0 - float(distance))


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return metadata or {}


def _chunk_score(chunk: dict[str, Any], score_key: str) -> float:
    value = chunk.get(score_key)
    return float(value) if value is not None else float("-inf")


def _patent_score(chunks: list[dict[str, Any]], score_key: str) -> dict[str, float]:
    ordered = sorted(chunks, key=lambda item: _chunk_score(item, score_key), reverse=True)
    values = [_chunk_score(chunk, score_key) for chunk in ordered]
    top_values = values[:3]
    max_score = values[0] if values else 0.0
    top3_mean = mean(top_values) if top_values else 0.0
    return {
        "max_score": float(max_score),
        "top3_mean_score": float(top3_mean),
        "score": float(0.5 * max_score + 0.5 * top3_mean),
    }


def search(
    query: str,
    collection: PatentCollection,
    embedding_model=None,
    top_k: int = 10,
    candidate_count: int = 50,
    reranker=None,
):
    """Search the Chroma collection and aggregate the top chunks by patent."""

    embedding_model = _ensure_embedding_model(embedding_model)
    collection_count = collection.count()

    if collection_count <= 0:
        return {
            "query": query,
            "collection_name": collection.name,
            "collection_path": str(collection.path),
            "collection_count": 0,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME if reranker is not None else None,
            "requested_candidate_count": max(1, int(candidate_count)),
            "candidate_count": 0,
            "top_k": int(top_k),
            "results": [],
            "candidate_chunks": [],
            "pipeline": {
                "retrieved_candidates": 0,
                "unique_patents": 0,
                "returned_patents": 0,
                "reranked": reranker is not None,
            },
            "timings": {},
        }

    timings: dict[str, float] = {}
    t0 = perf_counter()
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    timings["embedding"] = perf_counter() - t0

    t1 = perf_counter()
    requested_candidates = min(max(1, int(candidate_count)), collection_count)
    raw = collection.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=requested_candidates,
        include=["documents", "metadatas", "distances"],
    )
    timings["vector_search"] = perf_counter() - t1

    documents = raw.get("documents", [[]])[0] or []
    metadatas = raw.get("metadatas", [[]])[0] or []
    distances = raw.get("distances", [[]])[0] or []

    chunks: list[dict[str, Any]] = []
    for rank, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        safe_metadata = _safe_metadata(metadata)
        chunks.append(
            {
                "chunk_index": safe_metadata.get("chunk_index", rank - 1),
                "document": document or "",
                "patent_number": str(safe_metadata.get("patent_number", "Unknown")),
                "title": str(safe_metadata.get("title", "Title unavailable")),
                "decision": str(safe_metadata.get("decision", safe_metadata.get("main_cpc_label", ""))),
                "embedding_rank": rank,
                "distance": float(distance),
                "embedding_similarity": _similarity_from_distance(distance),
            }
        )

    candidate_count = min(requested_candidates, len(chunks))
    candidate_chunks = chunks[:candidate_count]
    ordered_chunks = candidate_chunks
    score_key = "embedding_similarity"

    if reranker is not None and candidate_chunks:
        t2 = perf_counter()
        pairs = [(query, chunk["document"]) for chunk in candidate_chunks]
        reranker_scores = np.asarray(reranker.predict(pairs, show_progress_bar=False), dtype=np.float32)
        for chunk, score in zip(candidate_chunks, reranker_scores):
            chunk["reranker_score"] = float(score)
        ordered_chunks = sorted(candidate_chunks, key=lambda chunk: chunk["reranker_score"], reverse=True)
        for reranker_rank, chunk in enumerate(ordered_chunks, start=1):
            chunk["reranker_rank"] = reranker_rank
        timings["reranking"] = perf_counter() - t2
        score_key = "reranker_score"

    patent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in ordered_chunks:
        patent_groups[chunk["patent_number"]].append(chunk)

    patents: list[dict[str, Any]] = []
    for patent_number, patent_chunks in patent_groups.items():
        patent_chunks = sorted(patent_chunks, key=lambda chunk: _chunk_score(chunk, score_key), reverse=True)
        similarity_scores = _patent_score(patent_chunks, "embedding_similarity")
        patent_scores = _patent_score(patent_chunks, score_key)
        patent = {
            "patent_number": patent_number,
            "title": patent_chunks[0]["title"] if patent_chunks else "",
            "decision": patent_chunks[0]["decision"] if patent_chunks else "",
            "score": patent_scores["score"],
            "max_similarity": similarity_scores["max_score"],
            "top3_mean_similarity": similarity_scores["top3_mean_score"],
            "retrieved_chunk_count": len(patent_chunks),
            "chunks": patent_chunks[:3],
        }
        if any("reranker_score" in chunk for chunk in patent_chunks):
            rerank_scores = _patent_score(patent_chunks, "reranker_score")
            patent["max_reranker_score"] = rerank_scores["max_score"]
            patent["top3_mean_reranker_score"] = rerank_scores["top3_mean_score"]
        patents.append(patent)

    patents.sort(key=lambda patent: patent["score"], reverse=True)
    for rank, patent in enumerate(patents, start=1):
        patent["rank"] = rank

    top_results = patents[: min(int(top_k), len(patents))]

    return {
        "query": query,
        "collection_name": collection.name,
        "collection_path": str(collection.path),
        "collection_count": collection_count,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME if reranker is not None else None,
        "requested_candidate_count": requested_candidates,
        "candidate_count": len(candidate_chunks),
        "top_k": int(top_k),
        "results": top_results,
        "candidate_chunks": ordered_chunks,
        "pipeline": {
            "retrieved_candidates": len(candidate_chunks),
            "unique_patents": len(patents),
            "returned_patents": len(top_results),
            "reranked": reranker is not None,
        },
        "timings": timings,
    }
