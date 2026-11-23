"""Document-level search built on top of the existing ``src.rag_search.search``.

The retrieval implementation is NOT duplicated: each section of an uploaded
document is sent through the same ``search()`` used for plain-text queries, and
the per-section patent rankings are merged with Reciprocal Rank Fusion (RRF).
RRF needs no score calibration across sections, so it is safe whether or not the
cross-encoder reranker is on.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

RRF_K = 60


def _chunk_strength(chunk: dict) -> float:
    if "reranker_score" in chunk:
        return float(chunk["reranker_score"])
    return float(chunk.get("embedding_similarity", 0.0))


def search_document(
    sections: list[str],
    *,
    collection: Any,
    embedding_model: Any,
    top_k: int,
    candidate_count: int,
    reranker: Any = None,
    search_fn: Callable[..., dict] | None = None,
) -> dict:
    """Search each section and fuse the rankings into one response.

    The returned dict follows the shape of ``rag_search.search`` (same keys the UI
    reads) plus ``mode``, ``sections_searched`` and ``section_errors``. Per patent
    it adds ``matched_sections``. ``score`` is the RRF score (a rank-fusion value,
    not a similarity or probability).
    """
    if search_fn is None:
        from src.rag_search import search as search_fn  # lazy: keeps this module testable

    sections = [s for s in sections if s and s.strip()]
    if not sections:
        raise ValueError("There is no document text to search with.")

    agg: dict[str, dict] = {}
    base_response: dict | None = None
    all_candidates: list[dict] = []
    timings: dict[str, float] = defaultdict(float)
    retrieved_total = 0
    errors: list[str] = []
    searched = 0

    for idx, section in enumerate(sections):
        try:
            resp = search_fn(
                query=section,
                collection=collection,
                embedding_model=embedding_model,
                top_k=top_k,
                candidate_count=candidate_count,
                reranker=reranker,
            )
        except Exception as exc:  # keep going; report at the end
            errors.append(f"Section {idx + 1}: {exc}")
            continue
        searched += 1
        base_response = base_response or resp
        for name, value in (resp.get("timings") or {}).items():
            if isinstance(value, (int, float)):
                timings[name] += float(value)
        retrieved_total += int((resp.get("pipeline") or {}).get("retrieved_candidates", 0))
        all_candidates.extend(resp.get("candidate_chunks", []))

        for patent in resp.get("results", []):
            key = str(patent.get("patent_number"))
            entry = agg.setdefault(
                key, {"patent": patent, "rrf": 0.0, "sections": set(), "chunks": {}}
            )
            entry["rrf"] += 1.0 / (RRF_K + int(patent.get("rank", 1)))
            entry["sections"].add(idx)
            for chunk in patent.get("chunks", []):
                slot = chunk.get("chunk_index", id(chunk))
                if slot not in entry["chunks"] or _chunk_strength(chunk) > _chunk_strength(entry["chunks"][slot]):
                    entry["chunks"][slot] = {**chunk, "matched_section": idx + 1}

    if base_response is None:
        raise RuntimeError("Every section search failed. " + " | ".join(errors[:3]))

    results = []
    for entry in agg.values():
        chunks = sorted(entry["chunks"].values(), key=_chunk_strength, reverse=True)
        merged = dict(entry["patent"])
        merged.update(
            score=entry["rrf"],
            chunks=chunks,
            retrieved_chunk_count=len(chunks),
            matched_sections=len(entry["sections"]),
            max_similarity=max((float(c.get("embedding_similarity", 0.0)) for c in chunks), default=0.0),
        )
        merged.pop("top3_mean_similarity", None)
        results.append(merged)

    results.sort(key=lambda p: (p["score"], p["max_similarity"]), reverse=True)
    results = results[:top_k]
    for rank, patent in enumerate(results, start=1):
        patent["rank"] = rank

    response = {k: v for k, v in base_response.items() if k not in ("results", "candidate_chunks")}
    response.update(
        results=results,
        candidate_chunks=all_candidates,
        timings=dict(timings),
        candidate_count=candidate_count,
        pipeline={
            "retrieved_candidates": retrieved_total,
            "unique_patents": len(agg),
            "returned_patents": len(results),
            "reranked": bool(reranker),
        },
        mode="document",
        sections_searched=searched,
        section_errors=errors,
    )
    return response
