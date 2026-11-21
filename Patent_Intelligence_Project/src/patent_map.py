"""Global patent map: semantic domains (clusters of PatentSBERTa vectors), a 2-D layout and strongest-neighbour links.

Domain names are NOT invented: each is made from the most distinctive title/text terms of that cluster's own patents
(class-based TF-IDF). The CPC majority is shown beside it where CPC is recorded. Heavy results (layout, clusters,
links) are cached in the graph index folder, never in ChromaDB.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.patent_graph import NOT_RECORDED, SECTION_NAMES, SimilarityIndex, cpc_section

EDGES_PER_NODE = 10
CACHE_VERSION = 1
PALETTE = ["#4CC9E6", "#8FD18B", "#C792EA", "#FF8A65", "#6C8CFF", "#D4E157",
           "#4DB6AC", "#F48FB1", "#BCAAA4", "#E57373", "#81D4FA", "#9FA8DA"]
_BOILERPLATE = {
    "invention", "present", "embodiment", "embodiments", "claim", "claims", "according", "wherein", "comprising",
    "method", "methods", "system", "systems", "device", "devices", "apparatus", "use", "may", "said", "one", "least",
    "first", "second", "thereof", "provide", "provides", "includes", "including", "based", "related", "relates",
    "field", "art", "background", "described", "example", "examples", "also", "using", "used", "plurality", "portion",
    "disclosure", "described", "various", "number", "type", "form", "means", "thereby", "configured",
}


@dataclass
class Domain:
    index: int
    label: str                 # data-derived: top distinctive terms
    color: str
    size: int
    cpc_top: str               # majority CPC section among patents with CPC recorded
    cpc_share: float
    cpc_recorded: int

    @property
    def key(self) -> str:
        return f"D{self.index + 1}"

    @property
    def title(self) -> str:
        return f"{self.key} · {self.label}"

    @property
    def cpc_hint(self) -> str:
        if self.cpc_top == NOT_RECORDED:
            return "CPC not recorded"
        return f"{self.cpc_top} {SECTION_NAMES.get(self.cpc_top, '')} ({self.cpc_share:.0%} of {self.cpc_recorded} with CPC)"


@dataclass
class MapData:
    ids: list[str]
    xy: np.ndarray
    cluster: np.ndarray
    domains: list[Domain]
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_w: np.ndarray
    edge_rank: np.ndarray
    silhouette: float | None
    layout: str


# --------------------------------------------------------------------------- #

def _choose_k(X: np.ndarray, requested: int | None) -> tuple[int, float | None]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(X)
    if requested:
        return max(1, min(int(requested), n)), None
    if n < 12:
        return 1, None
    hi = max(2, min(12, n // 8))
    cands = range(2 if hi < 3 else 3, hi + 1)
    rng = np.random.default_rng(0)
    sample = X if n <= 3000 else X[rng.choice(n, 3000, replace=False)]
    best_k, best = None, -2.0
    for k in cands:
        labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(sample)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(sample, labels))
        if score > best:
            best_k, best = k, score
    return (best_k or 2), (best if best_k else None)


def _layout(X: np.ndarray) -> tuple[np.ndarray, str]:
    n = len(X)
    if n < 4:
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[np.cos(ang), np.sin(ang)], "circle (too few patents for a layout)"
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    reduced = PCA(n_components=min(50, n - 1, X.shape[1]), random_state=0).fit_transform(X)
    if n < 6:
        xy, name = reduced[:, :2], "PCA"
    else:
        perplexity = float(min(30.0, max(2.0, (n - 1) / 3)))
        xy = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=0).fit_transform(reduced)
        name = "t-SNE"
    xy = xy - xy.mean(axis=0)
    scale = float(np.abs(xy).max()) or 1.0
    return (xy / scale).astype(np.float32), name


def _knn_edges(M: np.ndarray, k: int = EDGES_PER_NODE, block: int = 1024):
    n = len(M)
    k = min(k, n - 1)
    if k <= 0:
        z = np.zeros(0)
        return z.astype(int), z.astype(int), z.astype(np.float32), z.astype(int)
    src, dst, w, rank = [], [], [], []
    for s in range(0, n, block):
        sims = M[s:s + block] @ M.T
        b = sims.shape[0]
        sims[np.arange(b), s + np.arange(b)] = -np.inf
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-vals, axis=1)
        top, vals = np.take_along_axis(top, order, axis=1), np.take_along_axis(vals, order, axis=1)
        src.append(np.repeat(np.arange(s, s + b), k))
        dst.append(top.ravel())
        w.append(vals.ravel())
        rank.append(np.tile(np.arange(1, k + 1), b))
    src, dst, w, rank = map(np.concatenate, (src, dst, w, rank))
    a, c = np.minimum(src, dst), np.maximum(src, dst)
    key = a.astype(np.int64) * n + c
    order = np.lexsort((rank, key))            # per undirected pair, best rank first
    _, first = np.unique(key[order], return_index=True)
    keep = order[first]
    return a[keep], c[keep], w[keep].astype(np.float32), rank[keep]


def _label_domains(records: dict, ids: list[str], cluster: np.ndarray, k: int) -> list[list[str]]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    docs = []
    for c in range(k):
        parts = []
        for i in np.where(cluster == c)[0][:400]:
            r = records.get(ids[i], {})
            excerpt = (r.get("excerpt") or "")[:500]
            if len(r.get("excerpt") or "") > 500 and " " in excerpt:
                excerpt = excerpt.rsplit(" ", 1)[0]          # never feed a word cut in half to the vectoriser
            parts.append(f"{r.get('title') or ''} {excerpt}")
        docs.append(" ".join(parts))
    try:
        vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS | _BOILERPLATE), ngram_range=(1, 2), sublinear_tf=True,
                              max_df=0.6 if k >= 3 else 1.0, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
        tfidf = vec.fit_transform(docs)
    except ValueError:                      # empty vocabulary: no usable text
        return [[] for _ in range(k)]
    names = np.array(vec.get_feature_names_out())
    out = []
    for c in range(k):
        row = tfidf[c].toarray().ravel()
        terms, used = [], set()
        for j in np.argsort(-row):
            if row[j] <= 0 or len(terms) == 3:
                break
            words = set(names[j].split())
            if words & used:
                continue
            terms.append(str(names[j]))
            used |= words
        out.append(terms)
    return out


def _cache_key(sim: SimilarityIndex, n_clusters: int | None) -> str:
    h = hashlib.sha1(f"v{CACHE_VERSION}|{n_clusters or 0}|{EDGES_PER_NODE}|{sim.matrix.shape}".encode())
    h.update("|".join(sim.ids).encode())
    h.update(np.ascontiguousarray(sim.matrix[:, :8]).tobytes())
    return h.hexdigest()[:16]


def build_map(sim: SimilarityIndex, records: dict, cache_dir: str | Path | None = None, n_clusters: int | None = None) -> MapData:
    """Domains + layout + links for every indexed patent. Expensive parts are cached beside the graph index."""
    ids, X = sim.ids, sim.matrix
    cache = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache = cache_dir / f"map_cache_{_cache_key(sim, n_clusters)}.npz"
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        xy, cluster, k = z["xy"], z["cluster"], int(z["k"])
        edges = (z["src"], z["dst"], z["w"], z["rank"])
        sil = float(z["sil"]) if float(z["sil"]) > -2 else None
        layout = str(z["layout"])
    else:
        k, sil = _choose_k(X, n_clusters)
        if k > 1:
            from sklearn.cluster import KMeans
            raw = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        else:
            raw = np.zeros(len(X), dtype=int)
        sizes = np.bincount(raw, minlength=k)
        remap = {old: new for new, old in enumerate(np.argsort(-sizes, kind="stable"))}
        cluster = np.array([remap[c] for c in raw], dtype=int)
        xy, layout = _layout(X)
        edges = _knn_edges(X)
        if cache is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, xy=xy, cluster=cluster, k=k, src=edges[0], dst=edges[1], w=edges[2], rank=edges[3],
                                sil=-9.0 if sil is None else sil, layout=layout)
            os.replace(tmp, cache)
            for old in cache_dir.glob("map_cache_*.npz"):
                if old != cache:
                    old.unlink(missing_ok=True)

    terms = _label_domains(records, ids, cluster, k)
    domains = []
    for c in range(k):
        members = np.where(cluster == c)[0]
        sections = [s for s in (cpc_section(records.get(ids[i])) for i in members) if s != NOT_RECORDED]
        top, share = NOT_RECORDED, 0.0
        if sections:
            vals, counts = np.unique(sections, return_counts=True)
            top, share = str(vals[counts.argmax()]), float(counts.max() / len(sections))
        domains.append(Domain(c, " · ".join(terms[c]) or f"domain {c + 1}", PALETTE[c % len(PALETTE)],
                              int(len(members)), top, share, len(sections)))
    return MapData(ids, xy, cluster, domains, edges[0], edges[1], edges[2], edges[3], sil, layout)
