"""Graph logic: cosine nearest neighbours over PatentSBERTa vectors and bounded SIMILAR_TO neighbourhoods."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

MAX_NODES = 120
NOT_RECORDED = "not recorded"
_CPC_RE = re.compile(r"^[A-HY]\d{2}[A-Z]$")
_CPC_KEYS = ("cpc_subclass", "main_cpc_label", "cpc", "cpc_label", "main_cpc")


def cpc_subclass(rec: dict | None) -> str:
    """CPC subclass (e.g. G06F) from recorded metadata, or 'not recorded'. Non-CPC values are ignored."""
    meta = (rec or {}).get("metadata") or {}
    lowered = {str(k).lower(): v for k, v in meta.items()}
    for key in _CPC_KEYS:
        value = lowered.get(key)
        if value:
            code = re.sub(r"[\s/]", "", str(value).upper())[:4]
            if _CPC_RE.match(code):
                return code
    return NOT_RECORDED


SECTION_NAMES = {
    "A": "Human necessities", "B": "Operations & transport", "C": "Chemistry & metallurgy", "D": "Textiles & paper",
    "E": "Fixed constructions", "F": "Mechanical engineering, lighting, heating", "G": "Physics",
    "H": "Electricity", "Y": "Cross-sectional technologies",
}


def cpc_section(rec: dict | None) -> str:
    """CPC section letter (A-H, Y) or 'not recorded'."""
    sub = cpc_subclass(rec)
    return sub[0] if sub != NOT_RECORDED else NOT_RECORDED


def pct(cosine: float) -> float:
    """Cosine similarity as a 0-100 percentage (negative cosines clip to 0). NOT a probability."""
    return 100.0 * min(1.0, max(0.0, float(cosine)))


def result_links(matrix: np.ndarray, per_node: int = 3, threshold: float = 0.3) -> list[tuple[int, int, float]]:
    """Strongest links among a result set: each result keeps its ``per_node`` best partners above ``threshold``."""
    n = len(matrix)
    pairs: dict[tuple[int, int], float] = {}
    for i in range(n):
        row = np.array(matrix[i], dtype=float)
        row[i] = -np.inf
        for j in np.argsort(-row)[:per_node]:
            if row[j] >= threshold:
                pairs[(min(i, int(j)), max(i, int(j)))] = float(row[j])
    return [(i, j, w) for (i, j), w in pairs.items()]


class SimilarityIndex:
    """Unit-normalised vectors in RAM (N x dim float32) with exact cosine search."""

    def __init__(self, ids: list[str], matrix: np.ndarray):
        m = np.asarray(matrix, dtype=np.float32)
        if m.ndim != 2 or m.shape[0] != len(ids):
            raise ValueError("ids and matrix do not line up.")
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.ids = list(ids)
        self.matrix = m / norms
        self._row = {pid: i for i, pid in enumerate(self.ids)}

    @classmethod
    def from_index(cls, index) -> "SimilarityIndex":
        ids, matrix = index.load_vectors()
        return cls(ids, matrix)

    def __len__(self) -> int:
        return len(self.ids)

    def has(self, pid: str | None) -> bool:
        return pid in self._row

    def row(self, pid: str) -> int:
        return self._row[pid]

    def vector(self, pid: str) -> np.ndarray:
        return self.matrix[self._row[pid]].copy()

    def search(self, vec, n: int, exclude: int | None = None, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Exact cosine search over the WHOLE index (optionally restricted by a boolean ``allowed`` mask)."""
        v = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if v.ndim != 1 or v.shape[0] != self.matrix.shape[1] or norm == 0:
            raise ValueError("Query vector does not match the index (wrong size or zero).")
        sims = self.matrix @ (v / norm)
        if allowed is not None:
            sims = np.where(allowed, sims, -np.inf)
        if exclude is not None:
            sims[exclude] = -np.inf
        n = min(int(n), int(np.isfinite(sims).sum()))
        if n <= 0:
            return []
        top = np.argpartition(-sims, n - 1)[:n]
        top = sorted(top.tolist(), key=lambda i: (-float(sims[i]), self.ids[i]))
        return [(i, float(sims[i])) for i in top]

    def similarity(self, a: str, b: str) -> float:
        return float(self.matrix[self._row[a]] @ self.matrix[self._row[b]])

    def matrix_for(self, ids: list[str]) -> np.ndarray:
        sub = self.matrix[[self._row[p] for p in ids]]
        return np.clip(sub @ sub.T, -1.0, 1.0)

    def neighbours(self, pid: str, k: int, threshold: float) -> list[tuple[str, float]]:
        """Top-k most similar patents with cosine >= threshold, best first (deterministic ties)."""
        row = self._row[pid]
        sims = self.matrix @ self.matrix[row]
        sims[row] = -np.inf
        n = len(self.ids)
        kk = min(k, n - 1)
        if kk <= 0:
            return []
        top = np.argpartition(-sims, kk - 1)[:kk]
        top = sorted(top.tolist(), key=lambda i: (-float(sims[i]), self.ids[i]))
        return [(self.ids[i], float(sims[i])) for i in top if sims[i] >= threshold]


@dataclass
class Node:
    pid: str
    depth: int
    parent: str | None = None


@dataclass
class Edge:
    source: str
    target: str
    similarity: float
    tree: bool = True  # False = extra link between nodes already in view


@dataclass
class Graph:
    center: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    truncated: bool = False


def build_graph(sim: SimilarityIndex, center: str, k: int = 6, threshold: float = 0.25,
                depth: int = 1, expanded: list[str] | tuple[str, ...] = ()) -> Graph:
    """Neighbourhood of ``center`` (depth 1 or 2), plus any individually expanded nodes. Capped at MAX_NODES."""
    graph = Graph(center=center, nodes={center: Node(center, 0)})
    linked: set[frozenset] = set()
    done: set[str] = set()

    def expand(pid: str) -> None:
        if pid in done or not sim.has(pid):
            return
        done.add(pid)
        for other, score in sim.neighbours(pid, k, threshold):
            pair = frozenset((pid, other))
            if other in graph.nodes:
                if pair not in linked:
                    linked.add(pair)
                    graph.edges.append(Edge(pid, other, score, tree=False))
                continue
            if len(graph.nodes) >= MAX_NODES:
                graph.truncated = True
                continue
            graph.nodes[other] = Node(other, graph.nodes[pid].depth + 1, pid)
            linked.add(pair)
            graph.edges.append(Edge(pid, other, score, tree=True))

    expand(center)
    if depth >= 2:
        for pid in [p for p, n in graph.nodes.items() if n.depth == 1]:
            expand(pid)
    for pid in expanded:
        if pid in graph.nodes:
            expand(pid)
    return graph


def neighbourhood_rows(graph: Graph, sim: SimilarityIndex, records: dict) -> list[dict]:
    rows = []
    for pid, node in graph.nodes.items():
        rec = records.get(pid, {})
        rows.append({
            "patent": pid,
            "title": rec.get("title") or "title not recorded",
            "hop": node.depth,
            "cosine_to_centre": 1.0 if pid == graph.center else round(sim.similarity(graph.center, pid), 4),
            "linked_from": node.parent or "—",
            "cpc_subclass": cpc_subclass(rec),
        })
    rows.sort(key=lambda r: (r["hop"], -r["cosine_to_centre"], r["patent"]))
    return rows


def subclass_distribution(graph: Graph, records: dict) -> dict[str, int]:
    counts = Counter(cpc_subclass(records.get(pid)) for pid in graph.nodes)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
