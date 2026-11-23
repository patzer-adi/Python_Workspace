"""PatentSBERTa graph index.

* Embedding model: ``AI-Growth-Lab/PatentSBERTa`` only. No fallback model, no stub encoder.
* Storage: a separate SQLite file (default ``data/graph_index/patent_sbert_graph.sqlite``).
  The existing Chroma collections are opened read-only (``get_collection`` only) and never written.
* Interrupted runs are safe: every batch is its own transaction, and a re-run resumes.

CLI (run from the project root):
    python -m src.patent_sbert_embeddings inspect
    python -m src.patent_sbert_embeddings build --limit 50
    python -m src.patent_sbert_embeddings status
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
DEFAULT_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512          # PatentSBERTa input window in tokens; longer text is truncated by the tokenizer
MAX_TEXT_CHARS = 6000         # cap per patent kept in RAM (~1,500 tokens, comfortably above the window)
MIN_FULL_TEXT_MEDIAN = 1500   # median chars of title+abstract+description below this = not full text
MIN_RECORD_CHARS = 200
EXCERPT_CHARS = 1500
SCHEMA_VERSION = 1

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_PATH = PROJECT_DIR / "data" / "chroma_db"
DEFAULT_GRAPH_DB = PROJECT_DIR / "data" / "graph_index" / "patent_sbert_graph.sqlite"
PATENT_COLLECTION = "patent_rag"
CHUNK_COLLECTION = "patent_rag_chunks"

_ID_KEYS = ("patent_number", "patent_id", "publication_number", "id")
_TITLE_KEYS = ("title", "patent_title", "invention_title")
_ABSTRACT_KEYS = ("abstract", "patent_abstract")
_BODY_KEYS = ("description", "detailed_description", "full_text", "fulltext", "body", "text")
_HIDDEN_META = set(_BODY_KEYS) | {"claims", "document"}


class IndexMismatchError(RuntimeError):
    """The existing graph index was built with a different model/configuration."""


class MissingPatentTextError(RuntimeError):
    """Full patent-level text is not available from the chosen source."""


class ModelLoadError(RuntimeError):
    """PatentSBERTa could not be loaded."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def normalize_id(value: Any) -> str:
    """Canonical patent id: str, no spaces/commas, no float '.0' suffix, upper-case."""
    text = re.sub(r"[\s,]", "", str(value if value is not None else ""))
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text.upper()


def get_graph_db_path() -> Path:
    path = Path(os.environ.get("PATENT_GRAPH_DB") or DEFAULT_GRAPH_DB).expanduser().resolve()
    chroma = Path(os.environ.get("PATENT_CHROMA_DB") or DEFAULT_CHROMA_PATH).expanduser().resolve()
    if path == chroma or chroma in path.parents:
        raise ValueError(f"The graph index must live outside the ChromaDB directory ({chroma}).")
    return path


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _first(meta: dict, keys: tuple[str, ...]) -> str:
    lowered = {str(k).lower(): v for k, v in (meta or {}).items()}
    for key in keys:
        value = lowered.get(key)
        if value is not None and _clean(value):
            return _clean(value)
    return ""


def build_text(title: str, abstract: str, body: str, cap: int = MAX_TEXT_CHARS) -> str:
    """Embedding input, in priority order: title, abstract, description/body."""
    t, a, b = _clean(title), _clean(abstract), _clean((body or "")[: cap * 2])
    if t and b.startswith(t):
        b = b[len(t):].lstrip(" .:-")
    parts = [t] if t else []
    if a and a not in b:
        parts.append(a)
    if b:
        parts.append(b)
    return " ".join(parts)[:cap]


def _public_metadata(meta: dict) -> dict:
    out: dict[str, Any] = {}
    for key, value in (meta or {}).items():
        k = str(key)
        if k.lower() in _HIDDEN_META or k.lower().startswith("chunk") or k.lower() == "title":
            continue
        if isinstance(value, (str, int, float, bool)) and str(value).strip() != "":
            out[k] = str(value)[:2000] if k.lower() == "abstract" else (value if not isinstance(value, str) else value[:300])
    return out


def _finalize(parts: dict) -> dict:
    """parts: title, abstract, body, metadata -> record the index and UI use."""
    meta = dict(parts["metadata"])
    if parts["abstract"]:
        meta["abstract"] = parts["abstract"][:2000]
    text = build_text(parts["title"], parts["abstract"], parts["body"])
    return {"title": parts["title"], "metadata": meta, "text": text, "excerpt": text[:EXCERPT_CHARS],
            "body_chars": len(_clean(parts["body"][: MAX_TEXT_CHARS * 2]))}


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS patents (
    pid TEXT PRIMARY KEY, title TEXT, metadata TEXT, excerpt TEXT,
    text_hash TEXT NOT NULL, meta_hash TEXT NOT NULL, vector BLOB NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class GraphIndex:
    """SQLite-backed patent vector store. One transaction per write batch."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con: sqlite3.Connection | None = sqlite3.connect(self.path)
        self._con.executescript(_SCHEMA)
        self._con.commit()

    @property
    def con(self) -> sqlite3.Connection:
        if self._con is None:
            raise RuntimeError("GraphIndex is closed.")
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def get_meta(self) -> dict[str, str]:
        return dict(self.con.execute("SELECT key, value FROM meta"))

    def set_meta(self, values: dict[str, Any]) -> None:
        with self.con:
            self.con.executemany(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [(k, str(v)) for k, v in values.items()],
            )

    def count(self) -> int:
        return int(self.con.execute("SELECT COUNT(*) FROM patents").fetchone()[0])

    def status(self) -> dict:
        meta = self.get_meta()
        return {
            "count": self.count(),
            "model": meta.get("model"),
            "dim": int(meta["dim"]) if meta.get("dim") else None,
            "max_seq_length": meta.get("max_seq_length"),
            "updated": meta.get("updated"),
            "path": str(self.path),
        }

    def check_compatible(self, model_name: str, dim: int) -> None:
        meta = self.get_meta()
        if not meta and self.count() == 0:
            return
        if meta.get("model") != model_name:
            raise IndexMismatchError(
                f"This index was built with {meta.get('model')!r}, not {model_name!r}. "
                "Use 'Rebuild' to start a new graph index (the Chroma data is not touched)."
            )
        if meta.get("dim") and int(meta["dim"]) != dim:
            raise IndexMismatchError(f"Stored vector size is {meta['dim']} but the model produces {dim}.")
        if meta.get("schema_version") and int(meta["schema_version"]) != SCHEMA_VERSION:
            raise IndexMismatchError("The index schema version differs; rebuild the graph index.")

    def current_hashes(self) -> dict[str, tuple[str, str]]:
        return {r[0]: (r[1], r[2]) for r in self.con.execute("SELECT pid, text_hash, meta_hash FROM patents")}

    def upsert(self, rows: list[tuple[str, str, str, str, str, str, np.ndarray]]) -> None:
        """rows: (pid, title, metadata_json, excerpt, text_hash, meta_hash, vector)."""
        with self.con:
            self.con.executemany(
                "INSERT INTO patents(pid,title,metadata,excerpt,text_hash,meta_hash,vector) VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(pid) DO UPDATE SET title=excluded.title, metadata=excluded.metadata, "
                "excerpt=excluded.excerpt, text_hash=excluded.text_hash, meta_hash=excluded.meta_hash, "
                "vector=excluded.vector",
                [(p, t, m, e, th, mh, np.asarray(v, dtype="<f4").tobytes()) for p, t, m, e, th, mh, v in rows],
            )

    def update_metadata(self, rows: list[tuple[str, str, str, str, str]]) -> None:
        """rows: (pid, title, metadata_json, excerpt, meta_hash); vector untouched."""
        with self.con:
            self.con.executemany(
                "UPDATE patents SET title=?, metadata=?, excerpt=?, meta_hash=? WHERE pid=?",
                [(t, m, e, mh, p) for p, t, m, e, mh in rows],
            )

    def all_metadata(self) -> dict[str, dict]:
        out = {}
        for pid, title, meta, excerpt in self.con.execute("SELECT pid, title, metadata, excerpt FROM patents ORDER BY pid"):
            out[pid] = {"title": title or "", "metadata": json.loads(meta or "{}"), "excerpt": excerpt or ""}
        return out

    def load_vectors(self) -> tuple[list[str], np.ndarray]:
        n = self.count()
        dim = int(self.get_meta().get("dim") or 0)
        ids: list[str] = []
        matrix = np.empty((n, dim), dtype=np.float32)
        for i, (pid, blob) in enumerate(self.con.execute("SELECT pid, vector FROM patents ORDER BY pid")):
            vec = np.frombuffer(blob, dtype="<f4")
            if vec.shape[0] != dim:
                raise IndexMismatchError(f"Patent {pid} has a vector of size {vec.shape[0]}, expected {dim}.")
            matrix[i] = vec
            ids.append(pid)
        return ids, matrix

    def clear(self) -> None:
        """Empty THIS graph index only (never touches Chroma)."""
        with self.con:
            self.con.execute("DELETE FROM patents")
            self.con.execute("DELETE FROM meta")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def load_patent_sbert():
    """Load the real PatentSBERTa on CPU. Raises ModelLoadError with the exact cause; never substitutes."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ModelLoadError(f"sentence-transformers is not installed: {exc}") from exc
    try:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
    except Exception as exc:
        raise ModelLoadError(f"Could not load {MODEL_NAME}: {type(exc).__name__}: {exc}") from exc
    if (getattr(model, "max_seq_length", None) or 0) > MAX_SEQ_LENGTH:
        model.max_seq_length = MAX_SEQ_LENGTH
    return model


def _text_hash(text: str, max_len: int) -> str:
    return hashlib.sha256(f"{MODEL_NAME}|{max_len}|{text}".encode("utf-8")).hexdigest()


def _meta_hash(rec: dict) -> str:
    payload = json.dumps([rec.get("title", ""), rec.get("metadata", {})], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_index(records: dict[str, dict], index: GraphIndex, model, batch_size: int = DEFAULT_BATCH_SIZE,
                progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Encode new/changed patents with PatentSBERTa and store them. Additive; resumable."""
    dim = int(model.get_sentence_embedding_dimension())
    max_len = int(getattr(model, "max_seq_length", None) or MAX_SEQ_LENGTH)
    index.check_compatible(MODEL_NAME, dim)
    existing = index.current_hashes()

    todo, meta_only = [], []
    for pid, rec in records.items():
        if not rec.get("text"):
            raise ValueError(f"Patent {pid} has no text to embed.")
        th, mh = _text_hash(rec["text"], max_len), _meta_hash(rec)
        old = existing.get(pid)
        if old is None or old[0] != th:
            todo.append((pid, rec, th, mh))
        elif old[1] != mh:
            meta_only.append((pid, rec, mh))

    meta = index.get_meta()
    index.set_meta({
        "model": MODEL_NAME, "dim": dim, "normalized": "true", "max_seq_length": max_len,
        "schema_version": SCHEMA_VERSION, "created": meta.get("created") or _now(),
        "text_policy": f"title + abstract + description, capped at {MAX_TEXT_CHARS} chars, then truncated to {max_len} tokens",
    })

    if meta_only:
        index.update_metadata([(p, r["title"], json.dumps(r["metadata"]), r["excerpt"], mh) for p, r, mh in meta_only])

    total, done, group = len(todo), 0, max(batch_size, 32)
    for start in range(0, total, group):
        part = todo[start:start + group]
        vecs = np.asarray(
            model.encode([r["text"] for _, r, _, _ in part], batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        if vecs.shape != (len(part), dim) or not np.isfinite(vecs).all():
            raise RuntimeError(f"Encoder returned an invalid array of shape {vecs.shape} (expected {(len(part), dim)}).")
        if not np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3):
            raise RuntimeError("Encoder output is not unit-normalised.")
        index.upsert([(p, r["title"], json.dumps(r["metadata"]), r["excerpt"], th, mh, v)
                      for (p, r, th, mh), v in zip(part, vecs)])
        index.set_meta({"updated": _now()})
        done += len(part)
        if progress:
            progress(done, total, f"Encoded {done:,} of {total:,} patents")
    if progress and total == 0:
        progress(1, 1, "Nothing new to encode")
    if total == 0 and meta_only:
        index.set_meta({"updated": _now()})
    return {"encoded": total, "already_current": len(records) - total, "metadata_updated": len(meta_only),
            "total": index.count(), "dim": dim, "model": MODEL_NAME}


# --------------------------------------------------------------------------- #
# Reading existing Chroma data (read-only)
# --------------------------------------------------------------------------- #

def _open_collection(name: str, db_path: str | Path | None = None):
    from chromadb import PersistentClient

    path = Path(db_path or os.environ.get("PATENT_CHROMA_DB") or DEFAULT_CHROMA_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ChromaDB directory not found: {path}")
    client = PersistentClient(path=str(path))
    names = [getattr(c, "name", c) for c in client.list_collections()]
    if name not in names:
        raise ValueError(f"Collection '{name}' not found in {path}. Available: {', '.join(names) or '<none>'}")
    return client.get_collection(name=name)  # never get_or_create


def _pages(col, include: list[str], size: int = 2000):
    offset, total = 0, col.count()
    while offset < total:
        batch = col.get(include=include, limit=size, offset=offset)
        if not batch["ids"]:
            break
        yield batch
        offset += size


def _from_patent_rag(db_path) -> tuple[dict[str, dict], dict]:
    col = _open_collection(PATENT_COLLECTION, db_path)
    total = col.count()
    parts: dict[str, dict] = {}
    duplicates, skipped, key_seen, doc_lens = 0, [], set(), []
    fields = {"abstract": 0, "description_metadata": 0, "document_text": 0}
    for batch in _pages(col, ["metadatas", "documents"]):
        for cid, meta, doc in zip(batch["ids"], batch["metadatas"], batch["documents"]):
            meta = meta or {}
            key_seen.update(str(k) for k in meta)
            doc_lens.append(len(doc or ""))
            pid = normalize_id(_first(meta, _ID_KEYS) or cid)
            title, abstract = _first(meta, _TITLE_KEYS), _first(meta, _ABSTRACT_KEYS)
            body = _first(meta, _BODY_KEYS)
            source = "description_metadata" if body else "document_text"
            if not body:
                body = doc or ""
            if len(_clean(abstract)) + len(_clean(body[:MAX_TEXT_CHARS])) < MIN_RECORD_CHARS:
                skipped.append(pid)
                continue
            cand = {"title": title, "abstract": abstract, "body": body[: MAX_TEXT_CHARS * 2],
                    "metadata": _public_metadata(meta), "_src": source}
            if pid in parts:
                duplicates += 1
                old = parts[pid]
                if len(cand["body"]) + len(cand["abstract"]) <= len(old["body"]) + len(old["abstract"]):
                    continue
            parts[pid] = cand
    records = {}
    for pid, p in parts.items():
        fields[p.pop("_src")] += 1
        fields["abstract"] += 1 if p["abstract"] else 0
        records[pid] = _finalize(p)
    lens = [r["body_chars"] + len(r["metadata"].get("abstract", "")) + len(r["title"]) for r in records.values()]
    med = int(median(lens)) if lens else 0
    if not records or med < MIN_FULL_TEXT_MEDIAN:
        raise MissingPatentTextError(
            f"'{PATENT_COLLECTION}' does not hold full patent-level text. Records: {total}; usable: {len(records)}; "
            f"median title+abstract+description length: {med} chars (need >= {MIN_FULL_TEXT_MEDIAN}). "
            f"Metadata keys present: {sorted(key_seen) or 'none'}. "
            f"Stored document length (first {len(doc_lens)}): median {int(median(doc_lens)) if doc_lens else 0} chars. "
            "Missing: abstract/description text per patent. Either supply it via the metadata CSV option, "
            "or choose the 'chunks' source explicitly to reconstruct text from all chunks of each patent."
        )
    notes = [
        f"{PATENT_COLLECTION}: {total} records, {len(records)} usable patents; median text {med} chars.",
        f"Text fields: {fields['abstract']} with abstract, {fields['description_metadata']} with description metadata, "
        f"{fields['document_text']} from the stored document.",
    ]
    if duplicates:
        notes.append(f"{duplicates} duplicate id(s) merged (longest text kept).")
    if skipped:
        notes.append(f"{len(skipped)} record(s) skipped for too little text, e.g. {', '.join(skipped[:5])}.")
    return records, {"source": PATENT_COLLECTION, "total_in_source": total, "skipped": len(skipped), "notes": notes}


def _join_overlap(prev: str, nxt: str, max_overlap: int = 400) -> str:
    for k in range(min(max_overlap, len(prev), len(nxt)), 19, -1):
        if prev.endswith(nxt[:k]):
            return nxt[k:]
    return nxt


def _from_chunks(db_path) -> tuple[dict[str, dict], dict]:
    col = _open_collection(CHUNK_COLLECTION, db_path)
    total_chunks = col.count()
    first: dict[str, dict] = {}
    for batch in _pages(col, ["metadatas"], size=5000):
        for meta in batch["metadatas"]:
            meta = meta or {}
            raw = meta.get("patent_number")
            if raw is None:
                continue
            first.setdefault(normalize_id(raw), {"raw": raw, "meta": meta})
    if not first:
        raise MissingPatentTextError(f"'{CHUNK_COLLECTION}' has no chunks with a 'patent_number' metadata field.")
    records, gaps = {}, []
    for pid, item in first.items():
        got = col.get(where={"patent_number": item["raw"]}, include=["documents", "metadatas"])
        rows = sorted(
            ((int((m or {}).get("chunk_index", 0)), d or "") for m, d in zip(got["metadatas"], got["documents"])),
            key=lambda r: r[0],
        )
        idx = [r[0] for r in rows]
        if idx and idx != list(range(idx[0], idx[0] + len(idx))):
            gaps.append(pid)
        body, last = "", ""
        for _, doc in rows:
            piece = _join_overlap(last, doc) if last else doc
            body += (" " if body else "") + piece
            last = doc
            if len(body) >= MAX_TEXT_CHARS * 2:
                break
        meta = item["meta"]
        records[pid] = _finalize({"title": _first(meta, _TITLE_KEYS), "abstract": _first(meta, _ABSTRACT_KEYS),
                                  "body": body, "metadata": _public_metadata(meta)})
    notes = [
        f"{CHUNK_COLLECTION}: {total_chunks:,} chunks reconstructed into {len(records):,} patents (chunks ordered by chunk_index, "
        f"overlaps trimmed, capped at {MAX_TEXT_CHARS} chars; the model reads the first ~{MAX_SEQ_LENGTH} tokens).",
        "Abstract and description are not separated in chunks; the text is the ordered chunk concatenation.",
    ]
    if gaps:
        notes.append(f"{len(gaps)} patent(s) have non-contiguous chunk indices, e.g. {', '.join(gaps[:5])}.")
    return records, {"source": CHUNK_COLLECTION, "total_in_source": len(first), "skipped": 0, "notes": notes}


def _apply_manifest(records: dict[str, dict], path: str, notes: list[str]) -> None:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        id_col = next((cols[k] for k in _ID_KEYS if k in cols), None)
        if id_col is None:
            raise ValueError(f"The metadata CSV needs one of these id columns: {', '.join(_ID_KEYS)}.")
        matched = unmatched = 0
        for row in reader:
            pid = normalize_id(row.get(id_col))
            rec = records.get(pid)
            if rec is None:
                unmatched += 1
                continue
            matched += 1
            extra = {c.strip().lower(): v.strip() for c, v in row.items() if c and c != id_col and v and v.strip()}
            abstract = extra.pop("abstract", "")
            if abstract:
                rec["metadata"]["abstract"] = abstract[:2000]
            rec["metadata"].update({k: v[:300] for k, v in extra.items() if k not in _HIDDEN_META})
            if abstract:
                rec["text"] = build_text(rec["title"], abstract, rec["text"])
                rec["excerpt"] = rec["text"][:EXCERPT_CHARS]
    notes.append(f"Metadata CSV: {matched} row(s) matched, {unmatched} did not match any patent in the source.")


def load_records(source: str = "auto", manifest_csv: str | None = None, db_path: str | Path | None = None,
                 limit: int | None = None) -> tuple[dict[str, dict], dict]:
    """Return (records, info). ``auto`` means patent_rag only: it never falls back to chunks by itself."""
    if source in ("auto", PATENT_COLLECTION):
        records, info = _from_patent_rag(db_path)
    elif source in ("chunks", CHUNK_COLLECTION):
        records, info = _from_chunks(db_path)
    else:
        raise ValueError(f"Unknown record source {source!r}.")
    if manifest_csv:
        _apply_manifest(records, manifest_csv, info["notes"])
    if limit:
        records = dict(list(sorted(records.items()))[: int(limit)])
        info["notes"].append(f"Limited to the first {len(records)} patents (sorted by id) for a small-sample run.")
    info["usable"] = len(records)
    return records, info


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cmd_inspect(args) -> None:
    for name in (PATENT_COLLECTION, CHUNK_COLLECTION):
        try:
            col = _open_collection(name, args.chroma)
        except Exception as exc:
            print(f"[{name}] unavailable: {exc}")
            continue
        sample = col.get(limit=200, include=["metadatas", "documents"])
        keys = sorted({str(k) for m in sample["metadatas"] for k in (m or {})})
        lens = [len(d or "") for d in sample["documents"]]
        print(f"[{name}] count={col.count()}  metadata keys={keys}")
        if lens:
            print(f"    document length over first {len(lens)}: min={min(lens)} median={int(median(lens))} max={max(lens)} chars")
        for m, d in list(zip(sample["metadatas"], sample["documents"]))[:2]:
            print("    sample metadata:", {k: (str(v)[:60]) for k, v in (m or {}).items()})
            print("    sample document:", (d or "")[:160].replace("\n", " "))


def _cmd_status(args) -> None:
    idx = GraphIndex(get_graph_db_path())
    try:
        print(json.dumps(idx.status(), indent=2))
    finally:
        idx.close()


def _cmd_build(args) -> None:
    records, info = load_records(args.source, args.manifest, args.chroma, args.limit)
    print(f"Source: {info['source']} | patents in source: {info['total_in_source']} | to index: {len(records)}")
    for note in info["notes"]:
        print(" -", note)
    model = load_patent_sbert()
    idx = GraphIndex(get_graph_db_path())
    try:
        if args.rebuild:
            idx.clear()
        summary = build_index(records, idx, model, args.batch_size,
                              progress=lambda d, t, m: print(f"   {m}", flush=True))
        print(json.dumps(summary, indent=2))
    finally:
        idx.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.patent_sbert_embeddings")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="Print the Chroma schema (read-only).")
    p.add_argument("--chroma")
    p.set_defaults(fn=_cmd_inspect)
    p = sub.add_parser("status", help="Show the graph index status.")
    p.set_defaults(fn=_cmd_status)
    p = sub.add_parser("build", help="Build or resume the graph index.")
    p.add_argument("--source", default="auto", choices=["auto", "patent_rag", "chunks"])
    p.add_argument("--manifest")
    p.add_argument("--chroma")
    p.add_argument("--limit", type=int, help="Index only the first N patents (sample run).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--rebuild", action="store_true", help="Empty the graph index first (Chroma is untouched).")
    p.set_defaults(fn=_cmd_build)
    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except (MissingPatentTextError, ModelLoadError, IndexMismatchError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
