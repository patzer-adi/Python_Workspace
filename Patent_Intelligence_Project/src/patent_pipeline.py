#!/usr/bin/env python3
"""
Patent Intelligence Pipeline
----------------------------
One-command, CPU-oriented pipeline for model-specific patent chunking,
embedding generation, and storage in seven independent persistent ChromaDBs.

Run from the project root:
    python -u patent_pipeline.py
Smoke test (isolated paths; does not touch full-run outputs):
    python -u patent_pipeline.py --smoke 10 --model minilm

Important:
- This script never intentionally opens the existing RAG or graph databases.
- It refuses to overwrite an existing model output directory for a full run.
- It writes each model's chunks to JSONL and stores real Sentence Transformer
  embeddings in that model's own persistent ChromaDB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

print('[BOOT] patent_pipeline.py started; importing datasets...', flush=True)
# This warning is emitted by Hugging Face Datasets when an internal format
# transform cannot be fingerprinted. It affects cache reuse, not data correctness.
warnings.filterwarnings(
    'ignore',
    message=r"Parameter 'format_kwargs'.*couldn't be hashed properly.*",
    category=UserWarning,
)
from datasets import DatasetDict, load_from_disk
print('[BOOT] datasets imported; importing Sentence Transformers...', flush=True)
from sentence_transformers import SentenceTransformer
print('[BOOT] Sentence Transformers imported; checking ChromaDB...', flush=True)

try:
    import chromadb
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'chromadb'. Install it in the active virtual "
        "environment with: python -m pip install chromadb"
    ) from exc


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "sample_dataset"
FULL_CHUNK_ROOT = ROOT / "results" / "dynamic_chunks"
FULL_DB_ROOT = ROOT / "data" / "patent_embeddings"
REPORT_ROOT = ROOT / "results"

TEXT_FIELDS = ("title", "abstract", "claims", "background", "summary", "description")
METADATA_FIELDS = (
    "patent_number", "decision", "filing_date", "patent_issue_date",
    "date_published", "cpc_label", "ipc_label",
)

@dataclass(frozen=True)
class ModelConfig:
    key: str
    model_id: str
    max_seq_length: int
    target_tokens: int
    overlap_tokens: int
    document_prefix: str = ""

CONFIGS: dict[str, ModelConfig] = {
    "patentsberta": ModelConfig("patentsberta", "AI-Growth-Lab/PatentSBERTa", 512, 450, 50),
    "paecter": ModelConfig("paecter", "mpi-inno-comp/paecter", 512, 450, 50),
    "nomic": ModelConfig("nomic", "nomic-ai/nomic-embed-text-v1.5", 8192, 4096, 300, "search_document: "),
    "bge": ModelConfig("bge", "BAAI/bge-base-en-v1.5", 512, 450, 50),
    "e5": ModelConfig("e5", "intfloat/e5-base-v2", 512, 450, 50, "passage: "),
    "mpnet": ModelConfig("mpnet", "sentence-transformers/all-mpnet-base-v2", 384, 340, 40),
    "minilm": ModelConfig("minilm", "sentence-transformers/all-MiniLM-L6-v2", 256, 220, 25),
}

# Conservative starting points for a 4-core CPU / 7.56 GiB RAM.
DEFAULT_BATCH = 8
PROGRESS_EVERY = 100
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    """Return text safely; never turn NaN/None into literal words."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    # pandas/numpy NaN-like values without importing pandas.
    try:
        if value != value:
            return ""
    except Exception:
        pass
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def json_safe(value: Any) -> str | int | float | bool | None:
    """Convert metadata to Chroma-compatible scalar values."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    text = clean_text(value)
    return text if text else None


def metadata_for(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in METADATA_FIELDS:
        value = row.get(field)
        safe = json_safe(value)
        if safe is not None:
            out[field] = safe
    return out


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:length]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """
    Conservative paragraph/sentence segmentation returning source character spans.
    It intentionally retains exact source substrings. Abbreviations are handled
    conservatively; final token-budget checks are authoritative regardless.
    """
    spans: list[tuple[int, int]] = []
    # Paragraph boundaries are useful split points; within each paragraph split
    # after likely sentence-ending punctuation followed by whitespace.
    for p_match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.S):
        p_start, p_end = p_match.span()
        paragraph = text[p_start:p_end]
        # Avoid treating common abbreviations and numbered claim markers as ends.
        pattern = re.compile(
            r"(?<=[.!?])(?:[\"')\]]*)\s+(?=[A-Z0-9(])"
        )
        local_starts = [0]
        for m in pattern.finditer(paragraph):
            prefix = paragraph[:m.start()].rstrip()
            if re.search(r"\b(?:e\.g|i\.e|etc|fig|no|vs|approx|inc|u\.s)\.$", prefix, re.I):
                continue
            # Do not split decimal numbers.
            if m.start() > 0 and m.start() < len(paragraph) and paragraph[m.start()-1].isdigit():
                continue
            local_starts.append(m.end())
        local_starts.append(len(paragraph))
        for i in range(len(local_starts) - 1):
            a, b = local_starts[i], local_starts[i + 1]
            while a < b and paragraph[a].isspace():
                a += 1
            while b > a and paragraph[b - 1].isspace():
                b -= 1
            if b > a:
                spans.append((p_start + a, p_start + b))
    if not spans and text.strip():
        a = len(text) - len(text.lstrip())
        b = len(text.rstrip())
        spans.append((a, b))
    return spans


class DynamicChunker:
    def __init__(self, model: SentenceTransformer, cfg: ModelConfig):
        self.model = model
        self.cfg = cfg
        self.tokenizer = model.tokenizer
        # We intentionally tokenize whole source sections before splitting them.
        # Raise the tokenizer's warning threshold so it does not warn about raw
        # unsplit text; the chunker's explicit token-budget checks remain strict.
        self.tokenizer.model_max_length = int(1e9)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                f"{cfg.key}: fast tokenizer with offset mappings is required; "
                "refusing an unsafe character-splitting fallback."
            )
        actual_limit = int(getattr(model, "max_seq_length", cfg.max_seq_length))
        if actual_limit < cfg.target_tokens:
            raise RuntimeError(
                f"{cfg.key}: runtime max_seq_length={actual_limit} is below "
                f"configured target={cfg.target_tokens}."
            )
        self.actual_limit = actual_limit
        # Token budget includes prefix and special tokens because every fit check
        # tokenizes the complete model-ready input.
        self._count_cache: dict[str, int] = {}

    def token_count(self, text: str, add_special_tokens: bool = True) -> int:
        if not text:
            return 0
        # Keep cache bounded; very long runs can otherwise retain too much RAM.
        if len(self._count_cache) > 4096:
            self._count_cache.clear()
        key = ("1" if add_special_tokens else "0") + text
        cached = self._count_cache.get(key)
        if cached is not None:
            return cached
        enc = self.tokenizer(
            text, add_special_tokens=add_special_tokens,
            truncation=False, return_attention_mask=False,
        )
        count = len(enc["input_ids"])
        self._count_cache[key] = count
        return count

    def fits(self, source_text: str) -> bool:
        return self.token_count(self.cfg.document_prefix + source_text, True) <= self.cfg.target_tokens

    def _safe_token_pieces(self, text: str) -> list[str]:
        """Split an oversized sentence using tokenizer offsets and word boundaries."""
        enc = self.tokenizer(
            text, add_special_tokens=False, truncation=False,
            return_offsets_mapping=True, return_attention_mask=False,
        )
        offsets = enc.get("offset_mapping")
        ids = enc.get("input_ids")
        if offsets is None or ids is None:
            raise RuntimeError(f"{self.cfg.key}: tokenizer returned no offset mappings.")
        if not ids:
            return [text] if text.strip() else []

        # Conservative capacity: actual full-input check below catches prefix and
        # special-token overhead. Use token offsets to avoid scanning huge strings.
        prefix_cost = self.token_count(self.cfg.document_prefix, True)
        capacity = max(1, self.cfg.target_tokens - prefix_cost - 4)
        pieces: list[str] = []
        token_start = 0
        n = len(ids)
        while token_start < n:
            token_end = min(n, token_start + capacity)
            char_start = offsets[token_start][0]
            char_end = offsets[token_end - 1][1]
            if char_end <= char_start:
                raise RuntimeError(f"{self.cfg.key}: invalid tokenizer offsets.")
            # Prefer moving the end back to whitespace within this token window.
            if token_end < n:
                window = text[char_start:char_end]
                last_space = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
                # Avoid creating a tiny fragment; only snap if it keeps a useful piece.
                if last_space >= max(1, int(len(window) * 0.55)):
                    char_end = char_start + last_space
                    while token_end > token_start + 1 and offsets[token_end - 2][1] > char_end:
                        token_end -= 1
                    char_end = offsets[token_end - 1][1]
            piece = text[char_start:char_end].strip()
            if not piece:
                # Guaranteed progress through at least one token.
                token_end = min(n, token_start + 1)
                char_start, char_end = offsets[token_start][0], offsets[token_start][1]
                piece = text[char_start:char_end].strip()
            if not piece:
                raise RuntimeError(f"{self.cfg.key}: could not split oversized text safely.")
            # Back off a few tokens until the complete model input fits.
            while not self.fits(piece) and token_end > token_start + 1:
                token_end -= 1
                char_end = offsets[token_end - 1][1]
                piece = text[char_start:char_end].strip()
            if not self.fits(piece):
                raise RuntimeError(
                    f"{self.cfg.key}: even a single tokenizer piece exceeds target "
                    f"{self.cfg.target_tokens}; cannot safely chunk this text."
                )
            pieces.append(piece)
            if token_end >= n:
                break
            # Overlap by approximate token count, then snap to a whitespace boundary.
            next_token = max(token_start + 1, token_end - self.cfg.overlap_tokens)
            next_char = offsets[next_token][0]
            while next_char < len(text) and not text[next_char].isspace() and next_token < token_end - 1:
                next_token += 1
                next_char = offsets[next_token][0]
            token_start = next_token
        return pieces

    def split_section(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        units: list[str] = []
        for start, end in sentence_spans(text):
            unit = text[start:end].strip()
            if not unit:
                continue
            if self.fits(unit):
                units.append(unit)
            else:
                units.extend(self._safe_token_pieces(unit))

        # Greedy packing. On overflow, flush the current chunk and carry whole
        # trailing units forward where possible. Always consume at least one unit.
        chunks: list[str] = []
        current: list[str] = []
        i = 0
        while i < len(units):
            candidate = " ".join(current + [units[i]]) if current else units[i]
            if self.fits(candidate):
                current.append(units[i])
                i += 1
                continue
            if current:
                chunks.append(" ".join(current).strip())
                # Whole-unit overlap; remove units until within overlap budget.
                tail: list[str] = []
                for unit in reversed(current):
                    proposal = " ".join([unit] + tail).strip()
                    if self.token_count(proposal, False) > self.cfg.overlap_tokens:
                        break
                    tail = [unit] + tail
                current = tail
                # Ensure forward progress if overlap prevents adding next unit.
                if current and not self.fits(" ".join(current + [units[i]])):
                    current = []
                continue
            # An individual unit should already have been split; if not, fail loudly.
            pieces = self._safe_token_pieces(units[i])
            if len(pieces) == 1 and not self.fits(pieces[0]):
                raise RuntimeError(f"{self.cfg.key}: unable to fit one chunk within target.")
            units[i:i + 1] = pieces
        if current:
            final = " ".join(current).strip()
            if not self.fits(final):
                raise RuntimeError(f"{self.cfg.key}: final chunk exceeds target.")
            chunks.append(final)
        # Final authoritative check.
        for j, chunk in enumerate(chunks):
            count = self.token_count(self.cfg.document_prefix + chunk, True)
            if count > self.cfg.target_tokens:
                raise RuntimeError(
                    f"{self.cfg.key}: chunk {j} has {count} tokens, target "
                    f"{self.cfg.target_tokens}."
                )
        return chunks


def source_rows(dataset: DatasetDict, smoke_n: int | None) -> Iterator[tuple[str, int, dict[str, Any]]]:
    remaining = smoke_n
    for split_name in ("train", "validation"):
        if split_name not in dataset:
            continue
        split = dataset[split_name]
        for idx in range(len(split)):
            if remaining is not None and remaining <= 0:
                return
            yield split_name, idx, dict(split[idx])
            if remaining is not None:
                remaining -= 1


def make_chunk_records(
    chunker: DynamicChunker, cfg: ModelConfig, split_name: str,
    row_index: int, row: dict[str, Any],
) -> list[dict[str, Any]]:
    metadata = metadata_for(row)
    patent_number = clean_text(row.get("patent_number"))
    patent_identity = patent_number or f"missing-{split_name}-{row_index}"
    records: list[dict[str, Any]] = []
    section_counter = 0
    for field in TEXT_FIELDS:
        source = clean_text(row.get(field))
        if not source:
            continue
        chunks = chunker.split_section(source)
        if not chunks:
            raise RuntimeError(
                f"non-empty source section produced no chunks: split={split_name}, "
                f"row={row_index}, patent={patent_identity}, section={field}"
            )
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = (
                f"{cfg.key}:{split_name}:{row_index}:{field}:{chunk_index}:"
                f"{short_hash(patent_identity, 8)}"
            )
            model_input = cfg.document_prefix + chunk_text
            count = chunker.token_count(model_input, True)
            if count > cfg.target_tokens:
                raise RuntimeError(
                    f"budget violation for {chunk_id}: {count} > {cfg.target_tokens}"
                )
            records.append({
                "chunk_id": chunk_id,
                "model_key": cfg.key,
                "model_id": cfg.model_id,
                "patent_number": patent_number or None,
                "section": field,
                "chunk_index": chunk_index,
                "source_split": split_name,
                "source_row_index": row_index,
                "source_text": chunk_text,
                "model_input_text": model_input,
                "token_count": count,
                "metadata": metadata,
            })
        section_counter += 1
    return records


def write_jsonl_line(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Checkpoint is unreadable: {path}: {exc}") from exc


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def chroma_metadata(record: dict[str, Any]) -> dict[str, Any]:
    md: dict[str, Any] = {
        "model_key": record["model_key"],
        "model_id": record["model_id"],
        "section": record["section"],
        "chunk_index": int(record["chunk_index"]),
        "source_split": record["source_split"],
        "source_row_index": int(record["source_row_index"]),
        "token_count": int(record["token_count"]),
        "document_prefix": CONFIGS[record["model_key"]].document_prefix,
    }
    if record.get("patent_number"):
        md["patent_number"] = str(record["patent_number"])
    for k, v in record.get("metadata", {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)) and not (isinstance(v, float) and not math.isfinite(v)):
            md[k] = v
        else:
            md[k] = str(v)
    return md


def run_one_model(
    cfg: ModelConfig, dataset: DatasetDict, batch_size: int,
    smoke_n: int | None, output_root: Path, db_root: Path,
) -> dict[str, Any]:
    model_chunk_dir = output_root / cfg.key
    model_db_dir = db_root / cfg.key
    model_chunk_dir.mkdir(parents=True, exist_ok=True)
    model_db_dir.parent.mkdir(parents=True, exist_ok=True)
    chunk_path = model_chunk_dir / "chunks.jsonl"
    checkpoint_path = model_chunk_dir / "checkpoint.json"
    manifest_path = model_chunk_dir / "manifest.json"
    log_path = model_chunk_dir / "run.log"

    # Smoke-test output roots are unique and isolated. For full runs, never
    # overwrite a prior completed output; resume only when a checkpoint exists.
    checkpoint = read_checkpoint(checkpoint_path)
    if chunk_path.exists() and checkpoint is None:
        raise RuntimeError(
            f"{chunk_path} already exists without a checkpoint. Refusing to overwrite it. "
            "Move it aside manually after inspecting it, or choose a new output root."
        )
    if manifest_path.exists() and checkpoint is None:
        raise RuntimeError(f"{manifest_path} exists without checkpoint; refusing overwrite.")

    print(f"\n{'=' * 72}\nMODEL {cfg.key}: {cfg.model_id}", flush=True)
    print(f"Target={cfg.target_tokens}, overlap={cfg.overlap_tokens}, "
          f"configured model limit={cfg.max_seq_length}", flush=True)
    start_time = time.monotonic()

    print(f"[{cfg.key}] Loading embedding model on CPU: {cfg.model_id}", flush=True)
    model = SentenceTransformer(cfg.model_id, device="cpu")
    print(f"[{cfg.key}] Embedding model loaded; max_seq_length={model.max_seq_length}", flush=True)
    actual_limit = int(model.max_seq_length)
    if actual_limit < cfg.target_tokens:
        raise RuntimeError(
            f"{cfg.key}: loaded model max_seq_length={actual_limit}, below "
            f"target={cfg.target_tokens}."
        )
    chunker = DynamicChunker(model, cfg)
    client = chromadb.PersistentClient(path=str(model_db_dir))
    collection = client.get_or_create_collection(
        name=f"patent_chunks_{cfg.key}",
        metadata={"model_key": cfg.key, "model_id": cfg.model_id, "hnsw:space": "cosine"},
    )

    start_split = "train"
    start_row = 0
    chunks_written = 0
    patents_processed = 0
    vectors_inserted = int(collection.count())
    errors: list[str] = []

    # Resume strategy: checkpoint only advances after a source patent has been
    # written and all its vectors upserted. If a crash occurs mid-patent, IDs are
    # deterministic and upsert makes retry safe; JSONL may contain duplicate
    # lines, so on resume a fresh canonical file is rebuilt below only if the
    # checkpoint indicates an incomplete run. To avoid destructive rewrites,
    # this implementation instead uses a unique run JSONL and refuses ambiguous
    # partial state.
    if checkpoint:
        if checkpoint.get("model_id") != cfg.model_id or checkpoint.get("config") != asdict(cfg):
            raise RuntimeError(f"{cfg.key}: checkpoint config mismatch; refusing unsafe resume.")
        if checkpoint.get("status") == "completed":
            if not chunk_path.exists():
                raise RuntimeError(f"{cfg.key}: checkpoint says complete but chunks file is missing.")
            expected = int(checkpoint.get("vectors_inserted", -1))
            actual = int(collection.count())
            if expected != actual:
                raise RuntimeError(
                    f"{cfg.key}: completed checkpoint says {expected} vectors, DB has {actual}."
                )
            print(f"{cfg.key}: already completed and verified; skipping.", flush=True)
            return checkpoint
        # The current implementation safely refuses ambiguous partial JSONL state.
        # Stable upsert IDs make DB inserts idempotent, but reconciling an interrupted
        # append-only JSONL file requires an explicit operator decision.
        raise RuntimeError(
            f"{cfg.key}: incomplete checkpoint found. Database writes are idempotent, "
            "but partial JSONL recovery is ambiguous. Preserve the files and inspect "
            "the checkpoint before choosing a recovery path."
        )

    # Refuse an existing non-empty collection without a checkpoint; never mix runs.
    if vectors_inserted:
        raise RuntimeError(
            f"{cfg.key}: database collection already contains {vectors_inserted} records "
            "but no compatible checkpoint exists. Refusing to mix data."
        )

    total_rows = sum(len(dataset[s]) for s in ("train", "validation") if s in dataset)
    if smoke_n is not None:
        total_rows = min(total_rows, smoke_n)

    checkpoint_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_key": cfg.key,
        "model_id": cfg.model_id,
        "config": asdict(cfg),
        "status": "running",
        "started_at": utc_now(),
        "source_rows_expected": total_rows,
        "patents_processed": 0,
        "chunks_written": 0,
        "vectors_inserted": 0,
        "last_completed_split": None,
        "last_completed_row": None,
    }
    # Exclusive creation ensures an existing file is never truncated.
    with checkpoint_path.open("x", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # Output file is exclusive-created; never truncates an old partial file.
    with chunk_path.open("x", encoding="utf-8") as chunk_file, log_path.open("x", encoding="utf-8") as log_file:
        batch_records: list[dict[str, Any]] = []
        last_report = time.monotonic()
        for split_name, row_index, row in source_rows(dataset, smoke_n):
            patent_label = clean_text(row.get("patent_number")) or f"missing-{split_name}-{row_index}"
            print(f"[{cfg.key}] Starting patent {patents_processed + 1}/{total_rows} "
                  f"({split_name} row={row_index}, patent={patent_label})", flush=True)
            try:
                records = make_chunk_records(chunker, cfg, split_name, row_index, row)
                print(f"[{cfg.key}] Chunking finished for patent {patent_label}: "
                      f"{len(records)} chunks; starting embedding/storage...", flush=True)
                # JSONL is written as records are processed; flush before checkpoint.
                for record in records:
                    write_jsonl_line(chunk_file, record)
                chunk_file.flush()
                os.fsync(chunk_file.fileno())

                # Batch records across patents, keeping memory bounded.
                for record in records:
                    batch_records.append(record)
                    if len(batch_records) >= batch_size:
                        persist_batch(model, collection, batch_records)
                        vectors_inserted += len(batch_records)
                        batch_records.clear()
                # Flush any remaining batch for this patent before checkpointing,
                # so the checkpoint never advances ahead of persisted vectors.
                if batch_records:
                    persist_batch(model, collection, batch_records)
                    vectors_inserted += len(batch_records)
                    batch_records.clear()

                patents_processed += 1
                chunks_written += len(records)
                checkpoint_data.update({
                    "patents_processed": patents_processed,
                    "chunks_written": chunks_written,
                    "vectors_inserted": vectors_inserted,
                    "last_completed_split": split_name,
                    "last_completed_row": row_index,
                    "updated_at": utc_now(),
                })
                # Atomic checkpoint update: create temp only if absent, then replace.
                tmp = checkpoint_path.with_suffix(".json.tmp")
                if tmp.exists():
                    raise RuntimeError(f"stale checkpoint temp exists: {tmp}; refusing overwrite.")
                with tmp.open("x", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, checkpoint_path)

            except Exception as exc:
                message = (
                    f"model={cfg.key}, split={split_name}, row={row_index}, "
                    f"patent={patent_label}: {type(exc).__name__}: {exc}"
                )
                errors.append(message)
                print(f"\nERROR: {message}", file=sys.stderr, flush=True)
                log_file.write(message + "\n")
                log_file.flush()
                # Do not mark complete. Stop this model to avoid hidden data loss.
                raise

            if patents_processed == 1 or patents_processed % PROGRESS_EVERY == 0:
                elapsed = time.monotonic() - start_time
                pct = 100.0 * patents_processed / max(1, total_rows)
                print(
                    f"[{cfg.key}] {split_name}: {patents_processed}/{total_rows} "
                    f"patents ({pct:.1f}%), chunks={chunks_written}, "
                    f"vectors={collection.count()}, elapsed={elapsed/60:.1f} min",
                    flush=True,
                )
                log_file.write(
                    f"{utc_now()} patents={patents_processed}/{total_rows} "
                    f"chunks={chunks_written} vectors={collection.count()}\n"
                )
                log_file.flush()

        # If no records were created for empty patents, nothing needs embedding.
        # Final validation below verifies collection count against JSONL.
        actual_count = int(collection.count())
        if actual_count != chunks_written:
            raise RuntimeError(
                f"{cfg.key}: expected {chunks_written} vectors from this run, "
                f"but ChromaDB reports {actual_count}."
            )

        stats = validate_jsonl_and_db(chunk_path, collection, cfg)
        checkpoint_data.update({
            "status": "completed",
            "completed_at": utc_now(),
            "patents_processed": patents_processed,
            "chunks_written": chunks_written,
            "vectors_inserted": actual_count,
            "embedding_dimension": model.get_embedding_dimension(),
            "validation": stats,
            "elapsed_seconds": round(time.monotonic() - start_time, 2),
        })
        tmp = checkpoint_path.with_suffix(".json.tmp")
        if tmp.exists():
            raise RuntimeError(f"stale checkpoint temp exists: {tmp}; refusing overwrite.")
        with tmp.open("x", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, checkpoint_path)
        with manifest_path.open("x", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        print(
            f"COMPLETED {cfg.key}: patents={patents_processed}, chunks={chunks_written}, "
            f"vectors={actual_count}, dim={model.get_embedding_dimension()}, "
            f"elapsed={(time.monotonic()-start_time)/60:.1f} min",
            flush=True,
        )
    del chunker, model, client, collection
    return checkpoint_data


def persist_batch(model: SentenceTransformer, collection, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    print(f"[EMBED] Generating embeddings for {len(records)} chunks...", flush=True)
    texts = [r["model_input_text"] for r in records]
    ids = [r["chunk_id"] for r in records]
    metadatas = [chroma_metadata(r) for r in records]
    vectors = model.encode(
        texts,
        batch_size=min(DEFAULT_BATCH, len(texts)),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
    )
    vectors = vectors.tolist()
    if len(vectors) != len(records):
        raise RuntimeError(f"embedding batch mismatch: {len(vectors)} vectors for {len(records)} chunks")
    dim = len(vectors[0]) if vectors else 0
    for i, vec in enumerate(vectors):
        if len(vec) != dim or not vec or any(not math.isfinite(float(x)) for x in vec):
            raise RuntimeError(f"invalid embedding vector at batch position {i}")
    # Chroma upsert makes deterministic-ID retries idempotent.
    collection.upsert(ids=ids, embeddings=vectors, documents=[r["source_text"] for r in records], metadatas=metadatas)
    print(f"[CHROMA] Persisted {len(records)} vectors; collection now has {collection.count()} records.", flush=True)


def validate_jsonl_and_db(path: Path, collection, cfg: ModelConfig) -> dict[str, Any]:
    count = 0
    ids_seen: set[str] = set()
    min_tokens: int | None = None
    max_tokens = 0
    token_sum = 0
    # IDs are held in memory for uniqueness checking; only IDs, not vectors/text.
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            required = ("chunk_id", "section", "source_text", "model_input_text", "token_count")
            if any(k not in rec or rec[k] in (None, "") for k in required):
                raise RuntimeError(f"missing required chunk field at {path}:{line_no}")
            if rec["chunk_id"] in ids_seen:
                raise RuntimeError(f"duplicate chunk ID: {rec['chunk_id']}")
            ids_seen.add(rec["chunk_id"])
            n = int(rec["token_count"])
            if n > cfg.target_tokens:
                raise RuntimeError(f"budget violation at {path}:{line_no}: {n} > {cfg.target_tokens}")
            min_tokens = n if min_tokens is None else min(min_tokens, n)
            max_tokens = max(max_tokens, n)
            token_sum += n
            count += 1
    db_count = int(collection.count())
    if db_count != count:
        raise RuntimeError(f"JSONL count={count} but ChromaDB count={db_count}")
    # Spot-check first record, if any.
    if count:
        sample = collection.get(limit=1, include=["documents", "metadatas", "embeddings"])
        if not sample["ids"] or not sample["documents"] or sample["embeddings"] is None:
            raise RuntimeError(f"{cfg.key}: ChromaDB sample verification failed")
        emb = sample["embeddings"]
        if len(emb) != 1 or not len(emb[0]):
            raise RuntimeError(f"{cfg.key}: invalid stored embedding sample")
    return {
        "jsonl_records": count,
        "chroma_records": db_count,
        "unique_ids": len(ids_seen),
        "target_token_budget": cfg.target_tokens,
        "min_tokens": min_tokens,
        "mean_tokens": round(token_sum / count, 2) if count else 0,
        "max_tokens": max_tokens,
        "budget_violations": 0,
        "sample_database_readback": bool(count),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk, embed, and index patent records.")
    parser.add_argument("--smoke", type=int, default=None,
                        help="Process only N source patents using an isolated smoke-test run (MiniLM by default).")
    parser.add_argument("--subset", type=int, default=None,
                        help="Process the first N source patents for ALL selected models, using a new isolated run directory.")
    parser.add_argument("--model", choices=tuple(CONFIGS), default=None,
                        help="Select one model. Without --smoke/--subset, runs all models on the full dataset.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help=f"Embedding batch size (default {DEFAULT_BATCH}).")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.smoke is not None and args.smoke < 1:
        parser.error("--smoke must be >= 1")
    if args.subset is not None and args.subset < 1:
        parser.error("--subset must be >= 1")
    if args.smoke is not None and args.subset is not None:
        parser.error("Use either --smoke or --subset, not both.")
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}", file=sys.stderr)
        return 2

    if args.smoke is not None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = ROOT / "results" / "patent_pipeline_smoke" / run_id
        output_root = run_root / "chunks"
        db_root = ROOT / "data" / "patent_embeddings_smoke" / run_id
        selected = [CONFIGS[args.model]] if args.model else [CONFIGS["minilm"]]
        report_json = run_root / "summary.json"
        source_limit = args.smoke
    elif args.subset is not None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = ROOT / "results" / "patent_subset_runs" / run_id
        output_root = run_root / "chunks"
        db_root = ROOT / "data" / "patent_subset_runs" / run_id / "chroma"
        selected = [CONFIGS[args.model]] if args.model else list(CONFIGS.values())
        report_json = run_root / "summary.json"
        source_limit = args.subset
    else:
        output_root = FULL_CHUNK_ROOT
        db_root = FULL_DB_ROOT
        selected = [CONFIGS[args.model]] if args.model else list(CONFIGS.values())
        report_json = REPORT_ROOT / "patent_pipeline_summary.json"
        source_limit = None

    output_root.mkdir(parents=True, exist_ok=True)
    db_root.mkdir(parents=True, exist_ok=True)
    print("Patent Intelligence Pipeline", flush=True)
    print(f"Dataset: {DATASET_PATH}", flush=True)
    print(f"Chunk output: {output_root}", flush=True)
    print(f"ChromaDB root: {db_root}", flush=True)
    print(f"Models selected: {', '.join(c.key for c in selected)}", flush=True)
    print(f"Source patent limit: {source_limit if source_limit is not None else 'FULL DATASET'}", flush=True)
    print("This pipeline writes new outputs only; existing RAG/graph DBs are not targets.", flush=True)

    print("[DATASET] Loading local dataset from disk; this may take a little while...", flush=True)
    dataset = load_from_disk(str(DATASET_PATH))
    print("[DATASET] Dataset loaded successfully.", flush=True)
    if not isinstance(dataset, DatasetDict):
        print("Expected a Hugging Face DatasetDict with train/validation splits.", file=sys.stderr)
        return 2
    print("Dataset splits: " + ", ".join(f"{k}={len(v)}" for k, v in dataset.items()), flush=True)

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for cfg in selected:
        try:
            summary = run_one_model(cfg, dataset, args.batch_size, source_limit, output_root, db_root)
            summaries.append(summary)
        except Exception as exc:
            failures.append({"model": cfg.key, "error": f"{type(exc).__name__}: {exc}"})
            print(f"\nMODEL FAILED: {cfg.key}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            # Continue with other independent models, preserving failed artifacts.
            continue

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset_path": str(DATASET_PATH),
        "dataset_splits": {k: len(v) for k, v in dataset.items()},
        "smoke_limit": args.smoke,
        "subset_limit": args.subset,
        "source_limit": source_limit,
        "models_requested": [c.key for c in selected],
        "models_completed": [s.get("model_key") for s in summaries],
        "failures": failures,
        "success": len(failures) == 0 and len(summaries) == len(selected),
        "summaries": summaries,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous summary. Timestamped fallback for reports.
    target_report = report_json
    if target_report.exists():
        target_report = report_json.with_name(
            f"{report_json.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{report_json.suffix}"
        )
    with target_report.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md_path = target_report.with_suffix(".md")
    with md_path.open("x", encoding="utf-8") as f:
        f.write("# Patent Intelligence Pipeline Summary\n\n")
        f.write(f"- Created: {report['created_at']}\n")
        f.write(f"- Success: {report['success']}\n")
        f.write(f"- Dataset: `{DATASET_PATH}`\n\n")
        f.write("| Model | Status | Patents | Chunks | Vectors | Dimension | Seconds |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for s in summaries:
            f.write(
                f"| {s.get('model_key')} | {s.get('status')} | "
                f"{s.get('patents_processed')} | {s.get('chunks_written')} | "
                f"{s.get('vectors_inserted')} | {s.get('embedding_dimension')} | "
                f"{s.get('elapsed_seconds')} |\n"
            )
        if failures:
            f.write("\n## Failures\n")
            for failure in failures:
                f.write(f"- **{failure['model']}**: {failure['error']}\n")
    print(f"\nSummary JSON: {target_report}", flush=True)
    print(f"Summary Markdown: {md_path}", flush=True)
    print(f"Overall success: {report['success']}", flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
