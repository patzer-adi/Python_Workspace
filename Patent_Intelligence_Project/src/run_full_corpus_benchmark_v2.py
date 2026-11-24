#!/usr/bin/env python3
"""
Patent Embedding Benchmark V2

Compares seven existing embedding models plus three locally fine-tuned BGE models
on the same local patent corpus.

Important:
- This is corpus-wide patent-to-patent nearest-neighbour evaluation.
- CPC subclass agreement (first four CPC characters) is a weak proxy, not
  human relevance or proof of prior-art retrieval quality.
- The benchmark uses the same text, corpus order, sequence-length cap and
  cosine-neighbour procedure for every model.
- Models are loaded and evaluated sequentially to limit RAM use.
- Outputs are written only beneath the selected output directory.

Run from the project root:
  source .venv_finetune/bin/activate
  python src/run_full_corpus_benchmark_v2.py --limit 3000 --expected-rows 25247

Full run:
  python src/run_full_corpus_benchmark_v2.py --expected-rows 25247

The script defaults to 256 tokens to match the fine-tuning configuration.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "data" / "sample_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "full_corpus_benchmark_v2"
LOCAL_FINE_TUNED_DIR = PROJECT_DIR / "models" / "fine_tuned_bge"

# Prefixes follow the existing V1 benchmark choices for the symmetric
# patent-to-patent comparison. Fine-tuned BGE models use the same raw
# Title + Abstract format used during fine-tuning.
MODELS: list[dict[str, Any]] = [
    {
        "name": "PatentSBERTa",
        "model_id": "AI-Growth-Lab/PatentSBERTa",
        "local_path": None,
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "PaECTER",
        "model_id": "mpi-inno-comp/paecter",
        "local_path": None,
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "Nomic-Embed-v1.5",
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "local_path": None,
        "prefix": "clustering: ",
        "trust_remote_code": True,
    },
    {
        "name": "BGE-base-en-v1.5",
        "model_id": "BAAI/bge-base-en-v1.5",
        "local_path": None,
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "E5-base-v2",
        "model_id": "intfloat/e5-base-v2",
        "local_path": None,
        "prefix": "query: ",
        "trust_remote_code": False,
    },
    {
        "name": "all-mpnet-base-v2",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "local_path": None,
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "all-MiniLM-L6-v2",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "local_path": None,
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "BGE-Patent-MNRL",
        "model_id": "local:bge_patent_mnrl",
        "local_path": LOCAL_FINE_TUNED_DIR / "bge_patent_mnrl",
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "BGE-Similarity-Contrastive",
        "model_id": "local:bge_similarity_contrastive",
        "local_path": LOCAL_FINE_TUNED_DIR / "bge_similarity_contrastive",
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "BGE-Hybrid-Contrastive",
        "model_id": "local:bge_hybrid_contrastive",
        "local_path": LOCAL_FINE_TUNED_DIR / "bge_hybrid_contrastive",
        "prefix": "",
        "trust_remote_code": False,
    },
]


def write_json(path: Path, data: dict[str, Any]) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, Path):
            return str(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def log(message: str, log_path: Path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def safe_name(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def load_rows(dataset_dir: Path) -> tuple[pd.DataFrame, str | None]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    saved = load_from_disk(str(dataset_dir))
    if not hasattr(saved, "items"):
        raise ValueError(
            f"Expected a DatasetDict at {dataset_dir}, got {type(saved).__name__}"
        )

    frames = []
    for split_name, split in saved.items():
        available = set(split.column_names)
        cols = [
            column
            for column in ["patent_number", "title", "abstract", "cpc_label"]
            if column in available
        ]
        frame = split.select_columns(cols).to_pandas()
        frame["source_split"] = split_name
        frames.append(frame)

    if not frames:
        raise ValueError("DatasetDict has no splits.")

    df = pd.concat(frames, ignore_index=True)
    if "title" not in df.columns or "abstract" not in df.columns:
        raise ValueError("Dataset must contain both 'title' and 'abstract'.")

    if "patent_number" not in df.columns:
        df["patent_number"] = [f"row_{i:06d}" for i in range(len(df))]

    df["patent_number"] = df["patent_number"].fillna("").astype(str).str.strip()
    empty_ids = df["patent_number"].eq("")
    if empty_ids.any():
        df.loc[empty_ids, "patent_number"] = [
            f"row_{i:06d}" for i in df.index[empty_ids]
        ]

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["abstract"] = df["abstract"].fillna("").astype(str).str.strip()

    cpc_col = "cpc_label" if "cpc_label" in df.columns else None
    if cpc_col:
        df["cpc_label_clean"] = df[cpc_col].fillna("").astype(str).str.strip()
    else:
        df["cpc_label_clean"] = ""

    df["cpc_subclass"] = df["cpc_label_clean"].str.slice(0, 4)
    df["text"] = (df["title"] + "\n\n" + df["abstract"]).str.strip()
    df.loc[df["text"].eq(""), "text"] = "[No title or abstract available]"

    if df["patent_number"].duplicated().any():
        duplicates = int(df["patent_number"].duplicated().sum())
        raise ValueError(
            f"Found {duplicates} duplicate patent_number values; inspect source first."
        )
    return df, cpc_col


def dataset_fingerprint(ids: list[str], texts: list[str], labels: list[str]) -> str:
    digest = hashlib.sha256()
    for group in (ids, texts, labels):
        digest.update(b"\n--FIELD--\n")
        for value in group:
            digest.update(value.encode("utf-8", errors="replace"))
            digest.update(b"\0")
    return digest.hexdigest()


def random_cpc_baseline(labels: list[str]) -> float:
    valid = [label[:4] for label in labels if label]
    if not valid:
        return float("nan")
    frequencies = pd.Series(valid).value_counts(normalize=True)
    return float((frequencies**2).sum())


def calculate_cpc_metrics(
    neighbor_indices: np.ndarray,
    labels: list[str],
    top_k: int,
) -> dict[str, float | int]:
    """
    CPC-proxy metrics over all queries with a non-empty CPC subclass.

    Relevant means same first-four-character CPC subclass. The query itself
    is excluded from the candidate corpus. Precision uses all returned
    neighbours as its denominator, including neighbours with missing CPC.
    Recall uses all other corpus records in the query's CPC subclass as the
    relevant set. This is a broad, weak proxy and should not be interpreted
    as human relevance.
    """
    n = len(labels)
    subclass = [label[:4] if label else "" for label in labels]
    class_counts: dict[str, int] = {}
    for label in subclass:
        if label:
            class_counts[label] = class_counts.get(label, 0) + 1

    hits: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for query_i in range(n):
        query_label = subclass[query_i]
        if not query_label:
            continue

        retrieved = neighbor_indices[query_i, :top_k]
        relevance = [
            1 if subclass[int(index)] == query_label else 0
            for index in retrieved
        ]
        relevant_total = max(class_counts.get(query_label, 1) - 1, 0)

        hits.append(float(any(relevance)))
        precisions.append(float(sum(relevance) / max(len(relevance), 1)))
        recalls.append(
            float(sum(relevance) / relevant_total) if relevant_total else 0.0
        )

        first_relevant = next(
            (rank for rank, rel in enumerate(relevance, start=1) if rel), None
        )
        reciprocal_ranks.append(
            1.0 / first_relevant if first_relevant is not None else 0.0
        )

        dcg = sum(
            rel / np.log2(rank + 1)
            for rank, rel in enumerate(relevance, start=1)
        )
        ideal_count = min(relevant_total, top_k)
        idcg = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        ndcgs.append(float(dcg / idcg) if idcg > 0 else 0.0)

    baseline = random_cpc_baseline(labels)
    mean_precision = float(np.mean(precisions)) if precisions else float("nan")
    return {
        "labelled_query_count": len(hits),
        "same_cpc_hit_at_k": float(np.mean(hits)) if hits else float("nan"),
        "same_cpc_precision_at_k": mean_precision,
        "same_cpc_recall_at_k": float(np.mean(recalls)) if recalls else float("nan"),
        "same_cpc_mrr_at_k": (
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
        ),
        "same_cpc_ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "random_cpc_match_baseline": baseline,
        "precision_lift_over_random": (
            mean_precision / baseline
            if baseline > 0 and np.isfinite(mean_precision)
            else float("nan")
        ),
    }


def exact_cosine_topk(
    emb_path: Path,
    n: int,
    dim: int,
    top_k: int,
    block_size: int,
    log_path: Path,
    model_key: str,
) -> np.ndarray:
    """
    Compute exact top-k indices using blockwise matrix multiplication.
    Only a block_size x n score matrix is held in memory.
    """
    embeddings = np.memmap(
        emb_path, dtype=np.float32, mode="r", shape=(n, dim)
    )
    k = min(top_k, n - 1)
    neighbors_path = emb_path.parent / f"top{k}_indices.npy"
    temporary_path = emb_path.parent / f"top{k}_indices.tmp.npy"
    neighbor_indices = np.lib.format.open_memmap(
        temporary_path, mode="w+", dtype=np.int32, shape=(n, k)
    )

    total_blocks = (n + block_size - 1) // block_size
    for block_no, start in enumerate(range(0, n, block_size), start=1):
        end = min(start + block_size, n)
        scores = np.asarray(embeddings[start:end]) @ embeddings.T

        for local_i, score_row in enumerate(scores):
            query_i = start + local_i
            score_row[query_i] = -np.inf
            candidate_idx = np.argpartition(score_row, -k)[-k:]
            candidate_idx = candidate_idx[
                np.argsort(score_row[candidate_idx])[::-1]
            ]
            neighbor_indices[query_i] = candidate_idx.astype(np.int32)

        del scores
        if block_no % 10 == 0 or block_no == total_blocks:
            log(
                f"  [{model_key}] neighbours: {end:,}/{n:,} "
                f"queries ({block_no}/{total_blocks} blocks)",
                log_path,
            )

    neighbor_indices.flush()
    del neighbor_indices
    del embeddings
    gc.collect()

    temporary_path.replace(neighbors_path)
    return np.load(neighbors_path, mmap_mode="r")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable patent embedding benchmark V2"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--neighbor-block-size", type=int, default=32)
    parser.add_argument("--expected-rows", type=int, default=25247)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional exact model names to run; default is all ten.",
    )
    args = parser.parse_args()

    if min(
        args.batch_size,
        args.chunk_size,
        args.max_seq_length,
        args.top_k,
        args.neighbor_block_size,
    ) < 1:
        parser.error("batch-size, chunk-size, max-seq-length, top-k and block-size must be positive")
    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer")

    selected_models = MODELS
    if args.models:
        requested = set(args.models)
        selected_models = [m for m in MODELS if m["name"] in requested]
        unknown = requested - {m["name"] for m in selected_models}
        if unknown:
            parser.error(f"Unknown model name(s): {', '.join(sorted(unknown))}")

    dataset_dir = args.dataset.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "benchmark.log"

    log("=" * 76, log_path)
    log("PATENT EMBEDDING BENCHMARK V2", log_path)
    log(f"Dataset: {dataset_dir}", log_path)
    log(f"Output: {output_dir}", log_path)
    log(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}", log_path)
    log("Device: CPU", log_path)
    log(f"Max sequence length: {args.max_seq_length}", log_path)

    df, cpc_col = load_rows(dataset_dir)
    source_rows = len(df)
    if args.expected_rows and source_rows != args.expected_rows:
        raise SystemExit(
            f"SAFETY STOP: source has {source_rows:,} rows; expected "
            f"{args.expected_rows:,}. Verify the dataset or pass --expected-rows 0."
        )

    if args.limit:
        if args.limit > source_rows:
            raise SystemExit(
                f"--limit {args.limit:,} exceeds source row count {source_rows:,}."
            )
        df = (
            df.sample(n=args.limit, random_state=0, replace=False)
            .sort_index()
            .reset_index(drop=True)
        )

    n = len(df)
    if n < 2:
        raise SystemExit("At least two patents are required.")
    ids = df["patent_number"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()
    labels = df["cpc_label_clean"].astype(str).tolist()
    titles = df["title"].astype(str).tolist()
    abstracts = df["abstract"].astype(str).tolist()
    fingerprint = dataset_fingerprint(ids, texts, labels)

    manifest_columns = [
        "patent_number", "source_split", "title", "abstract",
        "cpc_label_clean", "cpc_subclass",
    ]
    df[manifest_columns].to_csv(output_dir / "dataset_manifest.csv", index=False)

    run_manifest = {
        "benchmark_version": 2,
        "dataset_path": str(dataset_dir),
        "source_rows": source_rows,
        "rows_used": n,
        "limit": args.limit,
        "unique_patent_numbers": int(df["patent_number"].nunique()),
        "dataset_fingerprint_sha256": fingerprint,
        "text_fields": ["title", "abstract"],
        "cpc_source_column": cpc_col,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "max_seq_length": args.max_seq_length,
        "top_k": args.top_k,
        "neighbor_block_size": args.neighbor_block_size,
        "device": "cpu",
        "evaluation": "all-corpus patent-to-patent nearest neighbours; CPC subclass proxy",
        "models": [
            {
                "name": model["name"],
                "model_id": model["model_id"],
                "local_path": str(model["local_path"]) if model["local_path"] else None,
                "prefix": model["prefix"],
            }
            for model in selected_models
        ],
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    log(
        f"Source rows: {source_rows:,}; rows used: {n:,}; "
        f"unique IDs: {df['patent_number'].nunique():,}",
        log_path,
    )
    log(f"CPC column: {cpc_col or 'unavailable'}; fingerprint: {fingerprint}", log_path)

    summary_path = output_dir / "model_comparison.csv"
    if summary_path.exists():
        try:
            old_rows = pd.read_csv(summary_path).to_dict("records")
        except Exception:
            old_rows = []
    else:
        old_rows = []
    results = {str(row.get("model")): row for row in old_rows if row.get("model")}

    for config in selected_models:
        name = config["name"]
        key = safe_name(name)
        model_dir = output_dir / "embeddings" / key
        model_dir.mkdir(parents=True, exist_ok=True)
        emb_path = model_dir / "embeddings.f32"
        state_path = output_dir / "state" / f"{key}.json"
        metrics_path = output_dir / "metrics" / f"{key}.json"

        model_identifier = config["model_id"]
        local_path = config["local_path"]
        load_source = str(local_path) if local_path else model_identifier

        run_key_payload = {
            "dataset_fingerprint": fingerprint,
            "model_id": model_identifier,
            "prefix": config["prefix"],
            "max_seq_length": args.max_seq_length,
            "text_fields": ["title", "abstract"],
            "normalize_embeddings": True,
            "dtype": "float32",
            "rows": n,
        }
        run_key = hashlib.sha256(
            json.dumps(run_key_payload, sort_keys=True).encode()
        ).hexdigest()

        previous = results.get(name, {})
        neighbors_path = model_dir / f"top{min(args.top_k, n - 1)}_indices.npy"
        if (
            previous.get("status") == "SUCCESS"
            and previous.get("run_key") == run_key
            and emb_path.exists()
            and neighbors_path.exists()
            and metrics_path.exists()
        ):
            log(f"\n[{name}] Complete for this exact configuration; skipping.", log_path)
            continue

        log(f"\n[{name}] Loading {load_source}", log_path)
        model = None
        start_model = time.perf_counter()
        encode_seconds = 0.0
        try:
            if local_path and not local_path.exists():
                raise FileNotFoundError(f"Local model directory missing: {local_path}")

            model = SentenceTransformer(
                load_source,
                device="cpu",
                trust_remote_code=config["trust_remote_code"],
            )
            model.max_seq_length = args.max_seq_length
            dimension = int(model.get_sentence_embedding_dimension())
            expected_bytes = n * dimension * np.dtype(np.float32).itemsize

            state: dict[str, Any] = {}
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    state = {}

            can_resume = (
                state.get("run_key") == run_key
                and state.get("model_id") == model_identifier
                and state.get("dimension") == dimension
                and emb_path.exists()
                and emb_path.stat().st_size == expected_bytes
            )
            if not can_resume:
                if emb_path.exists():
                    emb_path.unlink()
                state = {
                    "model": name,
                    "model_id": model_identifier,
                    "run_key": run_key,
                    "dataset_fingerprint": fingerprint,
                    "rows": n,
                    "dimension": dimension,
                    "completed_rows": 0,
                    "status": "ENCODING",
                }
                write_json(state_path, state)

            completed = int(state.get("completed_rows", 0))
            if completed < 0 or completed > n:
                completed = 0
            completed_at_start = completed

            embeddings = np.memmap(
                emb_path,
                dtype=np.float32,
                mode="r+" if emb_path.exists() else "w+",
                shape=(n, dimension),
            )

            encode_start = time.perf_counter()
            for start in range(completed, n, args.chunk_size):
                end = min(start + args.chunk_size, n)
                batch_texts = [config["prefix"] + text for text in texts[start:end]]
                vectors = model.encode(
                    batch_texts,
                    batch_size=args.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                embeddings[start:end] = np.asarray(vectors, dtype=np.float32)
                embeddings.flush()
                state["completed_rows"] = end
                state["status"] = "ENCODING"
                write_json(state_path, state)
                log(f"  Encoded {end:,}/{n:,}", log_path)

            encode_seconds = time.perf_counter() - encode_start
            del embeddings
            del model
            model = None
            gc.collect()

            log(f"  Embeddings saved ({dimension} dimensions). Computing exact top-k.", log_path)
            neighbor_indices = exact_cosine_topk(
                emb_path=emb_path,
                n=n,
                dim=dimension,
                top_k=args.top_k,
                block_size=args.neighbor_block_size,
                log_path=log_path,
                model_key=name,
            )
            metric_values = calculate_cpc_metrics(
                neighbor_indices=neighbor_indices,
                labels=labels,
                top_k=args.top_k,
            )
            del neighbor_indices
            gc.collect()

            duration = time.perf_counter() - start_model
            newly_encoded = n - completed_at_start
            metric = {
                "model": name,
                "model_id": model_identifier,
                "local_path": str(local_path) if local_path else None,
                "status": "SUCCESS",
                "run_key": run_key,
                "dataset_fingerprint": fingerprint,
                "rows": n,
                "dimension": dimension,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.batch_size,
                "chunk_size": args.chunk_size,
                "prefix": config["prefix"],
                "text_fields": ["title", "abstract"],
                "embedding_file": str(emb_path),
                "neighbor_indices_file": str(neighbors_path),
                "encoding_seconds_this_invocation": encode_seconds,
                "newly_encoded_rows_this_invocation": newly_encoded,
                "patents_per_second_this_invocation": (
                    newly_encoded / encode_seconds if encode_seconds > 0 else None
                ),
                "total_seconds_this_invocation": duration,
                "device": "cpu",
                "top_k": min(args.top_k, n - 1),
                **metric_values,
            }
            write_json(metrics_path, metric)
            state.update(
                {
                    "completed_rows": n,
                    "status": "SUCCESS",
                    "dimension": dimension,
                    "run_key": run_key,
                }
            )
            write_json(state_path, state)
            results[name] = metric

            log(
                f"  SUCCESS: {duration / 60:.1f} min; "
                f"hit@{args.top_k}={metric_values['same_cpc_hit_at_k']:.4f}; "
                f"precision@{args.top_k}={metric_values['same_cpc_precision_at_k']:.4f}; "
                f"MRR@{args.top_k}={metric_values['same_cpc_mrr_at_k']:.4f}",
                log_path,
            )

        except Exception as exc:
            if model is not None:
                del model
            gc.collect()
            error_text = f"{type(exc).__name__}: {exc}"
            log(f"  FAILED: {error_text}", log_path)
            log(traceback.format_exc(), log_path)

            old_state = {}
            if state_path.exists():
                try:
                    old_state = json.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            old_state.update(
                {
                    "model": name,
                    "model_id": model_identifier,
                    "run_key": run_key,
                    "dataset_fingerprint": fingerprint,
                    "status": "FAILED",
                    "error": error_text,
                }
            )
            write_json(state_path, old_state)
            results[name] = {
                "model": name,
                "model_id": model_identifier,
                "status": "FAILED",
                "run_key": run_key,
                "dataset_fingerprint": fingerprint,
                "rows": n,
                "error": error_text,
            }
        finally:
            gc.collect()
            pd.DataFrame(list(results.values())).to_csv(summary_path, index=False)

    pd.DataFrame(list(results.values())).to_csv(summary_path, index=False)

    successful = [
        result for result in results.values()
        if result.get("status") == "SUCCESS" and result.get("run_key")
    ]

    def fmt(value: Any, digits: int = 4) -> str:
        try:
            number = float(value)
            return f"{number:.{digits}f}" if np.isfinite(number) else ""
        except (TypeError, ValueError):
            return ""

    report = [
        "# Patent Embedding Benchmark V2",
        "",
        f"- Source rows: **{source_rows:,}**",
        f"- Rows benchmarked: **{n:,}**",
        f"- Unique patent IDs: **{df['patent_number'].nunique():,}**",
        f"- Dataset fingerprint: `{fingerprint}`",
        "- Text: title + abstract.",
        f"- Maximum sequence length: **{args.max_seq_length}**",
        f"- CPC source column: `{cpc_col or 'unavailable'}`",
        f"- Successful model results: **{len(successful)}**",
        "- Evaluation: each patent is used as a query against the same corpus; self-matches are excluded.",
        "- Relevance proxy: equality of the first four CPC characters.",
        "",
        "## Results",
        "",
        "| Model | Status | Dim | Encode sec | Patents/sec | Hit@K | Precision@K | Recall@K | MRR@K | nDCG@K | Lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in selected_models:
        result = results.get(config["name"], {})
        report.append(
            f"| {config['name']} | {result.get('status', 'NOT RUN')} | "
            f"{result.get('dimension', '')} | "
            f"{fmt(result.get('encoding_seconds_this_invocation'), 2)} | "
            f"{fmt(result.get('patents_per_second_this_invocation'), 3)} | "
            f"{fmt(result.get('same_cpc_hit_at_k'))} | "
            f"{fmt(result.get('same_cpc_precision_at_k'))} | "
            f"{fmt(result.get('same_cpc_recall_at_k'))} | "
            f"{fmt(result.get('same_cpc_mrr_at_k'))} | "
            f"{fmt(result.get('same_cpc_ndcg_at_k'))} | "
            f"{fmt(result.get('precision_lift_over_random'))} |"
        )

    report += [
        "",
        "## Metric definitions",
        "",
        "- **Hit@K:** fraction of labelled queries with at least one same-subclass neighbour in the top K.",
        "- **Precision@K:** same-subclass neighbours divided by all returned top-K neighbours.",
        "- **Recall@K:** retrieved same-subclass neighbours divided by all other corpus records with that subclass.",
        "- **MRR@K:** mean reciprocal rank of the first same-subclass neighbour, with zero for no hit.",
        "- **nDCG@K:** discounted gain for same-subclass neighbours, normalized by the ideal ranking for that query.",
        "- **Random baseline:** sum of squared empirical frequencies of non-empty four-character CPC subclasses.",
        "- **Precision lift:** Precision@K divided by the random baseline.",
        "",
        "## Interpretation and limitations",
        "",
        "- CPC subclass agreement is a weak metadata proxy, not a human relevance judgement.",
        "- Same-corpus evaluation is useful for comparing neighbourhood structure but is not a held-out query/document retrieval test.",
        "- CPC subclasses can be broad; a same-subclass patent is not necessarily relevant prior art.",
        "- Model-specific prefixes and truncation can affect results; all choices are recorded in `run_manifest.json`.",
        "- For a production decision, add independently judged patent-query relevance labels and evaluate on held-out queries.",
        "",
        "## Output files",
        "",
        "- `model_comparison.csv`: consolidated metrics.",
        "- `metrics/`: per-model JSON metrics.",
        "- `embeddings/`: normalized float32 embedding matrices and top-K index arrays.",
        "- `state/`: resumable model encoding state.",
        "- `dataset_manifest.csv`: exact rows used in this run.",
        "- `run_manifest.json`: configuration and dataset fingerprint.",
    ]
    (output_dir / "benchmark_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    log(
        f"\nFinished. See {summary_path} and {output_dir / 'benchmark_report.md'}",
        log_path,
    )


if __name__ == "__main__":
    main()
