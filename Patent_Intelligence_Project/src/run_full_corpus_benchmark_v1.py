#!/usr/bin/env python3
"""
Resumable patent embedding benchmark for the local Hugging Face DatasetDict.

Default input: data/sample_dataset (currently verified as 25,247 unique records).
This script never downloads data and writes only under its chosen output directory.

Smoke test:
  ./.Patent_Intelligence_virtual_env_final/bin/python run_full_corpus_benchmark_v1.py \
      --limit 3000 --expected-rows 25247 \
      --output results/full_corpus_benchmark_smoke3000

Full run:
  ./.Patent_Intelligence_virtual_env_final/bin/python run_full_corpus_benchmark_v1.py \
      --expected-rows 25247 --output results/full_corpus_benchmark

The smoke test and full run have separate fingerprints and output directories.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "data" / "sample_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "full_corpus_benchmark"

# Prefixes are chosen for symmetric patent-to-patent similarity where model cards
# support a task prefix. These choices are recorded in every manifest/metric.
MODELS = [
    {"name": "PatentSBERTa", "hf_id": "AI-Growth-Lab/PatentSBERTa", "prefix": "", "trust_remote_code": False},
    {"name": "PaECTER", "hf_id": "mpi-inno-comp/paecter", "prefix": "", "trust_remote_code": False},
    {"name": "Nomic-Embed-v1.5", "hf_id": "nomic-ai/nomic-embed-text-v1.5", "prefix": "clustering: ", "trust_remote_code": True},
    {"name": "BGE-base-en-v1.5", "hf_id": "BAAI/bge-base-en-v1.5", "prefix": "", "trust_remote_code": False},
    {"name": "E5-base-v2", "hf_id": "intfloat/e5-base-v2", "prefix": "query: ", "trust_remote_code": False},
    {"name": "all-mpnet-base-v2", "hf_id": "sentence-transformers/all-mpnet-base-v2", "prefix": "", "trust_remote_code": False},
    {"name": "all-MiniLM-L6-v2", "hf_id": "sentence-transformers/all-MiniLM-L6-v2", "prefix": "", "trust_remote_code": False},
]


def write_json(path: Path, data: dict) -> None:
    # JSON has no NaN/Infinity values; normalize non-finite NumPy/Python floats to null.
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def log(message: str, log_path: Path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def safe_name(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def load_rows(dataset_dir: Path) -> tuple[pd.DataFrame, str | None]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    saved = load_from_disk(str(dataset_dir))
    if not hasattr(saved, "items"):
        raise ValueError(f"Expected a DatasetDict at {dataset_dir}, got {type(saved).__name__}")
    frames = []
    for split_name, split in saved.items():
        available = set(split.column_names)
        cols = [c for c in ["patent_number", "title", "abstract", "cpc_label"] if c in available]
        frame = split.select_columns(cols).to_pandas()
        frame["source_split"] = split_name
        frames.append(frame)
    if not frames:
        raise ValueError("DatasetDict has no splits.")
    df = pd.concat(frames, ignore_index=True)
    if "title" not in df or "abstract" not in df:
        raise ValueError("Dataset must contain both 'title' and 'abstract'.")
    if "patent_number" not in df:
        df["patent_number"] = [f"row_{i:06d}" for i in range(len(df))]
    df["patent_number"] = df["patent_number"].fillna("").astype(str).str.strip()
    empty_ids = df["patent_number"].eq("")
    if empty_ids.any():
        df.loc[empty_ids, "patent_number"] = [f"row_{i:06d}" for i in df.index[empty_ids]]
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
        raise ValueError(f"Found {duplicates} duplicate patent_number values; inspect source first.")
    return df, cpc_col


def dataset_fingerprint(ids: list[str], texts: list[str], labels: list[str]) -> str:
    h = hashlib.sha256()
    for group in (ids, texts, labels):
        h.update(b"\n--FIELD--\n")
        for value in group:
            h.update(value.encode("utf-8", errors="replace"))
            h.update(b"\0")
    return h.hexdigest()


def metric_baseline(labels: list[str]) -> float:
    """Chance subclass match for two independent draws from the labelled corpus."""
    valid = [x[:4] for x in labels if x]
    if not valid:
        return float("nan")
    counts = pd.Series(valid).value_counts(normalize=True)
    return float((counts ** 2).sum())


def cosine_topk(
    emb_path: Path,
    n: int,
    dim: int,
    ids: list[str],
    labels: list[str],
    titles: list[str],
    abstracts: list[str],
    output_path: Path,
    top_k: int,
    block_size: int,
    log_path: Path,
    query_indices: set[int],
    examples_path: Path,
    log_every_blocks: int = 10,
) -> dict:
    """Exact cosine top-k in blocks; vectors are normalized float32 memmap rows."""
    emb = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(n, dim))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    examples_tmp = examples_path.with_suffix(examples_path.suffix + ".tmp")
    rows_written = 0
    valid_queries = 0
    hit_values: list[float] = []
    precision_values: list[float] = []
    baseline = metric_baseline(labels)

    edge_fields = [
        "query_index", "query_patent_number", "neighbor_rank", "neighbor_index",
        "neighbor_patent_number", "cosine_similarity", "query_cpc", "neighbor_cpc", "same_cpc",
    ]
    example_fields = [
        "query_index", "query_patent_number", "query_title", "query_abstract", "neighbor_rank",
        "neighbor_index", "neighbor_patent_number", "neighbor_title", "neighbor_abstract",
        "cosine_similarity", "query_cpc", "neighbor_cpc", "same_cpc",
    ]
    with tmp_path.open("w", encoding="utf-8", newline="") as edge_f, examples_tmp.open(
        "w", encoding="utf-8", newline=""
    ) as example_f:
        edge_writer = csv.DictWriter(edge_f, fieldnames=edge_fields)
        example_writer = csv.DictWriter(example_f, fieldnames=example_fields)
        edge_writer.writeheader()
        example_writer.writeheader()

        total_blocks = (n + block_size - 1) // block_size
        for block_no, start in enumerate(range(0, n, block_size), start=1):
            end = min(start + block_size, n)
            # Only this block's score matrix is materialized in RAM.
            scores = np.asarray(emb[start:end]) @ emb.T
            for local_i, score_row in enumerate(scores):
                query_i = start + local_i
                score_row[query_i] = -np.inf
                k = min(top_k, n - 1)
                candidate_idx = np.argpartition(score_row, -k)[-k:]
                candidate_idx = candidate_idx[np.argsort(score_row[candidate_idx])[::-1]]
                qlabel = labels[query_i][:4] if labels[query_i] else ""
                query_has_label = bool(qlabel)
                labelled_matches = []
                for rank, neighbor_i_raw in enumerate(candidate_idx, start=1):
                    neighbor_i = int(neighbor_i_raw)
                    nlabel = labels[neighbor_i][:4] if labels[neighbor_i] else ""
                    same = bool(qlabel and nlabel and qlabel == nlabel)
                    if nlabel:
                        labelled_matches.append(int(same))
                    edge_writer.writerow({
                        "query_index": query_i,
                        "query_patent_number": ids[query_i],
                        "neighbor_rank": rank,
                        "neighbor_index": neighbor_i,
                        "neighbor_patent_number": ids[neighbor_i],
                        "cosine_similarity": f"{float(score_row[neighbor_i]):.7f}",
                        "query_cpc": labels[query_i],
                        "neighbor_cpc": labels[neighbor_i],
                        "same_cpc": int(same),
                    })
                    rows_written += 1
                    if query_i in query_indices and rank <= 5:
                        example_writer.writerow({
                            "query_index": query_i,
                            "query_patent_number": ids[query_i],
                            "query_title": titles[query_i],
                            "query_abstract": abstracts[query_i],
                            "neighbor_rank": rank,
                            "neighbor_index": neighbor_i,
                            "neighbor_patent_number": ids[neighbor_i],
                            "neighbor_title": titles[neighbor_i],
                            "neighbor_abstract": abstracts[neighbor_i],
                            "cosine_similarity": f"{float(score_row[neighbor_i]):.7f}",
                            "query_cpc": labels[query_i],
                            "neighbor_cpc": labels[neighbor_i],
                            "same_cpc": int(same),
                        })
                if query_has_label:
                    valid_queries += 1
                    hit_values.append(float(any(labelled_matches)))
                    if labelled_matches:
                        precision_values.append(float(np.mean(labelled_matches)))
            del scores
            if block_no % log_every_blocks == 0 or block_no == total_blocks:
                log(f"  Neighbours: {end:,}/{n:,} queries ({block_no}/{total_blocks} blocks)", log_path)

    tmp_path.replace(output_path)
    examples_tmp.replace(examples_path)
    del emb
    precision_at_k = float(np.mean(precision_values)) if precision_values else float("nan")
    hit_at_k = float(np.mean(hit_values)) if hit_values else float("nan")
    lift = precision_at_k / baseline if baseline > 0 and np.isfinite(precision_at_k) else float("nan")
    return {
        "neighbor_edges": rows_written,
        "labelled_query_count": valid_queries,
        "same_cpc_hit_at_k": hit_at_k,
        "same_cpc_precision_at_k": precision_at_k,
        "random_cpc_match_baseline": baseline,
        "precision_lift_over_random": lift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable patent embedding benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16, help="Model inference batch size.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Rows encoded before a checkpoint.")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--neighbor-block-size", type=int, default=64)
    parser.add_argument("--expected-rows", type=int, default=25247,
                        help="Expected number of rows before --limit sampling; use 0 to disable.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Deterministically sample this many records after validating source row count.")
    args = parser.parse_args()
    if min(args.batch_size, args.chunk_size, args.max_seq_length, args.top_k, args.neighbor_block_size) < 1:
        parser.error("batch-size, chunk-size, max-seq-length, top-k and neighbor-block-size must be positive")
    if args.limit < 0:
        parser.error("--limit must be zero (all records) or a positive integer")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "benchmark.log"
    log("=" * 72, log_path)
    log("PATENT EMBEDDING BENCHMARK V1", log_path)
    log(f"Dataset: {args.dataset.resolve()}", log_path)
    log(f"Output: {output_dir}", log_path)
    log(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}", log_path)
    log("Device: CPU", log_path)

    df, cpc_col = load_rows(args.dataset.resolve())
    source_rows = len(df)
    if args.expected_rows and source_rows != args.expected_rows:
        raise SystemExit(
            f"SAFETY STOP: source has {source_rows:,} rows; expected {args.expected_rows:,}. "
            "No model has been loaded. Verify the dataset or pass --expected-rows 0."
        )
    if args.limit:
        if args.limit > source_rows:
            raise SystemExit(f"--limit {args.limit:,} exceeds source row count {source_rows:,}.")
        df = df.sample(n=args.limit, random_state=0, replace=False).sort_index().reset_index(drop=True)
    n = len(df)
    ids = df["patent_number"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()
    labels = df["cpc_label_clean"].astype(str).tolist()
    titles = df["title"].astype(str).tolist()
    abstracts = df["abstract"].astype(str).tolist()
    fp = dataset_fingerprint(ids, texts, labels)

    # Ten deterministic query patents, shared across all models for this run.
    rng = np.random.default_rng(42)
    query_indices = set(int(x) for x in rng.choice(n, size=min(10, n), replace=False))
    (output_dir / "dataset_manifest.csv").write_text("", encoding="utf-8")
    df[["patent_number", "source_split", "title", "abstract", "cpc_label_clean", "cpc_subclass"]].to_csv(
        output_dir / "dataset_manifest.csv", index=False
    )
    run_manifest = {
        "dataset_path": str(args.dataset.resolve()),
        "source_rows": source_rows,
        "rows_used": n,
        "limit": args.limit,
        "unique_patent_numbers": int(df["patent_number"].nunique()),
        "dataset_fingerprint_sha256": fp,
        "text_fields": ["title", "abstract"],
        "cpc_source_column": cpc_col,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "max_seq_length": args.max_seq_length,
        "top_k": args.top_k,
        "neighbor_block_size": args.neighbor_block_size,
        "query_indices_for_examples": sorted(query_indices),
        "models": MODELS,
        "device": "cpu",
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    log(f"Source rows: {source_rows:,}; rows used: {n:,}; unique IDs: {df['patent_number'].nunique():,}", log_path)
    log(f"CPC column: {cpc_col or 'unavailable'}; dataset fingerprint: {fp}", log_path)

    summary_path = output_dir / "model_comparison.csv"
    if summary_path.exists():
        try:
            old_rows = pd.read_csv(summary_path).to_dict("records")
        except Exception:
            old_rows = []
    else:
        old_rows = []
    results = {str(r.get("model")): r for r in old_rows if r.get("model")}

    for config in MODELS:
        name = config["name"]
        key = safe_name(name)
        model_dir = output_dir / "embeddings" / key
        model_dir.mkdir(parents=True, exist_ok=True)
        emb_path = model_dir / "embeddings.f32"
        state_path = output_dir / "state" / f"{key}.json"
        neighbors_path = output_dir / "neighbors" / f"{key}_top{args.top_k}.csv"
        examples_path = output_dir / "examples" / f"{key}_top5_queries.csv"
        metrics_path = output_dir / "metrics" / f"{key}.json"

        run_key_payload = {
            "dataset_fingerprint": fp,
            "model_id": config["hf_id"],
            "prefix": config["prefix"],
            "max_seq_length": args.max_seq_length,
            "text_fields": ["title", "abstract"],
            "normalize_embeddings": True,
            "dtype": "float32",
            "rows": n,
        }
        run_key = hashlib.sha256(json.dumps(run_key_payload, sort_keys=True).encode()).hexdigest()
        previous = results.get(name, {})
        if (
            previous.get("status") == "SUCCESS"
            and previous.get("run_key") == run_key
            and neighbors_path.exists()
            and examples_path.exists()
            and emb_path.exists()
        ):
            log(f"\n[{name}] Complete for this exact configuration; skipping.", log_path)
            continue

        log(f"\n[{name}] Loading {config['hf_id']}", log_path)
        model = None
        start_model = time.perf_counter()
        encode_seconds = 0.0
        try:
            model = SentenceTransformer(
                config["hf_id"],
                device="cpu",
                trust_remote_code=config["trust_remote_code"],
            )
            model.max_seq_length = args.max_seq_length
            dim = int(model.get_sentence_embedding_dimension())
            expected_bytes = n * dim * np.dtype(np.float32).itemsize

            state = {}
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    state = {}
            can_resume = (
                state.get("run_key") == run_key
                and state.get("model_id") == config["hf_id"]
                and state.get("dimension") == dim
                and emb_path.exists()
                and emb_path.stat().st_size == expected_bytes
            )
            if not can_resume:
                if emb_path.exists():
                    emb_path.unlink()
                state = {
                    "model": name, "model_id": config["hf_id"], "run_key": run_key,
                    "dataset_fingerprint": fp, "rows": n, "dimension": dim,
                    "completed_rows": 0, "status": "ENCODING",
                }
                write_json(state_path, state)
            completed = int(state.get("completed_rows", 0))
            if completed < 0 or completed > n:
                completed = 0
            completed_at_start = completed
            emb = np.memmap(
                emb_path, dtype=np.float32, mode="r+" if emb_path.exists() else "w+", shape=(n, dim)
            )
            encode_start = time.perf_counter()
            for start in range(completed, n, args.chunk_size):
                end = min(start + args.chunk_size, n)
                batch_texts = [config["prefix"] + t for t in texts[start:end]]
                vectors = model.encode(
                    batch_texts,
                    batch_size=args.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                emb[start:end] = np.asarray(vectors, dtype=np.float32)
                emb.flush()
                state["completed_rows"] = end
                state["status"] = "ENCODING"
                write_json(state_path, state)
                if end == n or end // args.chunk_size != start // args.chunk_size:
                    log(f"  Encoded {end:,}/{n:,}", log_path)
            encode_seconds = time.perf_counter() - encode_start
            del emb
            gc.collect()
            del model
            model = None
            gc.collect()

            log(f"  Embeddings saved ({dim} dimensions). Building top-{args.top_k} neighbours.", log_path)
            metric_values = cosine_topk(
                emb_path, n, dim, ids, labels, titles, abstracts, neighbors_path,
                args.top_k, args.neighbor_block_size, log_path, query_indices, examples_path,
            )
            duration = time.perf_counter() - start_model
            newly_encoded = n - completed_at_start
            metric = {
                "model": name,
                "model_id": config["hf_id"],
                "status": "SUCCESS",
                "run_key": run_key,
                "dataset_fingerprint": fp,
                "rows": n,
                "dimension": dim,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.batch_size,
                "chunk_size": args.chunk_size,
                "prefix": config["prefix"],
                "text_fields": ["title", "abstract"],
                "embedding_file": str(emb_path),
                "neighbor_file": str(neighbors_path),
                "examples_file": str(examples_path),
                "encoding_seconds_this_invocation": encode_seconds,
                "newly_encoded_rows_this_invocation": newly_encoded,
                "patents_per_second_this_invocation": newly_encoded / encode_seconds if encode_seconds > 0 else None,
                "total_seconds_this_invocation": duration,
                "device": "cpu",
                "top_k": args.top_k,
                **metric_values,
            }
            write_json(metrics_path, metric)
            state.update({"completed_rows": n, "status": "SUCCESS", "dimension": dim, "run_key": run_key})
            write_json(state_path, state)
            results[name] = metric
            log(
                f"  SUCCESS: {duration/60:.1f} min; hit@{args.top_k}={metric_values['same_cpc_hit_at_k']:.4f}; "
                f"precision@{args.top_k}={metric_values['same_cpc_precision_at_k']:.4f}",
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
            old_state.update({
                "model": name, "model_id": config["hf_id"], "run_key": run_key,
                "dataset_fingerprint": fp, "status": "FAILED", "error": error_text,
            })
            write_json(state_path, old_state)
            results[name] = {
                "model": name, "model_id": config["hf_id"], "status": "FAILED",
                "run_key": run_key, "dataset_fingerprint": fp, "rows": n, "error": error_text,
            }
        finally:
            gc.collect()

        pd.DataFrame(list(results.values())).to_csv(summary_path, index=False)

    pd.DataFrame(list(results.values())).to_csv(summary_path, index=False)
    successful = [r for r in results.values() if r.get("status") == "SUCCESS" and r.get("run_key")]
    report = [
        "# Patent Embedding Benchmark V1",
        "",
        f"- Source rows: **{source_rows:,}**",
        f"- Rows benchmarked: **{n:,}**",
        f"- Unique patent IDs in run: **{df['patent_number'].nunique():,}**",
        f"- Dataset fingerprint: `{fp}`",
        "- Text: title + abstract; model maximum sequence length set to 512 by default.",
        f"- CPC source column: `{cpc_col or 'unavailable'}`",
        f"- Successful model results recorded: **{len(successful)}**",
        "",
        "## Results",
        "",
        "| Model | Status | Dimensions | Newly encoded rows | Encoding seconds | Patents/sec | Hit@k | Precision@k | Random baseline | Precision lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in MODELS:
        r = results.get(config["name"], {})
        def fmt(value, digits=4):
            try:
                return f"{float(value):.{digits}f}" if value is not None and np.isfinite(float(value)) else ""
            except (TypeError, ValueError):
                return ""
        report.append(
            f"| {config['name']} | {r.get('status', 'NOT RUN')} | {r.get('dimension', '')} | "
            f"{r.get('newly_encoded_rows_this_invocation', '')} | {fmt(r.get('encoding_seconds_this_invocation'), 2)} | "
            f"{fmt(r.get('patents_per_second_this_invocation'), 3)} | {fmt(r.get('same_cpc_hit_at_k'))} | "
            f"{fmt(r.get('same_cpc_precision_at_k'))} | {fmt(r.get('random_cpc_match_baseline'))} | "
            f"{fmt(r.get('precision_lift_over_random'))} |"
        )
    report += [
        "",
        "## Metric definitions",
        "",
        "- **Hit@k:** fraction of queries with a non-empty CPC subclass that have at least one matching labelled neighbour among top-k.",
        "- **Precision@k:** for each query with a CPC subclass, fraction of its top-k neighbours that have a non-empty CPC subclass matching the query; average over queries with at least one labelled neighbour.",
        "- **Random baseline:** sum of squared empirical frequencies of non-empty four-character CPC subclasses.",
        "- **Precision lift:** precision@k divided by the random baseline. This is a proxy metric, not a human relevance judgement or proof of prior-art retrieval quality.",
        "- Exact cosine top-k is computed in blocks; the full N-by-N similarity matrix is never stored.",
        "- Nomic uses `clustering: ` and E5 uses `query: ` for symmetric similarity; all chosen prefixes are recorded in the run manifest and metrics.",
        "- Resume compatibility includes dataset fingerprint, model ID, prefix, max sequence length, text fields, normalization, dtype and row count.",
        "- A smoke-test run and full run must use different output directories.",
        "",
        "Inspect `examples/` for the same ten deterministic query patents across models.",
    ]
    (output_dir / "benchmark_report.md").write_text("\n".join(report), encoding="utf-8")
    log(f"\nFinished. See {summary_path} and {output_dir / 'benchmark_report.md'}", log_path)


if __name__ == "__main__":
    main()
