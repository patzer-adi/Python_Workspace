import gc
import math
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Stratified dataset (3 CPC subclasses x 5 patents), built by
# build_stratified_dataset.py
DATA_FILE = PROJECT_DIR / "data" / "hupd_sample" / "patent_sample_15_stratified.csv"

RESULTS_DIR = PROJECT_DIR / "results"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"

RESULTS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# 15 patents is small; keep batch_size conservative and reliability-first
# rather than tuning for speed, per current stage of the project.
BATCH_SIZE = 2

# For 15 patents, evaluate top-3 retrieval per patent.
TOP_K = 3


# ============================================================
# MODELS
# ============================================================
# Prefixes below are DOCUMENT-side prefixes only, verified against each
# model's current model card. Query-side prefixes (search_query: / query: /
# a BGE query instruction) are a separate concern for whenever query-time
# retrieval is implemented -- do not reuse these document prefixes for
# queries later.

MODELS = [
    {
        "name": "PatentSBERTa",
        "hf_id": "AI-Growth-Lab/PatentSBERTa",
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "PaECTER",
        "hf_id": "mpi-inno-comp/paecter",
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "Nomic-Embed-v1.5",
        "hf_id": "nomic-ai/nomic-embed-text-v1.5",
        "prefix": "search_document: ",
        "trust_remote_code": True,
    },
    {
        "name": "BGE-base-en-v1.5",
        "hf_id": "BAAI/bge-base-en-v1.5",
        "prefix": "",  # BGE needs no prefix for documents; queries need an instruction prefix later
        "trust_remote_code": False,
    },
    {
        "name": "E5-base-v2",
        "hf_id": "intfloat/e5-base-v2",
        "prefix": "passage: ",
        "trust_remote_code": False,
    },
    {
        "name": "all-mpnet-base-v2",
        "hf_id": "sentence-transformers/all-mpnet-base-v2",
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "name": "all-MiniLM-L6-v2",
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "prefix": "",
        "trust_remote_code": False,
    },
]


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = RESULTS_DIR / "benchmark_log.txt"


def log(message):
    print(message, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# ============================================================
# DEVICE
# ============================================================

PREFERRED_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

log("=" * 70)
log("PATENT EMBEDDING MODEL BENCHMARK")
log("=" * 70)
log(f"Preferred device: {PREFERRED_DEVICE}")
log(f"PyTorch: {torch.__version__}")
log(f"Input file: {DATA_FILE}")
log("=" * 70)


# ============================================================
# LOAD PATENTS
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nStratified patent CSV not found:\n{DATA_FILE}\n\n"
        "Run build_stratified_dataset.py first to generate it."
    )

df = pd.read_csv(DATA_FILE)

required_columns = ["title", "abstract", "main_cpc_label"]
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

df["title"] = df["title"].fillna("")
df["abstract"] = df["abstract"].fillna("")
df["main_cpc_label"] = df["main_cpc_label"].fillna("")
df["cpc_subclass"] = df["main_cpc_label"].astype(str).str.slice(0, 4)

df["text"] = (
    df["title"].astype(str).str.strip()
    + "\n\n"
    + df["abstract"].astype(str).str.strip()
)

n_patents = len(df)
log(f"Number of patents: {n_patents}")
log(f"CPC subclasses present: {sorted(df['cpc_subclass'].unique().tolist())}")

# Chance baseline for same-CPC hit@3: with n_patents total, each patent has
# (class_size - 1) same-subclass "true positive" candidates among the other
# (n_patents - 1). A model with NO real signal would still hit this often
# by chance, drawing TOP_K without replacement (hypergeometric).
# This assumes equal class sizes; computed per-patent below for accuracy
# and averaged, since class sizes may not be perfectly balanced.


def chance_hit_rate_at_k(df, k):
    n = len(df)
    rates = []
    for i in range(n):
        subclass = df.iloc[i]["cpc_subclass"]
        same_class_others = (df["cpc_subclass"] == subclass).sum() - 1
        pool = n - 1
        if pool <= 0:
            continue
        # P(at least one same-class hit in k draws without replacement)
        k_eff = min(k, pool)
        other_class_others = pool - same_class_others
        if k_eff > other_class_others:
            rates.append(1.0)
            continue
        p_no_hit = math.comb(other_class_others, k_eff) / math.comb(pool, k_eff)
        rates.append(1 - p_no_hit)
    return float(np.mean(rates)) if rates else float("nan")


CHANCE_HIT_RATE = chance_hit_rate_at_k(df, TOP_K)
log(
    f"Chance-level same-CPC hit@{TOP_K} baseline for this sample: "
    f"{CHANCE_HIT_RATE:.4f} "
    "(a model with no real signal would score roughly this by luck alone)"
)


# ============================================================
# COMPLETION TRACKING  (FIXED: only SUCCESS counts as completed)
# ============================================================

SUMMARY_FILE = RESULTS_DIR / "model_comparison.csv"

completed_models = set()
benchmark_rows = []

if SUMMARY_FILE.exists():
    try:
        old_results = pd.read_csv(SUMMARY_FILE)
        if "model" in old_results.columns and "status" in old_results.columns:
            completed_models = set(
                old_results.loc[old_results["status"] == "SUCCESS", "model"].astype(str)
            )
            if completed_models:
                log(f"Previously completed (SUCCESS) models: {', '.join(sorted(completed_models))}")

            failed_before = set(
                old_results.loc[old_results["status"] == "FAILED", "model"].astype(str)
            )
            if failed_before:
                log(f"Previously FAILED models (will be retried): {', '.join(sorted(failed_before))}")

        # Keep prior rows for models we're not about to rerun; rows for
        # FAILED models get replaced when that model is retried below.
        benchmark_rows = [
            row for row in old_results.to_dict("records")
            if row.get("status") != "FAILED"
        ]
    except Exception:
        benchmark_rows = []


# ============================================================
# MODEL LOAD + ENCODE WITH MPS -> CPU FALLBACK
# ============================================================

def load_and_encode(config, texts_for_model, preferred_device):
    """
    Try preferred_device first; on any failure (e.g. an MPS op not
    implemented for a model's custom architecture), fall back to CPU.
    Returns (model, embeddings, load_time, encode_time, device_used, fell_back).
    """
    devices_to_try = [preferred_device]
    if preferred_device != "cpu":
        devices_to_try.append("cpu")

    last_error = None
    for device in devices_to_try:
        try:
            log(f"  Trying device: {device}")
            load_start = time.perf_counter()
            model = SentenceTransformer(
                config["hf_id"],
                device=device,
                trust_remote_code=config["trust_remote_code"],
            )
            load_time = time.perf_counter() - load_start

            encode_start = time.perf_counter()
            embeddings = model.encode(
                texts_for_model,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            encode_time = time.perf_counter() - encode_start

            fell_back = device != preferred_device
            return model, np.asarray(embeddings), load_time, encode_time, device, fell_back

        except Exception as e:
            last_error = e
            log(f"  [WARN] Device '{device}' failed for {config['name']}: {e}")
            if device != devices_to_try[-1]:
                log(f"  Retrying {config['name']} on next fallback device...")
            continue

    raise last_error


# ============================================================
# MODEL RUNNER
# ============================================================

for model_number, config in enumerate(MODELS, start=1):
    model_name = config["name"]
    model_id = config["hf_id"]

    log("\n" + "=" * 70)
    log(f"MODEL {model_number}/{len(MODELS)}: {model_name}")
    log("=" * 70)

    if model_name in completed_models:
        log(f"[SKIP] {model_name} already completed successfully.")
        continue

    model = None

    try:
        log(f"Model ID: {model_id}")

        texts = df["text"].tolist()
        prefix = config["prefix"]
        texts_for_model = [prefix + t for t in texts] if prefix else texts

        model, embeddings, load_time, encode_time, device_used, fell_back = load_and_encode(
            config, texts_for_model, PREFERRED_DEVICE
        )

        embedding_dimension = embeddings.shape[1]

        log(f"Model loaded in {load_time:.2f}s on device={device_used} (fallback={fell_back})")
        log(f"Encoding completed in {encode_time:.2f}s")
        log(f"Embedding shape: {embeddings.shape}")

        safe_name = model_name.lower().replace("-", "_").replace(" ", "_")

        # ---- save embeddings ----
        embedding_file = EMBEDDINGS_DIR / f"{safe_name}.npy"
        np.save(embedding_file, embeddings)
        log(f"Embeddings saved: {embedding_file}")

        # ---- similarity matrix ----
        similarity_matrix = cosine_similarity(embeddings)
        similarity_file = RESULTS_DIR / f"{safe_name}_similarity.npy"
        np.save(similarity_file, similarity_matrix)

        # ---- top-K retrieval + same-CPC hit@K (proxy metric) ----
        retrieval_rows = []
        hits = 0

        for i in range(n_patents):
            similarities = similarity_matrix[i].copy()
            similarities[i] = -np.inf
            top_indices = np.argsort(similarities)[::-1][:TOP_K]

            query_subclass = df.iloc[i]["cpc_subclass"]
            any_same_cpc = False

            for rank, j in enumerate(top_indices, start=1):
                matched_subclass = df.iloc[j]["cpc_subclass"]
                same_cpc = bool(matched_subclass == query_subclass)
                if same_cpc:
                    any_same_cpc = True

                retrieval_rows.append({
                    "model": model_name,
                    "query_patent": df.iloc[i]["patent_number"],
                    "query_title": df.iloc[i]["title"],
                    "query_cpc_subclass": query_subclass,
                    "rank": rank,
                    "retrieved_patent": df.iloc[j]["patent_number"],
                    "retrieved_title": df.iloc[j]["title"],
                    "retrieved_cpc_subclass": matched_subclass,
                    "similarity": float(similarities[j]),
                    "same_cpc": same_cpc,
                })

            if any_same_cpc:
                hits += 1

        same_cpc_hit_rate_at_k = hits / n_patents

        retrieval_file = RESULTS_DIR / f"{safe_name}_retrieval.csv"
        pd.DataFrame(retrieval_rows).to_csv(retrieval_file, index=False)
        log(f"Retrieval results saved: {retrieval_file}")

        # ---- descriptive-only diagnostics (NOT used for model selection) ----
        upper_values = similarity_matrix[np.triu_indices(n_patents, k=1)]
        average_similarity = float(np.mean(upper_values))
        min_similarity = float(np.min(upper_values))
        max_similarity = float(np.max(upper_values))

        result = {
            "model": model_name,
            "model_id": model_id,
            "status": "SUCCESS",
            "embedding_dimension": embedding_dimension,
            "load_time_seconds": round(load_time, 4),
            "encoding_time_seconds": round(encode_time, 4),
            "avg_time_per_patent": round(encode_time / n_patents, 4),
            f"same_cpc_hit_rate_at_{TOP_K}": round(same_cpc_hit_rate_at_k, 4),
            "device_used": device_used,
            "mps_fallback_to_cpu": fell_back,
            "error_message": "",
            # diagnostic only -- do not use to select a model
            "avg_pairwise_similarity_diagnostic": round(average_similarity, 6),
            "min_pairwise_similarity_diagnostic": round(min_similarity, 6),
            "max_pairwise_similarity_diagnostic": round(max_similarity, 6),
        }

        benchmark_rows.append(result)
        pd.DataFrame(benchmark_rows).to_csv(SUMMARY_FILE, index=False)

        log("")
        log(f"[OK] {model_name} COMPLETE")
        log(f"  Dimension: {embedding_dimension}")
        log(f"  Encoding time: {encode_time:.2f}s")
        log(f"  same_cpc_hit_rate_at_{TOP_K}: {same_cpc_hit_rate_at_k:.4f} "
            f"(chance baseline: {CHANCE_HIT_RATE:.4f})")

    except Exception as e:
        log("")
        log(f"[FAILED] {model_name}")
        log(f"Error: {str(e)}")

        error_file = RESULTS_DIR / f"{model_name.lower().replace('-', '_')}_error.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())

        failed_result = {
            "model": model_name,
            "model_id": model_id,
            "status": "FAILED",
            "embedding_dimension": None,
            "load_time_seconds": None,
            "encoding_time_seconds": None,
            "avg_time_per_patent": None,
            f"same_cpc_hit_rate_at_{TOP_K}": None,
            "device_used": None,
            "mps_fallback_to_cpu": None,
            "error_message": str(e),
            "avg_pairwise_similarity_diagnostic": None,
            "min_pairwise_similarity_diagnostic": None,
            "max_pairwise_similarity_diagnostic": None,
        }
        benchmark_rows.append(failed_result)
        pd.DataFrame(benchmark_rows).to_csv(SUMMARY_FILE, index=False)
        log("Continuing to the next model...")

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        log("Model memory released.")


# ============================================================
# FINAL SUMMARY + REPORT
# ============================================================

def dataframe_to_markdown(frame):
    """Minimal markdown table builder -- avoids depending on the optional
    'tabulate' package just for report generation."""
    cols = [str(c) for c in frame.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join([header, sep] + rows)


log("\n" + "=" * 70)
log("BENCHMARK COMPLETE")
log("=" * 70)

if not SUMMARY_FILE.exists():
    log("No results file was produced.")
else:
    final_df = pd.read_csv(SUMMARY_FILE)
    successful = (final_df["status"] == "SUCCESS").sum()
    failed = (final_df["status"] == "FAILED").sum()

    log(f"Successful models: {successful}/{len(MODELS)}")
    log(f"Failed models: {failed}/{len(MODELS)}")
    log("")
    log(str(final_df[[
        "model", "status", "embedding_dimension", "encoding_time_seconds",
        f"same_cpc_hit_rate_at_{TOP_K}",
    ]]))
    log(f"\nChance-level baseline for reference: {CHANCE_HIT_RATE:.4f}")
    log(f"\nMain comparison file:\n{SUMMARY_FILE}")

    # ---- auto-generated report ----
    report_path = RESULTS_DIR / "benchmark_report.md"

    subclass_counts = df["cpc_subclass"].value_counts()
    subclass_lines = "\n".join(
        f"- `{sc}`: {cnt} patents" for sc, cnt in subclass_counts.items()
    )

    model_table_rows = "\n".join(
        f"| {m['name']} | `{m['hf_id']}` |" for m in MODELS
    )

    if "status" in final_df.columns:
        results_cols = [
            "model", "status", "embedding_dimension", "load_time_seconds",
            "encoding_time_seconds", "avg_time_per_patent",
            f"same_cpc_hit_rate_at_{TOP_K}", "device_used",
            "mps_fallback_to_cpu",
            "avg_pairwise_similarity_diagnostic",
        ]
        results_cols = [c for c in results_cols if c in final_df.columns]
        results_table_md = dataframe_to_markdown(final_df[results_cols])
    else:
        results_table_md = "(no results)"

    report = f"""# Patent Embedding Model Benchmark

## Dataset
{n_patents} patents, {len(subclass_counts)} CPC subclasses, stratified sample
(3 subclasses x 5 patents, deterministic selection, seed=42):

{subclass_lines}

## Models Tested

| Model | HuggingFace ID |
|---|---|
{model_table_rows}

## Evaluation Method

- Text representation: `title + "\\n\\n" + abstract` for every model
  (full claims/description exceed these models' context limits per the
  project's earlier EDA on description length).
- Each model's documents were encoded with its own document-side prefix
  (e.g. `search_document:` for Nomic, `passage:` for E5, none for the
  others) and L2-normalized.
- Similarity: cosine similarity between all pairs of the {n_patents} embeddings.
- Retrieval: for each patent, the top-{TOP_K} most similar *other* patents
  were retrieved.
- **same_cpc_hit_rate_at_{TOP_K}** (proxy metric): fraction of patents for
  which at least one of the top-{TOP_K} retrieved patents shares the same
  CPC subclass as the query. This is a WEAK PROXY for retrieval quality,
  not a measure of true semantic similarity or prior-art relevance.
- **Chance baseline**: a model with no real signal would still score
  roughly **{CHANCE_HIT_RATE:.4f}** on this metric by chance alone, given
  the class sizes in this {n_patents}-patent sample. Any hit-rate close to
  this baseline should NOT be read as evidence the model is "working."
- `avg/min/max_pairwise_similarity_diagnostic` columns are descriptive only
  and were explicitly NOT used to select a model (different models have
  different baseline embedding-space geometry, making raw average
  similarity incomparable across models).

## Results

{results_table_md}

(Chance-level same_cpc_hit_rate_at_{TOP_K} baseline for this sample: **{CHANCE_HIT_RATE:.4f}**)

## Observations

<!-- Describe the measured differences between models here: which models
were fastest to load/encode, which had the largest gap above the chance
baseline on same_cpc_hit_rate, any that required a CPU fallback, any that
failed and why. Do not declare a winner in this section. -->

## Model Selection

Selected model: **[TO BE FILLED AFTER REVIEW]**

Rationale:
[TO BE FILLED BASED ON RESULTS -- consider: hit-rate above chance baseline,
whether the model is patent-specific vs general-purpose, encoding speed,
memory/resource requirements on the target 8GB machine, embedding
dimension, and implementation complexity (e.g. trust_remote_code
dependency for Nomic). Do not select based on average pairwise similarity.]

## Limitations

- Only {n_patents} patents were evaluated -- this is an initial engineering
  feasibility benchmark, not a statistically powered evaluation.
- CPC subclass co-membership is a proxy for relatedness, not a verified
  ground truth of semantic similarity or prior-art relevance.
- The chance baseline for same_cpc_hit_rate_at_{TOP_K} on this sample is
  {CHANCE_HIT_RATE:.4f} -- small differences between models near this
  baseline are not meaningful with this sample size.
- A larger, ideally manually labeled, retrieval evaluation set is needed
  before drawing strong conclusions about production model choice.
"""

    report_path.write_text(report, encoding="utf-8")
    log(f"\nReport written: {report_path}")

log("\nAll processing finished.")
