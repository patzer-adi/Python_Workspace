"""
Verify character-based chunk sizes and overlap settings
against the actual tokenizer limits of seven patent embedding models.

Reads:
    data/sample_dataset

Writes:
    results/chunk_config_check/chunk_size_tokenizer_report.csv
    results/chunk_config_check/model_tokenizer_summary.json

Does NOT:
    Generate embeddings
    Create or modify ChromaDB databases
    Modify the dataset
    Load model weights
"""

import csv
import json
import math
import statistics
import traceback
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import AutoConfig, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "sample_dataset"
OUTPUT_DIR = ROOT / "results" / "chunk_config_check"

SAMPLE_SIZE = 20

MODELS = {
    "patentsberta": "AI-Growth-Lab/PatentSBERTa",
    "paecter": "mpi-inno-comp/paecter",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "bge": "BAAI/bge-base-en-v1.5",
    "e5": "intfloat/e5-base-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

TEXT_FIELDS = [
    "title",
    "abstract",
    "claims",
    "background",
    "summary",
    "description",
]

# Candidate character limits to test.
CHUNK_SIZES = [800, 1000, 1200, 1400, 1600, 1800]

# Each policy determines overlap relative to chunk size.
OVERLAP_POLICIES = {
    "fixed_200": lambda size: min(200, size - 1),
    "10_percent": lambda size: round(size * 0.10),
    "15_percent": lambda size: round(size * 0.15),
}

# Prefixes are included when counting tokens.
# These are document prefixes, not query prefixes.
DOCUMENT_PREFIXES = {
    "patentsberta": "",
    "paecter": "",
    "nomic": "search_document: ",
    "bge": "",
    "e5": "passage: ",
    "mpnet": "",
    "minilm": "",
}


def text_value(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, tuple)):
        return " ".join(
            str(item).strip()
            for item in value
            if item is not None
        ).strip()

    return str(value).strip()


def combine_patent_text(row):
    """Keep source sections identifiable in the combined text."""
    sections = []

    for field in TEXT_FIELDS:
        value = text_value(row.get(field))

        if value:
            sections.append(
                f"[{field.upper()}]\n{value}"
            )

    return "\n\n".join(sections)


def load_sample():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = load_from_disk(str(DATASET_PATH))
    rows = []

    if isinstance(dataset, DatasetDict):
        # Sample both splits when available.
        split_names = [
            name
            for name in ("train", "validation")
            if name in dataset
        ]

        if not split_names:
            split_names = list(dataset.keys())

        per_split = max(
            1,
            SAMPLE_SIZE // len(split_names)
        )

        for split_name in split_names:
            split = dataset[split_name]
            count = min(len(split), per_split)

            rows.extend(
                split.select(range(count)).to_list()
            )

        # Fill any remaining sample slots.
        if len(rows) < SAMPLE_SIZE:
            for split_name in split_names:
                if len(rows) >= SAMPLE_SIZE:
                    break

                split = dataset[split_name]
                remaining = SAMPLE_SIZE - len(rows)
                count = min(len(split), remaining)

                rows.extend(
                    split.select(range(count)).to_list()
                )
    else:
        count = min(len(dataset), SAMPLE_SIZE)
        rows = dataset.select(range(count)).to_list()

    return rows[:SAMPLE_SIZE]


def make_chunks(text, chunk_size, overlap):
    """
    Character-based chunking.

    Prefer paragraph, sentence, or whitespace boundaries near
    the end of the target window. Fall back to the hard limit.
    """
    if not text:
        return []

    overlap = max(
        0,
        min(overlap, chunk_size - 1)
    )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(
            start + chunk_size,
            text_length
        )

        end = hard_end

        if hard_end < text_length:
            # Search for a natural boundary in the final 20%.
            floor = start + int(chunk_size * 0.80)

            candidates = [
                text.rfind("\n\n", floor, hard_end),
                text.rfind(". ", floor, hard_end),
                text.rfind("; ", floor, hard_end),
                text.rfind(" ", floor, hard_end),
            ]

            boundary = max(candidates)

            if boundary > start:
                if text[boundary:boundary + 2] in (
                    "\n\n", ". ", "; "
                ):
                    end = boundary + 2
                else:
                    end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Overlap ensures the next chunk retains context.
        start = max(start + 1, end - overlap)

    return chunks


def percentile(values, fraction):
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def positive_limit(value):
    if not isinstance(value, int):
        return None

    # Tokenizers sometimes use huge sentinel values to mean
    # that no maximum length is configured.
    if value <= 0 or value > 1_000_000:
        return None

    return value


def inspect_model(slug, model_id, texts):
    print(f"\n[{slug}] {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=False,
    )

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=False,
    )

    tokenizer_limit = positive_limit(
        getattr(tokenizer, "model_max_length", None)
    )

    config_limit = None

    for attribute in (
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "max_sequence_length",
    ):
        value = positive_limit(
            getattr(config, attribute, None)
        )

        if value:
            config_limit = value
            break

    # Use the most conservative available limit.
    limits = [
        value
        for value in (tokenizer_limit, config_limit)
        if value is not None
    ]

    effective_limit = min(limits) if limits else None
    prefix = DOCUMENT_PREFIXES[slug]

    print(f"  Tokenizer limit: {tokenizer_limit}")
    print(f"  Config limit:    {config_limit}")
    print(f"  Effective limit: {effective_limit}")
    print(f"  Document prefix: {prefix!r}")

    report_rows = []

    for chunk_size in CHUNK_SIZES:
        for policy_name, overlap_function in (
            OVERLAP_POLICIES.items()
        ):
            overlap = overlap_function(chunk_size)

            token_counts = []
            total_chunks = 0
            over_limit = 0
            max_chunk_chars = 0

            for text in texts:
                chunks = make_chunks(
                    text,
                    chunk_size,
                    overlap,
                )

                total_chunks += len(chunks)

                for chunk in chunks:
                    # Count actual tokenizer tokens, including
                    # special tokens and the document prefix.
                    encoded = tokenizer(
                        prefix + chunk,
                        add_special_tokens=True,
                        truncation=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )

                    token_count = len(
                        encoded["input_ids"]
                    )

                    token_counts.append(token_count)
                    max_chunk_chars = max(
                        max_chunk_chars,
                        len(chunk),
                    )

                    if (
                        effective_limit is not None
                        and token_count > effective_limit
                    ):
                        over_limit += 1

            if not token_counts:
                continue

            report_rows.append({
                "model_slug": slug,
                "model_id": model_id,
                "tokenizer_limit": tokenizer_limit,
                "config_limit": config_limit,
                "effective_token_limit": effective_limit,
                "document_prefix": prefix,
                "sample_patents": len(texts),
                "chunk_size_chars": chunk_size,
                "overlap_policy": policy_name,
                "overlap_chars": overlap,
                "chunks_tested": total_chunks,
                "mean_tokens": round(
                    statistics.mean(token_counts), 2
                ),
                "p95_tokens": round(
                    percentile(token_counts, 0.95), 2
                ),
                "max_tokens": max(token_counts),
                "max_actual_chunk_chars": max_chunk_chars,
                "chunks_over_limit": over_limit,
                "over_limit_percent": round(
                    100 * over_limit / len(token_counts), 3
                ),
                "safe_for_sample": (
                    over_limit == 0
                    if effective_limit is not None
                    else None
                ),
            })

    print(
        f"  Tested {len(CHUNK_SIZES)} character targets "
        f"and {len(OVERLAP_POLICIES)} overlap policies."
    )

    return report_rows, {
        "model_slug": slug,
        "model_id": model_id,
        "status": "success",
        "tokenizer_limit": tokenizer_limit,
        "config_limit": config_limit,
        "effective_token_limit": effective_limit,
        "document_prefix": prefix,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Patent chunk configuration verification")
    print(f"Project: {ROOT}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output:  {OUTPUT_DIR}")
    print("\nNo model weights or embeddings will be generated.")

    rows = load_sample()

    texts = [
        combine_patent_text(row)
        for row in rows
    ]

    texts = [
        text
        for text in texts
        if text
    ]

    if not texts:
        raise RuntimeError(
            "The sample contains no usable patent text."
        )

    print(f"\nUsable sample patents: {len(texts)}")

    all_report_rows = []
    model_summaries = []

    for slug, model_id in MODELS.items():
        try:
            report_rows, summary = inspect_model(
                slug,
                model_id,
                texts,
            )

            all_report_rows.extend(report_rows)
            model_summaries.append(summary)

        except Exception as error:
            print(
                f"  ERROR: {type(error).__name__}: {error}"
            )

            model_summaries.append({
                "model_slug": slug,
                "model_id": model_id,
                "status": "failed",
                "error": (
                    f"{type(error).__name__}: {error}"
                ),
                "traceback": traceback.format_exc(),
            })

    csv_path = (
        OUTPUT_DIR / "chunk_size_tokenizer_report.csv"
    )

    if all_report_rows:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(all_report_rows[0].keys()),
            )

            writer.writeheader()
            writer.writerows(all_report_rows)

    summary_path = (
        OUTPUT_DIR / "model_tokenizer_summary.json"
    )

    summary_path.write_text(
        json.dumps(model_summaries, indent=2),
        encoding="utf-8",
    )

    print("\nVerification finished.")
    print(f"CSV report: {csv_path}")
    print(f"JSON summary: {summary_path}")

    print(
        "\nInspect p95_tokens, max_tokens, "
        "chunks_over_limit and effective_token_limit."
    )

    print(
        "Important: a sample with zero over-limit chunks "
        "does not prove that every patent will fit."
    )


if __name__ == "__main__":
    main()

