import ast
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

SEED = 42
MAX_POSITIVES = 3

SOURCE = Path("data/hupd_sample/hupd_2016.csv")
OUTPUT = Path("experiments/finetuning/data")
OUTPUT.mkdir(parents=True, exist_ok=True)

print("Loading HUPD dataset...")
df = pd.read_csv(SOURCE)

df["cpc_labels"] = df["cpc_labels"].apply(ast.literal_eval)
df["cpc_labels"] = df["cpc_labels"].apply(lambda x: sorted(set(x)))

df["text"] = (
    "Title: " + df["title"].fillna("").astype(str)
    + "\nAbstract: " + df["abstract"].fillna("").astype(str)
)

# Split patents before generating pairs.
indices = list(range(len(df)))
random.Random(SEED).shuffle(indices)

n = len(indices)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

splits = {
    "train": indices[:train_end],
    "validation": indices[train_end:val_end],
    "test": indices[val_end:],
}

# Save patent split assignments for reproducibility.
split_rows = []
for split_name, split_indices in splits.items():
    for idx in split_indices:
        split_rows.append({
            "application_number": df.iloc[idx]["application_number"],
            "split": split_name,
        })

pd.DataFrame(split_rows).to_csv(
    OUTPUT / "patent_splits.csv", index=False
)

def generate_pairs(split_indices, split_name):
    subset = df.iloc[split_indices].reset_index(drop=True)

    # Build an inverted CPC index.
    cpc_index = defaultdict(set)

    for idx, labels in enumerate(subset["cpc_labels"]):
        for label in labels:
            cpc_index[label].add(idx)

    pairs = set()

    for anchor_idx, anchor in subset.iterrows():
        candidates = set()

        for label in anchor["cpc_labels"]:
            candidates.update(cpc_index[label])

        candidates.discard(anchor_idx)

        scored = []

        for candidate_idx in candidates:
            candidate = subset.iloc[candidate_idx]

            shared = set(anchor["cpc_labels"]) & set(
                candidate["cpc_labels"]
            )
            union = set(anchor["cpc_labels"]) | set(
                candidate["cpc_labels"]
            )

            if not shared:
                continue

            jaccard = len(shared) / len(union)
            scored.append((jaccard, candidate_idx, len(shared)))

        # Prefer patents with stronger CPC overlap.
        scored.sort(reverse=True)

        for jaccard, candidate_idx, shared_count in scored[:MAX_POSITIVES]:
            pair = tuple(sorted((anchor_idx, candidate_idx)))

            if pair in pairs:
                continue

            positive = subset.iloc[candidate_idx]

            pairs.add(pair)

    rows = []

    for anchor_idx, positive_idx in pairs:
        anchor = subset.iloc[anchor_idx]
        positive = subset.iloc[positive_idx]

        shared = set(anchor["cpc_labels"]) & set(
            positive["cpc_labels"]
        )
        union = set(anchor["cpc_labels"]) | set(
            positive["cpc_labels"]
        )

        rows.append({
            "anchor_id": anchor["application_number"],
            "positive_id": positive["application_number"],
            "anchor_text": anchor["text"],
            "positive_text": positive["text"],
            "shared_cpc_count": len(shared),
            "cpc_jaccard": len(shared) / len(union),
        })

    result = pd.DataFrame(rows)
    output_file = OUTPUT / f"{split_name}_pairs.csv"
    result.to_csv(output_file, index=False)

    print(f"{split_name}: {len(subset):,} patents, {len(result):,} pairs")

    return len(result)


print("\nGenerating CPC-informed pairs...")

for name, split_indices in splits.items():
    generate_pairs(split_indices, name)

print("\nPatent splits and pair datasets saved to:", OUTPUT)
print("Original dataset was not modified.")
