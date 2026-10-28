"""
Build a small stratified HUPD benchmark: 3 CPC subclasses x 5 patents each.

WHY STRATIFIED, AND HOW SUBCLASSES/PATENTS ARE CHOSEN (deterministic):

1. Scan sample/2016/*.json, keep records with non-empty title AND abstract
   AND a main_cpc_label.
2. Derive a CPC "subclass" key by taking the first 4 characters of
   main_cpc_label (e.g. "G06F30416" -> "G06F"). This matches CPC's
   section+class+subclass structure.
3. Group valid records by subclass. Keep only subclasses with
   >= PATENTS_PER_SUBCLASS valid records.
4. Choose the N_SUBCLASSES subclasses with the MOST valid records
   (ties broken alphabetically). This is fully deterministic -- no
   randomness in *which* subclasses get picked -- and favors data-rich,
   less-idiosyncratic subclasses over ones that barely clear the minimum.
5. Within each chosen subclass, sample exactly PATENTS_PER_SUBCLASS records
   using random.Random(SEED).sample(...), so *which* patents get picked
   is reproducible across reruns.

This stratification exists so that "top-3 retrieval" has a chance of
meaning something: each patent now has same-subclass siblings to be
retrieved against, instead of being compared to 14 random, likely
unrelated, patents. CPC subclass co-membership is used downstream only
as a weak, proxy ground-truth signal for retrieval evaluation -- it is
NOT a claim of true semantic similarity or prior-art relevance.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
N_SUBCLASSES = 3
PATENTS_PER_SUBCLASS = 5


def load_patent_record(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    title = (data.get("title") or "").strip()
    abstract = (data.get("abstract") or "").strip()
    main_cpc = (data.get("main_cpc_label") or "").strip()

    if not title or not abstract or not main_cpc:
        return None

    return {
        "application_number": data.get("application_number", ""),
        "publication_number": data.get("publication_number", ""),
        "patent_number": data.get("patent_number", ""),
        "title": title,
        "abstract": abstract,
        "main_cpc_label": main_cpc,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("sample/2016"),
        help="Directory containing the extracted HUPD JSON files (default: sample/2016).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/hupd_sample/patent_sample_15_stratified.csv"),
        help="Where to write the resulting CSV.",
    )
    args = parser.parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(
            f"Source directory not found: {args.source_dir}\n"
            "Pass the correct path with --source-dir, e.g.:\n"
            "  python build_stratified_dataset.py --source-dir /path/to/sample/2016"
        )

    json_files = sorted(args.source_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found under {args.source_dir}")

    print(f"Scanning {len(json_files)} JSON files under {args.source_dir} ...")

    by_subclass = defaultdict(list)
    n_valid = 0

    for jp in json_files:
        record = load_patent_record(jp)
        if record is None:
            continue
        subclass = record["main_cpc_label"][:4]
        if len(subclass) < 4:
            continue
        by_subclass[subclass].append(record)
        n_valid += 1

    print(f"Valid records (non-empty title+abstract+CPC): {n_valid}")
    print(f"Distinct CPC subclasses seen: {len(by_subclass)}")

    eligible = {
        subclass: records
        for subclass, records in by_subclass.items()
        if len(records) >= PATENTS_PER_SUBCLASS
    }

    if len(eligible) < N_SUBCLASSES:
        raise ValueError(
            f"Only {len(eligible)} subclasses have >= {PATENTS_PER_SUBCLASS} "
            f"valid patents; need at least {N_SUBCLASSES}. "
            "Lower PATENTS_PER_SUBCLASS, point --source-dir at a larger sample, "
            "or accept fewer subclasses."
        )

    # Deterministic pick: most data-rich subclasses first, ties broken
    # alphabetically by subclass code.
    ranked = sorted(eligible.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    chosen_subclasses = ranked[:N_SUBCLASSES]

    print("\nChosen CPC subclasses (most data-rich, ties broken alphabetically):")
    for subclass, records in chosen_subclasses:
        print(f"  {subclass}: {len(records)} valid candidates")

    rng = random.Random(SEED)
    selected_rows = []
    for subclass, records in chosen_subclasses:
        sample = rng.sample(records, PATENTS_PER_SUBCLASS)
        selected_rows.extend(sample)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "application_number",
        "publication_number",
        "patent_number",
        "title",
        "abstract",
        "main_cpc_label",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    print(f"\nWrote {len(selected_rows)} patents to {args.out_csv}")
    print(f"Random seed used for within-subclass sampling: {SEED}")
    print("Per-subclass counts in output:")
    for subclass, _ in chosen_subclasses:
        print(f"  {subclass}: {PATENTS_PER_SUBCLASS}")


if __name__ == "__main__":
    main()
