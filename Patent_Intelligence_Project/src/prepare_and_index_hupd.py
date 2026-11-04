#!/usr/bin/env python3
"""
Prepare the existing HUPD sample_dataset and index it in a NEW ChromaDB
collection. The source dataset and existing collections are never deleted.

Run from the project root:
    ./.Patent_Intelligence_virtual_env_final/bin/python src/prepare_and_index_hupd.py

Optional:
    .../python src/prepare_and_index_hupd.py --batch-size 16 --chunk-size 1800
    .../python src/prepare_and_index_hupd.py --limit 100

The script is resumable: deterministic chunk IDs and collection.upsert() mean
re-running it updates the same chunks rather than duplicating them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "data" / "sample_dataset"
DEFAULT_DB_DIR = PROJECT_DIR / "data" / "chroma_db"
DEFAULT_COLLECTION = "patent_sample_2016_chunks"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Prefer the richer text fields, while gracefully skipping fields absent in
# a particular version of the dataset.
TEXT_FIELDS = (
    "title",
    "abstract",
    "claims",
    "background",
    "summary",
    "description",
)
METADATA_FIELDS = (
    "patent_number",
    "decision",
    "filing_date",
    "patent_issue_date",
    "date_published",
    "cpc_label",
    "ipc_label",
    "examiner_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, chunk, embed, and index an on-disk HUPD DatasetDict."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records per split, useful for a smoke test.",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="DANGER: delete and recreate ONLY the named target collection.",
    )
    return parser.parse_args()


def as_text(value: Any) -> str:
    """Convert scalar/list-like dataset values to readable text safely."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(as_text(item) for item in value if item is not None).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    try:
        # Handles NaN-like values without requiring pandas.
        if value != value:
            return ""
    except Exception:
        pass
    return str(value).strip()


def metadata_value(value: Any) -> str | int | float | bool:
    """Chroma metadata must contain simple scalar values, not lists/dicts."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        try:
            if isinstance(value, float) and value != value:
                return ""
        except Exception:
            pass
        return value
    return as_text(value)


def make_document(record: dict[str, Any], available_fields: set[str]) -> str:
    sections = []
    for field in TEXT_FIELDS:
        if field not in available_fields:
            continue
        value = as_text(record.get(field))
        if value:
            sections.append(f"{field.upper()}:\n{value}")
    return "\n\n".join(sections).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Create overlapping character chunks, preferring a nearby word boundary."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            # Prefer a paragraph, sentence, or word boundary in the final
            # quarter of the chunk; otherwise use the hard boundary.
            lower_bound = start + int(chunk_size * 0.75)
            candidates = [
                text.rfind("\n\n", lower_bound, end),
                text.rfind(". ", lower_bound, end),
                text.rfind(" ", lower_bound, end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if text[boundary:boundary + 2] == "\n\n" else 1)

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return chunks


def stable_chunk_id(
    split_name: str, patent_number: str, row_index: int, chunk_index: int, chunk: str
) -> str:
    # Include split and row index to avoid collapsing duplicate patent numbers
    # or records across splits. The same run produces the same IDs.
    raw = (
        f"{split_name}\0{patent_number}\0{row_index}\0{chunk_index}\0"
        f"{hashlib.sha256(chunk.encode('utf-8')).hexdigest()}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def open_target_collection(db_dir: Path, name: str, reset: bool):
    from chromadb import PersistentClient

    db_dir.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(db_dir))
    names = {item.name for item in client.list_collections()}
    if reset and name in names:
        print(
            f"WARNING: deleting only requested target collection {name!r}.",
            flush=True,
        )
        client.delete_collection(name=name)
        names.remove(name)

    if name in names:
        collection = client.get_collection(name=name)
        print(f"Resuming existing collection: {name} (current chunks: {collection.count()})")
    else:
        collection = client.create_collection(
            name=name,
            metadata={
                "description": "HUPD sample_dataset chunk embeddings",
                "embedding_model": DEFAULT_MODEL,
                "hnsw:space": "cosine",
            },
            # Chroma uses the hnsw:space key to configure cosine distance.
            # Collection metadata is fixed when the collection is created.
        )
        print(f"Created new collection: {name}")
    return client, collection


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    db_dir = args.db_dir.expanduser().resolve()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be >= 0 and smaller than --chunk-size")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Expected the existing data/sample_dataset directory."
        )

    from datasets import load_from_disk
    from sentence_transformers import SentenceTransformer

    print("=" * 72)
    print("HUPD SAMPLE PREPARATION + CHROMADB INDEXING")
    print(f"Dataset:    {dataset_dir}")
    print(f"Chroma DB:  {db_dir}")
    print(f"Collection: {args.collection}")
    print(f"Model:      {args.model}")
    print("=" * 72)

    dataset_dict = load_from_disk(str(dataset_dir))
    split_names = list(dataset_dict.keys())
    if not split_names:
        raise ValueError("The loaded DatasetDict contains no splits.")

    for split_name in split_names:
        split = dataset_dict[split_name]
        print(f"Split {split_name!r}: {len(split):,} records; columns={split.column_names}")

    # Union of available columns across splits; fields are checked per record.
    available_fields = {
        field
        for split_name in split_names
        for field in dataset_dict[split_name].column_names
    }
    if not any(field in available_fields for field in TEXT_FIELDS):
        raise ValueError(
            f"No recognized text columns found. Expected one or more of {TEXT_FIELDS}; "
            f"found {sorted(available_fields)}"
        )

    print("\nLoading embedding model (CPU-friendly default batch size is 16)...")
    model = SentenceTransformer(args.model, device="cpu")
    dimension = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dimension}")

    client, collection = open_target_collection(
        db_dir, args.collection, args.reset_collection
    )
    # Record model/dataset information in a sidecar manifest. Keep a local
    # manifest rather than assuming arbitrary existing collection metadata
    # can be modified in place.
    manifest_path = db_dir.parent / f"{args.collection}_manifest.json"
    started = time.time()
    stats: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "chroma_db_dir": str(db_dir),
        "collection": args.collection,
        "embedding_model": args.model,
        "embedding_dimension": dimension,
        "chunk_size_characters": args.chunk_size,
        "chunk_overlap_characters": args.chunk_overlap,
        "split_stats": {},
        "errors": [],
    }

    for split_name in split_names:
        split = dataset_dict[split_name]
        n_records = len(split)
        if args.limit is not None:
            n_records = min(n_records, args.limit)

        print(f"\nProcessing split {split_name!r}: {n_records:,} records")
        split_stats = {
            "records_considered": n_records,
            "records_with_text": 0,
            "records_without_text": 0,
            "chunks_upserted": 0,
            "errors": 0,
        }

        pending_ids: list[str] = []
        pending_docs: list[str] = []
        pending_meta: list[dict[str, Any]] = []

        def flush_batch() -> None:
            if not pending_docs:
                return
            embeddings = model.encode(
                pending_docs,
                batch_size=min(args.batch_size, len(pending_docs)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            collection.upsert(
                ids=pending_ids,
                documents=pending_docs,
                metadatas=pending_meta,
                embeddings=embeddings.tolist(),
            )
            split_stats["chunks_upserted"] += len(pending_docs)
            pending_ids.clear()
            pending_docs.clear()
            pending_meta.clear()

        for row_index in range(n_records):
            try:
                record = split[row_index]
                patent_number = as_text(record.get("patent_number"))
                if not patent_number:
                    # Keep rows without patent numbers identifiable and traceable.
                    patent_number = f"missing-id-{row_index}"

                document = make_document(record, set(split.column_names))
                if not document:
                    split_stats["records_without_text"] += 1
                    continue
                split_stats["records_with_text"] += 1

                chunks = chunk_text(document, args.chunk_size, args.chunk_overlap)
                base_metadata = {
                    key: metadata_value(record.get(key))
                    for key in METADATA_FIELDS
                    if key in split.column_names
                }
                base_metadata.update(
                    {
                        "split": str(split_name),
                        "row_index": int(row_index),
                        "patent_number": patent_number,
                        "source_dataset": dataset_dir.name,
                        "embedding_model": args.model,
                    }
                )

                for chunk_index, chunk in enumerate(chunks):
                    chunk_id = stable_chunk_id(
                        split_name, patent_number, row_index, chunk_index, chunk
                    )
                    metadata = dict(base_metadata)
                    metadata["chunk_index"] = int(chunk_index)
                    metadata["chunk_count_for_record"] = int(len(chunks))
                    pending_ids.append(chunk_id)
                    pending_docs.append(chunk)
                    pending_meta.append(metadata)

                    if len(pending_docs) >= args.batch_size:
                        flush_batch()

                if (row_index + 1) % 500 == 0 or row_index + 1 == n_records:
                    flush_batch()
                    print(
                        f"  {row_index + 1:,}/{n_records:,} records | "
                        f"{split_stats['chunks_upserted']:,} chunks written in this run",
                        flush=True,
                    )
            except Exception as exc:
                split_stats["errors"] += 1
                message = f"{split_name}[{row_index}]: {type(exc).__name__}: {exc}"
                stats["errors"].append(message)
                print(f"  [WARN] {message}", file=sys.stderr, flush=True)

        flush_batch()
        stats["split_stats"][split_name] = split_stats
        print(f"Finished split {split_name!r}: {split_stats}")

        stats["collection_count_after_split"] = int(collection.count())
        manifest_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    stats["collection_count_final"] = int(collection.count())
    stats["elapsed_seconds"] = round(time.time() - started, 2)
    stats["completed_at_unix"] = time.time()
    manifest_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("INDEXING FINISHED")
    print(f"Collection:       {args.collection}")
    print(f"Chunks in index:  {collection.count():,}")
    print(f"Embedding model:  {args.model}")
    print(f"Vector dimension: {dimension}")
    print(f"Manifest:        {manifest_path}")
    print(f"Elapsed:          {stats['elapsed_seconds']:.1f} seconds")
    if stats["errors"]:
        print(f"Warnings/errors:  {len(stats['errors'])} (see manifest)")
    print("The source dataset and other ChromaDB collections were not deleted.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
