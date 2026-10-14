
#!/usr/bin/env python3
"""
Memory-conscious, resumable preparation of local HUPD data.

Training: filing years 2014–2017
Validation: filing year 2018

The script:
- Uses existing Feather metadata and yearly tar.gz archives.
- Stores metadata, extracted records, and checkpoints in SQLite.
- Does not extract entire archives.
- Commits progress in small transactions.
- Skips completed years when restarted.
- Re-scans an interrupted year's archive but skips records already saved.
- Exports JSONL files after processing all years.

Existing project datasets and ChromaDB are not modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import sys
import tarfile
from pathlib import Path

import pyarrow.feather as feather


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = Path.home() / (
    ".cache/huggingface/hub/datasets--HUPD--hupd/"
    "snapshots/f570a84b03663180b6034c1f7f4c15864f94385e/"
    "hupd_metadata_2022-02-22.feather"
)

ARCHIVE_DIR = Path.home() / (
    ".cache/huggingface/datasets/downloads/extracted/"
    "71f7bbd1bd7065878913762ac320dc984b63ecaca"
    "786495a835d363914dd6458/data"
)

OUT_DIR = ROOT / "data" / "hupd_2014_2018"
DB_PATH = OUT_DIR / "checkpoint.sqlite3"

# Only load metadata needed for the final records.
META_COLUMNS = [
    "application_number",
    "filing_date",
    "patent_number",
    "patent_issue_date",
    "invention_title",
    "main_cpc_label",
    "main_ipcr_label",
    "date_application_published",
    "earliest_pgpub_date",
    "decision",
    "examiner_full_name",
]

TEXT_FIELDS = (
    "title",
    "abstract",
    "claims",
    "background",
    "summary",
    "description",
)

BATCH_SIZE = 1000
METADATA_BATCH_SIZE = 25_000


def connect_db() -> sqlite3.Connection:
    """Create or open the on-disk database."""
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-32768")  # About 32 MiB
    con.execute("PRAGMA temp_store=FILE")

    con.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (
            year INTEGER NOT NULL,
            app TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            PRIMARY KEY (year, app)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS records (
            year INTEGER NOT NULL,
            app TEXT NOT NULL,
            split TEXT NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY (year, app)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    con.commit()
    return con


def state_get(con: sqlite3.Connection, key: str):
    row = con.execute(
        "SELECT value FROM state WHERE key=?", (key,)
    ).fetchone()
    return row[0] if row else None


def state_set(con: sqlite3.Connection, key: str, value: str):
    con.execute(
        "INSERT INTO state(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def json_safe(value):
    """Normalize common Arrow/Python values for JSON serialization."""
    if value is None:
        return None
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return value


def prepare_metadata(con: sqlite3.Connection) -> None:
    """Import only required metadata columns, in resumable batches."""
    if state_get(con, "metadata_loaded") == "yes":
        print("Metadata already indexed in checkpoint database.", flush=True)
        return

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata file missing: {METADATA_PATH}")

    print("Reading selected metadata columns from Feather...", flush=True)

    # Memory-map the source where supported. Only selected columns are read.
    table = feather.read_table(
        METADATA_PATH,
        columns=META_COLUMNS,
        memory_map=True,
    )

    print(f"Metadata rows in source: {table.num_rows:,}", flush=True)

    inserted = 0
    skipped = 0

    # If interrupted, INSERT OR IGNORE makes re-importing safe.
    for batch in table.to_batches(max_chunksize=METADATA_BATCH_SIZE):
        columns = batch.to_pydict()
        rows = []

        for i in range(batch.num_rows):
            app_value = columns["application_number"][i]
            filing_date = columns["filing_date"][i]

            if app_value is None or filing_date is None:
                skipped += 1
                continue

            app = str(app_value).strip()
            date_text = str(filing_date)[:10]

            if not app.isdigit():
                skipped += 1
                continue

            try:
                year = int(date_text[:4])
            except (ValueError, TypeError):
                skipped += 1
                continue

            if year not in range(2014, 2019):
                continue

            meta = {
                name: json_safe(columns[name][i])
                for name in META_COLUMNS
            }
            meta["application_number"] = app
            meta["filing_date"] = date_text

            rows.append((
                year,
                app,
                json.dumps(meta, ensure_ascii=False),
            ))

        con.executemany(
            "INSERT OR IGNORE INTO metadata(year,app,meta_json) "
            "VALUES(?,?,?)",
            rows,
        )
        con.commit()
        inserted += len(rows)

        if inserted and inserted % 100_000 < METADATA_BATCH_SIZE:
            print(
                f"  Metadata rows processed: {inserted:,}",
                flush=True,
            )

        del rows, columns
        gc.collect()

    del table
    gc.collect()

    state_set(con, "metadata_loaded", "yes")
    con.commit()

    print(
        f"Metadata indexing finished. Processed rows: {inserted:,}; "
        f"invalid rows skipped: {skipped:,}",
        flush=True,
    )

    for year in range(2014, 2019):
        count = con.execute(
            "SELECT COUNT(*) FROM metadata WHERE year=?", (year,)
        ).fetchone()[0]
        print(f"  {year}: {count:,} metadata IDs", flush=True)


def normalize(raw: dict, meta: dict, split: str) -> dict:
    """Map raw HUPD JSON and metadata to consistent output fields."""
    record = {
        "patent_number": (
            raw.get("patent_number") or meta.get("patent_number")
        ),
        "application_number": str(meta["application_number"]),
        "decision": raw.get("decision") or meta.get("decision"),
        "title": raw.get("title") or meta.get("invention_title") or "",
        "abstract": raw.get("abstract") or "",
        "claims": raw.get("claims") or "",
        "background": raw.get("background") or "",
        "summary": raw.get("summary") or "",
        "description": (
            raw.get("full_description")
            or raw.get("description")
            or ""
        ),
        "cpc_label": (
            raw.get("main_cpc_label") or meta.get("main_cpc_label")
        ),
        "ipc_label": (
            raw.get("main_ipcr_label") or meta.get("main_ipcr_label")
        ),
        "filing_date": raw.get("filing_date") or meta.get("filing_date"),
        "patent_issue_date": (
            raw.get("patent_issue_date") or meta.get("patent_issue_date")
        ),
        "date_published": (
            raw.get("date_published")
            or meta.get("date_application_published")
            or meta.get("earliest_pgpub_date")
        ),
        "examiner_id": (
            raw.get("examiner_id") or meta.get("examiner_full_name")
        ),
        "split": split,
    }

    return record


def process_year(con: sqlite3.Connection, year: int) -> None:
    """Scan one archive; committed records survive an interruption."""
    if state_get(con, f"year_{year}_complete") == "yes":
        # Avoid a full COUNT(*) scan of the large checkpoint database.
        print(f"\nSkipping completed {year}.", flush=True)
        return

    archive_path = ARCHIVE_DIR / f"{year}.tar.gz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive missing: {archive_path}")

    print(f"\nScanning {archive_path.name}...", flush=True)

    split = "validation" if year == 2018 else "train"
    matched = 0
    scanned = 0
    pending = 0

    # Streaming mode: do not build a list of archive members.
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            scanned += 1

            if scanned % 500_000 == 0:
                print(
                    f"  {year}: scanned {scanned:,} archive members; "
                    f"saved {matched:,} new records",
                    flush=True,
                )

            if not member.isfile() or not member.name.endswith(".json"):
                continue

            member_name = member.name.lstrip("./")
            prefix = f"{year}/"
            if not member_name.startswith(prefix):
                continue

            app = member_name.rsplit("/", 1)[-1][:-5]
            if not app.isdigit():
                continue

            # Check for an expected metadata ID and skip already-saved rows.
            row = con.execute(
                "SELECT meta_json FROM metadata WHERE year=? AND app=?",
                (year, app),
            ).fetchone()

            if row is None:
                continue

            already_saved = con.execute(
                "SELECT 1 FROM records WHERE year=? AND app=?",
                (year, app),
            ).fetchone()

            if already_saved:
                continue

            file_object = archive.extractfile(member)
            if file_object is None:
                continue

            try:
                with file_object:
                    raw = json.load(file_object)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue

            if not isinstance(raw, dict):
                continue

            meta = json.loads(row[0])
            record = normalize(raw, meta, split)

            if not any(
                str(record.get(field) or "").strip()
                for field in TEXT_FIELDS
            ):
                continue

            con.execute(
                "INSERT OR IGNORE INTO records(year,app,split,record_json) "
                "VALUES(?,?,?,?)",
                (
                    year,
                    app,
                    split,
                    json.dumps(record, ensure_ascii=False),
                ),
            )

            matched += 1
            pending += 1

            # Commit frequently; SQLite is the durable checkpoint.
            if pending >= BATCH_SIZE:
                con.commit()
                pending = 0
                print(
                    f"  {year}: {matched:,} new records committed",
                    flush=True,
                )

    con.commit()

    state_set(con, f"year_{year}_complete", "yes")
    con.commit()

    total = con.execute(
        "SELECT COUNT(*) FROM records WHERE year=?", (year,)
    ).fetchone()[0]

    print(
        f"Finished {year}: {total:,} records saved; "
        f"{scanned:,} archive members scanned.",
        flush=True,
    )


def export_jsonl(con: sqlite3.Connection) -> None:
    """Export saved SQLite records to separate JSONL files."""
    train_path = OUT_DIR / "train.jsonl"
    validation_path = OUT_DIR / "validation.jsonl"

    print("\nExporting saved records to JSONL...", flush=True)

    for split, path in (
        ("train", train_path),
        ("validation", validation_path),
    ):
        count = 0

        # Rebuild the export from the durable database, so interrupted
        # exports can safely be repeated without duplicate lines.
        with path.open("w", encoding="utf-8") as output:
            cursor = con.execute(
                "SELECT record_json FROM records "
                "WHERE split=? ORDER BY year,app",
                (split,),
            )

            for (record_json,) in cursor:
                output.write(record_json + "\n")
                count += 1

                if count % 50_000 == 0:
                    print(
                        f"  Exported {split}: {count:,}",
                        flush=True,
                    )

    stats = {
        "source": "Existing local HUPD metadata and archives",
        "training_years": [2014, 2015, 2016, 2017],
        "validation_years": [2018],
        "records_by_year": {},
        "train_records": 0,
        "validation_records": 0,
        "outputs": {
            "train": "data/hupd_2014_2018/train.jsonl",
            "validation": "data/hupd_2014_2018/validation.jsonl",
            "checkpoint_database": "data/hupd_2014_2018/checkpoint.sqlite3",
        },
    }

    for year in range(2014, 2019):
        count = con.execute(
            "SELECT COUNT(*) FROM records WHERE year=?", (year,)
        ).fetchone()[0]
        stats["records_by_year"][str(year)] = count

    stats["train_records"] = con.execute(
        "SELECT COUNT(*) FROM records WHERE split='train'"
    ).fetchone()[0]
    stats["validation_records"] = con.execute(
        "SELECT COUNT(*) FROM records WHERE split='validation'"
    ).fetchone()[0]

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nFinal counts:", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete this script's checkpoint database and its WAL files, "
            "then restart from scratch. Does not delete source archives."
        ),
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata missing: {METADATA_PATH}")
    if not ARCHIVE_DIR.is_dir():
        raise FileNotFoundError(f"Archive directory missing: {ARCHIVE_DIR}")

    if args.reset:
        for path in (
            DB_PATH,
            Path(str(DB_PATH) + "-wal"),
            Path(str(DB_PATH) + "-shm"),
        ):
            if path.exists():
                path.unlink()
        for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
            path = OUT_DIR / name
            if path.exists():
                path.unlink()
        print("This script's outputs and checkpoints reset.", flush=True)

    con = connect_db()

    try:
        prepare_metadata(con)

        for year in range(2014, 2019):
            process_year(con, year)

        export_jsonl(con)

    except KeyboardInterrupt:
        con.commit()
        print(
            "\nInterrupted. Committed records remain in the checkpoint DB.\n"
            "Run the same command again to continue.",
            file=sys.stderr,
        )
        return 130
    finally:
        con.close()

    print("\nAll years processed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
