from pathlib import Path
import json
import traceback

from datasets import load_dataset
from datasets.utils import _dill


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "hupd_2014_2018"

TRAIN_START = "2014-01-01"
TRAIN_END = "2017-12-31"
VAL_START = "2018-01-01"
VAL_END = "2018-12-31"

METADATA_URL = (
    "https://huggingface.co/datasets/HUPD/hupd/"
    "resolve/main/hupd_metadata_2022-02-22.feather"
)


# --------------------------------------------------
# Python 3.14 compatibility fix
# --------------------------------------------------

def compatible_batch_setitems(self, items, obj=None):
    """Support Python 3.14's additional obj argument."""
    if self._legacy_no_dict_keys_sorting:
        if obj is None:
            return super(_dill.Pickler, self)._batch_setitems(items)
        return super(_dill.Pickler, self)._batch_setitems(items, obj)

    # Keep the deterministic dictionary-key ordering used by datasets.
    try:
        items = sorted(items)
    except Exception:
        from datasets.fingerprint import Hasher
        items = sorted(items, key=lambda item: Hasher.hash(item[0]))

    if obj is None:
        return super(_dill.Pickler, self)._batch_setitems(items)

    return super(_dill.Pickler, self)._batch_setitems(items, obj)


_dill.Pickler._batch_setitems = compatible_batch_setitems


# --------------------------------------------------
# Load and save HUPD
# --------------------------------------------------

def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists:\n{OUTPUT_DIR}\n"
            "Inspect it before choosing another output directory. "
            "Nothing has been overwritten."
        )

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    print("Loading HUPD metadata and filtering filing dates...")
    print(f"Training period: {TRAIN_START} to {TRAIN_END}")
    print(f"Validation period: {VAL_START} to {VAL_END}")
    print(f"Output directory: {OUTPUT_DIR}")

    try:
        dataset = load_dataset(
            "HUPD/hupd",
            name="all",
            data_files=METADATA_URL,
            icpr_label=None,
            train_filing_start_date=TRAIN_START,
            train_filing_end_date=TRAIN_END,
            val_filing_start_date=VAL_START,
            val_filing_end_date=VAL_END,
        )

        print("\nDataset loaded successfully:")
        print(dataset)

        dataset.save_to_disk(str(OUTPUT_DIR))

        manifest = {
            "dataset": "HUPD/hupd",
            "training_start": TRAIN_START,
            "training_end": TRAIN_END,
            "validation_start": VAL_START,
            "validation_end": VAL_END,
            "splits": {
                split: len(dataset[split])
                for split in dataset.keys()
            },
        }

        with (OUTPUT_DIR / "manifest.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(manifest, file, indent=2)

        print("\nDataset saved successfully!")
        print(json.dumps(manifest, indent=2))

    except Exception:
        print("\nHUPD preparation failed:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
