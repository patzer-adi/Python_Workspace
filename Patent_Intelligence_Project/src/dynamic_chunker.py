
"""
Dynamic patent chunking pipeline.

- Processes patents one at a time.
- Uses model-specific token budgets and document prefixes.
- Preserves source sections and patent metadata.
- Writes separate JSONL chunk files for each model.
- Does not generate embeddings or modify ChromaDB.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "sample_dataset"
OUTPUT_DIR = ROOT / "results" / "dynamic_chunks"

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
)


@dataclass(frozen=True)
class ModelChunkConfig:
    model_id: str
    hard_limit: int
    target_tokens: int
    overlap_tokens: int
    document_prefix: str = ""


MODEL_CONFIGS = {
    "patentsberta": ModelChunkConfig(
        "AI-Growth-Lab/PatentSBERTa", 512, 450, 50
    ),
    "paecter": ModelChunkConfig(
        "mpi-inno-comp/paecter", 512, 450, 50
    ),
    "nomic": ModelChunkConfig(
        "nomic-ai/nomic-embed-text-v1.5",
        8192, 4096, 300, "search_document: "
    ),
    "bge": ModelChunkConfig(
        "BAAI/bge-base-en-v1.5", 512, 450, 50
    ),
    "e5": ModelChunkConfig(
        "intfloat/e5-base-v2", 512, 450, 50, "passage: "
    ),
    "mpnet": ModelChunkConfig(
        "sentence-transformers/all-mpnet-base-v2", 384, 340, 40
    ),
    "minilm": ModelChunkConfig(
        "sentence-transformers/all-MiniLM-L6-v2", 256, 220, 25
    ),
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(
            str(item) for item in value if item is not None
        ).strip()
    return str(value).strip()


def patent_sections(record: dict) -> Iterator[tuple[str, str]]:
    """Yield nonempty source sections separately."""
    for field in TEXT_FIELDS:
        value = clean_text(record.get(field))
        if value:
            yield field, value


class DynamicPatentChunker:
    def __init__(
        self,
        config: ModelChunkConfig,
        tokenizer=None,
        safety_margin: int = 0,
    ):
        if safety_margin < 0:
            raise ValueError("Safety margin cannot be negative.")
        if config.overlap_tokens >= config.target_tokens:
            raise ValueError("Overlap must be smaller than target.")
        if config.target_tokens > config.hard_limit - safety_margin:
            raise ValueError("Target exceeds the usable token limit.")

        self.config = config
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            config.model_id
        )
        self.usable_limit = config.hard_limit - safety_margin

        # Reserve a little room for any difference between token counting
        # and the actual embedding model's input preparation.
        self.budget = min(config.target_tokens, self.usable_limit)

    def count_tokens(
        self,
        text: str,
        include_prefix: bool = True,
        add_special_tokens: bool = True,
    ) -> int:
        if include_prefix:
            text = self.config.document_prefix + text

        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
            verbose=False,
        )
        return len(encoded["input_ids"])

    def _fits(self, text: str) -> bool:
        return bool(text.strip()) and (
            self.count_tokens(text) <= self.budget
        )

    def _largest_fitting_end(
        self, text: str, start: int
    ) -> int:
        """Find a character end whose actual token count fits the budget."""
        low = start + 1
        high = len(text)
        best = start

        while low <= high:
            mid = (low + high) // 2
            candidate = text[start:mid]

            if self._fits(candidate):
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        if best == start:
            raise ValueError(
                "Could not fit even a single character within the budget."
            )

        return best

    @staticmethod
    def _natural_boundary(
        text: str, start: int, end: int
    ) -> int:
        """Prefer a sentence or whitespace boundary near the chunk end."""
        minimum = start + int((end - start) * 0.85)
        window = text[minimum:end]

        candidates = [
            window.rfind("\n\n"),
            window.rfind(". "),
            window.rfind("? "),
            window.rfind("! "),
            window.rfind("; "),
            window.rfind(" "),
            window.rfind("\n"),
        ]

        boundary = max(candidates)

        if boundary < 0:
            return end

        return minimum + boundary + 1

    def _overlap_start(self, text: str, start: int, end: int) -> int:
        """Choose a character start near the requested token overlap."""
        if self.config.overlap_tokens <= 0:
            return end

        low = start + 1
        high = end - 1
        best = end

        while low <= high:
            mid = (low + high) // 2
            overlap_text = text[mid:end]

            count = self.count_tokens(
                overlap_text,
                include_prefix=False,
                add_special_tokens=False,
            )

            if count <= self.config.overlap_tokens:
                best = mid
                high = mid - 1
            else:
                low = mid + 1

        return best

    def split_section(self, section_text: str) -> Iterator[dict]:
        """
        Yield chunks while retaining original text substrings.

        Character boundaries are selected dynamically according to the
        tokenizer's measured token count. No chunk count is imposed.
        """
        text = section_text
        start = 0

        while start < len(text):
            while start < len(text) and text[start].isspace():
                start += 1

            if start >= len(text):
                break

            end = self._largest_fitting_end(text, start)

            if end < len(text):
                natural_end = self._natural_boundary(text, start, end)
                if natural_end > start and self._fits(text[start:natural_end]):
                    end = natural_end

            chunk_text = text[start:end].strip()

            if not chunk_text:
                raise RuntimeError("The chunker produced an empty chunk.")

            token_count = self.count_tokens(chunk_text)

            if token_count > self.usable_limit:
                raise RuntimeError(
                    f"Token limit exceeded: {token_count} > "
                    f"{self.usable_limit}"
                )

            yield {
                "text": chunk_text,
                "token_count": token_count,
            }

            if end >= len(text):
                break

            next_start = self._overlap_start(text, start, end)

            # Always advance, even if overlap is unusually large.
            start = max(start + 1, next_start)

    def chunk_patent(self, record: dict) -> Iterator[dict]:
        patent_id = str(record.get("patent_number", "unknown"))
        chunk_index = 0

        metadata = {
            key: record.get(key)
            for key in METADATA_FIELDS
            if record.get(key) is not None
        }

        for section, section_text in patent_sections(record):
            for part in self.split_section(section_text):
                text = part["text"]
                actual_count = self.count_tokens(text)

                if actual_count > self.usable_limit:
                    raise RuntimeError(
                        f"Unsafe chunk for patent {patent_id}: "
                        f"{actual_count} tokens"
                    )

                yield {
                    "id": f"{patent_id}:{section}:{chunk_index}",
                    "patent_number": patent_id,
                    "section": section,
                    "chunk_index": chunk_index,
                    "text": text,
                    "document_text": (
                        self.config.document_prefix + text
                    ),
                    "token_count": actual_count,
                    "metadata": metadata,
                }
                chunk_index += 1


def load_splits():
    dataset = load_from_disk(str(DATASET_PATH))

    if isinstance(dataset, DatasetDict):
        return [
            (name, dataset[name])
            for name in ("train", "validation")
            if name in dataset
        ]

    return [("data", dataset)]


def run_model(slug: str, config: ModelChunkConfig):
    print(f"\n{'=' * 60}", flush=True)
    print(f"Model: {slug} | {config.model_id}", flush=True)
    print(
        f"Target={config.target_tokens}, "
        f"overlap={config.overlap_tokens}, "
        f"hard limit={config.hard_limit}",
        flush=True,
    )

    chunker = DynamicPatentChunker(config)
    output_dir = OUTPUT_DIR / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "chunks.jsonl"
    temporary_path = output_dir / "chunks.jsonl.tmp"

    total_patents = 0
    total_chunks = 0
    maximum_tokens = 0

    try:
        with temporary_path.open("w", encoding="utf-8") as out:
            for split_name, dataset in load_splits():
                for row_index in range(len(dataset)):
                    record = dataset[row_index]
                    total_patents += 1

                    for chunk in chunker.chunk_patent(record):
                        chunk["source_split"] = split_name
                        chunk["source_row"] = row_index

                        if chunk["token_count"] > config.hard_limit:
                            raise RuntimeError(
                                f"Hard-limit violation: {chunk['id']}"
                            )

                        out.write(
                            json.dumps(chunk, ensure_ascii=False, default=str)
                            + "\n"
                        )
                        total_chunks += 1
                        maximum_tokens = max(
                            maximum_tokens, chunk["token_count"]
                        )

                    if total_patents % 500 == 0:
                        print(
                            f"{slug}: patents={total_patents:,}, "
                            f"chunks={total_chunks:,}, "
                            f"max_tokens={maximum_tokens}",
                            flush=True,
                        )

        # Publish completed output only after successful processing.
        temporary_path.replace(output_path)

    except Exception:
        # Keep the incomplete output under a temporary name for inspection.
        print(
            f"ERROR in {slug}. Partial output: {temporary_path}",
            flush=True,
        )
        raise

    print(f"Completed model: {slug}", flush=True)
    print(f"Patents processed: {total_patents:,}", flush=True)
    print(f"Chunks generated: {total_chunks:,}", flush=True)
    print(f"Maximum observed tokens: {maximum_tokens}", flush=True)
    print(f"Output: {output_path}", flush=True)


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Dynamic patent chunking pipeline", flush=True)
    print(f"Dataset: {DATASET_PATH}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print("No embeddings or database changes will be made.", flush=True)

    for slug, config in MODEL_CONFIGS.items():
        run_model(slug, config)

    print("\nAll seven models completed.", flush=True)


if __name__ == "__main__":
    main()
