import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)

SEED = 42
BATCH_SIZE = 8
EPOCHS = 1
MAX_TRAIN_PAIRS = 5000
MODEL_NAME = "BAAI/bge-base-en-v1.5"

DATA_DIR = Path("experiments/finetuning/data")
OUTPUT_DIR = Path(
    "experiments/finetuning/models/bge-base-en-v1.5-cpc-mnrl-pilot"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Training device:", device)

print("Loading training pairs...")
df = pd.read_csv(DATA_DIR / "train_pairs.csv")

df = df.sample(
    n=min(MAX_TRAIN_PAIRS, len(df)),
    random_state=SEED,
).reset_index(drop=True)

examples = []
patent_ids = []

for row in df.itertuples(index=False):
    examples.append(
        InputExample(
            texts=[row.anchor_text, row.positive_text]
        )
    )
    patent_ids.append((row.anchor_id, row.positive_id))


class UniquePatentBatchSampler(Sampler):
    """Prevent repeated patent IDs within an MNRL batch."""

    def __init__(self, patent_ids, batch_size, seed=42):
        self.patent_ids = patent_ids
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def make_batches(self, epoch):
        indices = list(range(len(self.patent_ids)))
        random.Random(self.seed + epoch).shuffle(indices)

        batches = []
        batch = []
        used_ids = set()

        for idx in indices:
            a, b = self.patent_ids[idx]

            if a in used_ids or b in used_ids:
                if batch:
                    batches.append(batch)
                batch = []
                used_ids = set()

            batch.append(idx)
            used_ids.update((a, b))

            if len(batch) >= self.batch_size:
                batches.append(batch)
                batch = []
                used_ids = set()

        if batch:
            batches.append(batch)

        return batches

    def __iter__(self):
        batches = self.make_batches(self.epoch)
        self.epoch += 1
        yield from batches

    def __len__(self):
        return len(self.make_batches(self.epoch))


sampler = UniquePatentBatchSampler(
    patent_ids,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

print("Loading pretrained model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME, device=device)
model.max_seq_length = 256

train_loader = DataLoader(
    examples,
    batch_sampler=sampler,
    collate_fn=model.smart_batching_collate,
)

print("Training examples:", len(examples))
print("Training batches:", len(train_loader))
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)

loss = MultipleNegativesRankingLoss(model)

warmup_steps = max(10, int(len(train_loader) * EPOCHS * 0.1))

print("Warmup steps:", warmup_steps)
print("Starting fine-tuning...")

model.fit(
    train_objectives=[(train_loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": 2e-5},
    weight_decay=0.01,
    output_path=str(OUTPUT_DIR),
    save_best_model=False,
    use_amp=False,
    show_progress_bar=True,
    checkpoint_path=str(OUTPUT_DIR / "checkpoints"),
    checkpoint_save_steps=500,
    checkpoint_save_total_limit=2,
)

print("\nFine-tuning completed.")
print("Model saved to:", OUTPUT_DIR)
