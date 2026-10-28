import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

print("Loading Nomic on CPU...")
model = SentenceTransformer(
    MODEL_ID,
    trust_remote_code=True,
    device="cpu",
)

model.max_seq_length = 512

text = "A machine learning system for detecting fraudulent financial transactions."
print("Encoding one sentence...")

with torch.inference_mode():
    embedding = model.encode(
        ["clustering: " + text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

print("SUCCESS")
print("Shape:", embedding.shape)
print("Finite values:", bool(torch.isfinite(torch.as_tensor(embedding)).all()))
print("Vector norm:", float(torch.linalg.vector_norm(torch.as_tensor(embedding[0]))))
print("First 8 values:", embedding[0][:8])
