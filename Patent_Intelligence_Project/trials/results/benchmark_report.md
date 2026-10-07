# Patent Embedding Model Benchmark

## Dataset
15 patents, 3 CPC subclasses, stratified sample
(3 subclasses x 5 patents, deterministic selection, seed=42):

- `G06F`: 5 patents
- `H01L`: 5 patents
- `H04L`: 5 patents

## Models Tested

| Model | HuggingFace ID |
|---|---|
| PatentSBERTa | `AI-Growth-Lab/PatentSBERTa` |
| PaECTER | `mpi-inno-comp/paecter` |
| Nomic-Embed-v1.5 | `nomic-ai/nomic-embed-text-v1.5` |
| BGE-base-en-v1.5 | `BAAI/bge-base-en-v1.5` |
| E5-base-v2 | `intfloat/e5-base-v2` |
| all-mpnet-base-v2 | `sentence-transformers/all-mpnet-base-v2` |
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` |

## Evaluation Method

- Text representation: `title + "\n\n" + abstract` for every model
  (full claims/description exceed these models' context limits per the
  project's earlier EDA on description length).
- Each model's documents were encoded with its own document-side prefix
  (e.g. `search_document:` for Nomic, `passage:` for E5, none for the
  others) and L2-normalized.
- Similarity: cosine similarity between all pairs of the 15 embeddings.
- Retrieval: for each patent, the top-3 most similar *other* patents
  were retrieved.
- **same_cpc_hit_rate_at_3** (proxy metric): fraction of patents for
  which at least one of the top-3 retrieved patents shares the same
  CPC subclass as the query. This is a WEAK PROXY for retrieval quality,
  not a measure of true semantic similarity or prior-art relevance.
- **Chance baseline**: a model with no real signal would still score
  roughly **0.6703** on this metric by chance alone, given
  the class sizes in this 15-patent sample. Any hit-rate close to
  this baseline should NOT be read as evidence the model is "working."
- `avg/min/max_pairwise_similarity_diagnostic` columns are descriptive only
  and were explicitly NOT used to select a model (different models have
  different baseline embedding-space geometry, making raw average
  similarity incomparable across models).

## Results

| model | status | embedding_dimension | load_time_seconds | encoding_time_seconds | avg_time_per_patent | same_cpc_hit_rate_at_3 | device_used | mps_fallback_to_cpu | avg_pairwise_similarity_diagnostic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PatentSBERTa | SUCCESS | 768.0 | 87.6064 | 2.5045 | 0.167 | 0.8667 | mps | False | 0.339384 |
| PaECTER | SUCCESS | 1024.0 | 257.5343 | 1.6566 | 0.1104 | 1.0 | mps | False | 0.87086 |
| BGE-base-en-v1.5 | SUCCESS | 768.0 | 41.8802 | 0.3365 | 0.0224 | 0.9333 | mps | False | 0.537 |
| E5-base-v2 | SUCCESS | 768.0 | 59.2454 | 0.4872 | 0.0325 | 0.8667 | mps | False | 0.750778 |
| all-mpnet-base-v2 | SUCCESS | 768.0 | 77.9209 | 0.4425 | 0.0295 | 0.9333 | mps | False | 0.258699 |
| all-MiniLM-L6-v2 | SUCCESS | 384.0 | 23.7415 | 0.5077 | 0.0338 | 0.9333 | mps | False | 0.211325 |
| Nomic-Embed-v1.5 | FAILED | nan | nan | nan | nan | nan | nan | nan | nan |

(Chance-level same_cpc_hit_rate_at_3 baseline for this sample: **0.6703**)

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

- Only 15 patents were evaluated -- this is an initial engineering
  feasibility benchmark, not a statistically powered evaluation.
- CPC subclass co-membership is a proxy for relatedness, not a verified
  ground truth of semantic similarity or prior-art relevance.
- The chance baseline for same_cpc_hit_rate_at_3 on this sample is
  0.6703 -- small differences between models near this
  baseline are not meaningful with this sample size.
- A larger, ideally manually labeled, retrieval evaluation set is needed
  before drawing strong conclusions about production model choice.
