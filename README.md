#Patent Intelligence: Semantic Search, Retrieval & RAG

An end-to-end patent intelligence pipeline that combines transformer-based embeddings, domain-specific fine-tuning, vector search, and Retrieval-Augmented Generation (RAG) to make patent discovery and analysis more effective.

The project explores how semantic representation learning can improve patent retrieval beyond traditional keyword-based search.

Project Overview

Patent documents contain technical information expressed through specialised terminology, lengthy descriptions, claims, and abstracts. Finding relevant patents using keyword matching alone can be challenging.

This project develops a pipeline to:

* Process and analyse patent datasets.
* Generate semantic embeddings using multiple pretrained models.
* Fine-tune BGE models using weak CPC-based supervision.
* Compare embedding models through retrieval benchmarks.
* Perform semantic retrieval using FAISS and ChromaDB.
* Explore RAG-based patent analysis and summarisation.
* Provide an interactive interface through Streamlit.
* Experiment with locally hosted and API-based language models.

Architecture

Patent Dataset
      |
      v
Data Processing & Cleaning
      |
      v
Patent Text Preparation
(Title + Abstract + Description)
      |
      v
Embedding Generation
      |
      +-----------------------+
      |                       |
      v                       v
Pretrained Models       Fine-tuned BGE Models
      |                       |
      +-----------+-----------+
                  |
                  v
          Vector Retrieval
          (FAISS / ChromaDB)
                  |
                  v
          Retrieved Patents
                  |
                  v
          RAG / LLM Analysis
                  |
                  v
          Streamlit Interface

Dataset

The project uses the Harvard USPTO Patent Dataset (HUPD).

Dataset processing included:

* Processing 26,808 patent JSON records.
* Creating a structured CSV dataset.
* Extracting patent titles, abstracts, descriptions, and CPC classifications.
* Analysing CPC subclass distributions.
* Preparing patent text for embedding generation and retrieval experiments.

Some experiments use smaller subsets to reduce computational requirements.

Embedding Models

Multiple pretrained embedding models were evaluated to understand their effectiveness for patent semantic retrieval.

Model	Description
PatentSBERTa	Patent-domain sentence embeddings
PaECTER	Patent similarity and retrieval
BGE-base-en-v1.5	General-purpose embedding model
E5-base-v2	General-purpose text embeddings
all-mpnet-base-v2	Sentence-level semantic embeddings
all-MiniLM-L6-v2	Lightweight sentence embeddings
Nomic Embed v1.5	General-purpose embedding model

Fine-tuned BGE Models

Three BGE-based models were fine-tuned using patent data and weak supervision derived from CPC classifications.

The models explore different training objectives:

1. BGE Patent MNRL

Fine-tuned using Multiple Negatives Ranking Loss (MNRL).

View model on Hugging Face

2. BGE Similarity Contrastive

Fine-tuned using a similarity-based contrastive learning approach.

View model on Hugging Face

3. BGE Hybrid Contrastive

Fine-tuned using a hybrid contrastive training approach.

View model on Hugging Face

Loading the fine-tuned models

The models are publicly hosted on Hugging Face. Their weights are maintained separately from this GitHub repository to avoid committing large binary files.

Install the required library:

pip install sentence-transformers

Load a model:

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(
    "patzer-adi/bge-patent-mnrl"
)
texts = [
    "A method for efficient neural network computation",
    "A system for semiconductor device fabrication"
]
embeddings = model.encode(
    texts,
    normalize_embeddings=True
)
print(embeddings.shape)

To use the other models, replace the model identifier with:

"patzer-adi/bge-patent-similarity-contrastive"

or

"patzer-adi/bge-patent-hybrid-contrastive"

Retrieval and Vector Search

The project explores semantic retrieval using vector databases and similarity search.

FAISS

* Vector indexing and similarity search.
* Efficient retrieval of nearest patent embeddings.

ChromaDB

* Persistent vector storage.
* Document and metadata management.
* Semantic retrieval.

The retrieval pipeline uses patent embeddings to identify semantically similar documents.

Additional retrieval experiments include reranking and Reciprocal Rank Fusion (RRF).

Benchmarking

A full-corpus benchmarking pipeline was developed to compare pretrained and fine-tuned embedding models.

The initial benchmark smoke test used:

Parameter	Value
Source patent records	25,247
Patents benchmarked	1,500
Maximum sequence length	256
Text representation	Title + Abstract
Successful models	9
Failed models	1

Preliminary results

Results below are from the 1,500-patent smoke test, not a completed full-corpus evaluation.

Model	Hit@20	Precision@20	Recall@20	MRR@20	nDCG@20
PatentSBERTa	0.7520	0.1799	0.2066	0.4563	0.2789
PaECTER	0.7993	0.2212	0.2679	0.5074	0.3430
BGE-base-en-v1.5	0.7680	0.1943	0.2216	0.4642	0.2971
E5-base-v2	0.7727	0.1993	0.2349	0.4791	0.3062
all-mpnet-base-v2	0.7873	0.2112	0.2497	0.4881	0.3274
all-MiniLM-L6-v2	0.7600	0.1986	0.2219	0.4605	0.2978
BGE Patent MNRL	0.8287	0.2711	0.3323	0.5554	0.4171
BGE Similarity Contrastive	0.7740	0.2443	0.2764	0.5001	0.3621
BGE Hybrid Contrastive	0.7853	0.2451	0.2743	0.4982	0.3554

The BGE Patent MNRL model achieved the strongest results across the reported metrics in this preliminary benchmark.

Evaluation methodology and limitations

* CPC subclass overlap is used as a weak relevance proxy.
* Queries are drawn from the same patent corpus.
* Self-matches are excluded.
* The benchmark does not use human-annotated relevance judgements.
* The results are not a held-out test evaluation.
* One model, Nomic Embed v1.5, failed because of a Transformers compatibility issue.

These results should therefore be interpreted as preliminary retrieval comparisons rather than definitive evidence of real-world patent relevance.

RAG and Language Model Integration

The project explores Retrieval-Augmented Generation for patent analysis.

The intended workflow is:

1. Accept a patent-related query.
2. Retrieve semantically relevant patents.
3. Supply retrieved patent information as context.
4. Use a language model to generate a contextual response.

Language model integrations explored include:

* Google Gemini API.
* Ollama for local model inference.

The project also explores generating structured patent briefs from retrieved information.

Interactive Application

A Streamlit interface provides an interactive layer for exploring the patent intelligence pipeline.

The application connects the retrieval and language-model components to support patent discovery and analysis.

Technology Stack

Category	Technologies
Programming	Python
Data processing	Pandas, NumPy
Machine learning	PyTorch, Transformers
Embeddings	Sentence Transformers
Fine-tuning	BGE, contrastive learning, MNRL
Vector search	FAISS, ChromaDB
Language models	Gemini API, Ollama
Interface	Streamlit
Deployment experiments	Docker, Microsoft Azure
Model hosting	Hugging Face
Version control	Git, GitHub

Repository Structure

Patent_Intelligence_Project/
│
├── data/
│
├── experiments/
│   └── finetuning/
│       └── scripts/
│           ├── prepare_pairs.py
│           └── train_bge.py
│
├── models/
│
├── notebooks/
│
├── results/
│
├── src/
│   └── run_full_corpus_benchmark_v2.py
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore

Generated datasets, local environments, experiment outputs, and large model weights are excluded from the GitHub repository where appropriate.

Installation

Clone the repository:

git clone https://github.com/patzer-adi/Python_Workspace.git

Navigate to the project:

cd Python_Workspace/Patent_Intelligence_Project

Create a virtual environment:

python -m venv .venv

Activate it:

macOS / Linux

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Configure API credentials in a local .env file when using external language-model APIs. Never commit API keys or secrets.

Benchmark Execution

The benchmark script supports limited runs for testing and larger runs for more extensive evaluation.

Example smoke test:

python src/run_full_corpus_benchmark_v2.py \
    --limit 1500 \
    --expected-rows 25247 \
    --output results/full_corpus_benchmark_v2_smoke1500

Refer to the script’s command-line arguments for additional configuration.

Future Improvements

* Complete full-corpus benchmarking.
* Improve evaluation using held-out query sets.
* Introduce human-annotated relevance judgements.
* Explore stronger hard-negative mining strategies.
* Evaluate hybrid lexical and semantic retrieval.
* Improve reranking and retrieval evaluation.
* Expand patent-domain fine-tuning experiments.
* Improve RAG response evaluation and factual grounding.
* Strengthen deployment and application testing.

Author

Aditya Gowari

Saily Kadam

M.Sc. Scientific Computing
Savitribai Phule Pune University

GitHub: patzer-adi

Github: saily-kadam

Hugging Face: patzer-adi

⸻

This project is an ongoing exploration of semantic search, domain-specific representation learning, and retrieval-augmented language models for patent intelligence.