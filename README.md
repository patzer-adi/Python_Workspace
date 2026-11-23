# Patent Intelligence Pipeline
A semantic patent search and analysis system that helps researchers and innovators discover patents related to a technical idea, investigate existing approaches, and compare relevant inventions using embedding models and large language models (LLMs).

The project explores how modern NLP, dense vector retrieval, reranking, and LLM-based analysis can be combined to build a practical patent intelligence system.

## 1. Project Objective
The goal is to develop a system that accepts a natural-language description of an invention or technical problem and retrieves patents with semantically related content.

Rather than relying exclusively on exact keyword matches, the system represents patent documents and user queries as dense numerical embeddings, allowing retrieval based on semantic similarity.

The system aims to:

- Retrieve patents relevant to a proposed technical idea.
- Identify related technical approaches and supporting patent passages.
- Compare retrieval quality across different embedding models.
- Improve candidate rankings using cross-encoder reranking.
- Combine rankings from multiple embedding models using Reciprocal Rank Fusion.
- Generate evidence-grounded research briefs using an LLM.
- Support further investigation through patent details, metadata, and similarity graphs.
**Important:** This is a patent discovery and research-support tool, not a legal novelty assessment or patentability determination.

## 2. End-to-End Pipeline
The project consists of two main stages: offline patent indexing and online query processing.

### A. Offline Patent Indexing
The offline pipeline converts patent documents into searchable vector representations.

```
Patent Dataset (HUPD / 2016 collection)
                 |
                 v
        Data Ingestion
                 |
                 v
      Record Validation
                 |
                 v
     Text Cleaning and Processing
                 |
                 v
       Patent Field Extraction
                 |
                 v
       Patent Text Preparation
                 |
                 v
          Text Chunking
                 |
                 v
       Embedding Model
                 |
                 v
        Vector Generation
                 |
                 v
          ChromaDB
```
Each patent retains its identifying information and associated metadata.

The text representation uses relevant patent content, including titles, abstracts, claims, and descriptions, depending on the indexing experiment.

Patent chunks are associated with metadata such as:

- Patent number
- Title
- Section
- Chunk index
- CPC and IPC classifications, where available
- Publication date
- Patent decision information
The resulting embeddings and metadata are stored in persistent ChromaDB collections.

### B. Online Query Processing
When a user submits an invention description, the application processes it through the retrieval pipeline.

```
       User's Technical Idea
                 |
                 v
         Query Processing
                 |
                 v
       Selected Embedding Model
                 |
                 v
            Query Vector
                 |
                 v
       ChromaDB Vector Search
                 |
                 v
       Candidate Patent Chunks
                 |
                 v
       Optional Cross-Encoder
             Reranking
                 |
                 v
        Patent-Level Grouping
                 |
                 v
       Ranked Patent Results
                 |
                 v
       LLM-Based Research Brief
                 |
                 v
          User Interface
```
The same embedding model used to index a collection is used to encode its search queries.

The application retrieves candidate chunks, groups them by patent, ranks the resulting patents, and presents supporting passages for inspection.

## 3. Embedding Models
A major experimental component of this project is comparing pretrained embedding models for patent retrieval.

The application supports seven embedding models:

Model | Model identifier

MiniLM | `sentence-transformers/all-MiniLM-L6-v2`

MPNet | `sentence-transformers/all-mpnet-base-v2`

BGE | `BAAI/bge-base-en-v1.5`

E5 | `intfloat/e5-base-v2`

PatentSBERTa | `AI-Growth-Lab/PatentSBERTa`

PaECTER | `mpi-inno-comp/paecter`

Nomic | `nomic-ai/nomic-embed-text-v1.5`
These models provide different approaches to representing technical language in vector space.

The project investigates their performance on patent retrieval rather than assuming that a general-purpose embedding model will necessarily perform best on patent documents.

Each model uses its corresponding vector collection.

## 4. Retrieval and Ranking

### Dense Vector Retrieval
The user's query is converted into a vector representation and compared against indexed patent chunks using vector similarity search.

ChromaDB retrieves the nearest candidate chunks from the selected collection.

### Patent-Level Grouping
Since a patent may contain multiple relevant chunks, retrieved chunks are grouped by their patent number.

The application calculates a patent-level ranking score using:

`Patent Score = 0.5 × Best Chunk Score + 0.5 × Mean of Top 3 Chunk Scores`

This is a ranking heuristic designed to combine the strongest matching passage with evidence from other relevant passages.

It is not a calibrated probability of relevance.

### Cross-Encoder Reranking
The application optionally applies a cross-encoder to the retrieved query–document pairs.

Unlike bi-encoder retrieval, which compares independently generated vectors, a cross-encoder evaluates the query and candidate passage together.

The reranker used is:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

This provides a second-stage ranking mechanism for the initially retrieved candidates.

### Reciprocal Rank Fusion (RRF)
The project also implements multi-model comparison using Reciprocal Rank Fusion.

Different embedding models independently retrieve and rank patents. Their rankings are then combined using RRF.

The fusion score is calculated as:

`RRF(d) = Σ 1 / (k + rank(d))`

where `rank(d)` is the rank of a patent in an individual model's results and `k` is the RRF constant.

This allows patents appearing consistently near the top across different models to receive stronger combined rankings.

The application also provides model-wise comparisons and top-list overlap analysis.

## 5. LLM-Based Research Brief Generation
After retrieving relevant patents, the application can generate a research brief grounded in the retrieved patent evidence.

The LLM receives the user's technical idea and selected retrieved patent information.

The generated brief is intended to help investigate:

- Common technical aspects
- Differences between the proposed idea and existing patents
- Distinctive technical approaches
- Potential areas of overlap
- Patent status and relevant evidence
- Further directions for investigation
The application supports a local Ollama-based LLM provider by default, with support for OpenAI-compatible endpoints.

The default configuration uses `llama3.2`.

LLM-generated conclusions must be checked against the original patent claims and supporting passages. They do not establish legal novelty or infringement.

## 6. Application Features
The Streamlit application provides the following functionality:

- **Patent Search:** Search patents using natural-language descriptions of technical ideas.
- **Research Brief:** Generate an LLM-based research brief grounded in retrieved evidence.
- **Relevant Patents:** Explore ranked patent results and export results as CSV.
- **Patent Details:** Inspect abstracts, metadata, similarity scores, and supporting chunks.
- **Advanced Settings:** Select embedding models, configure candidate retrieval depth, and enable or disable reranking.
- **Model Comparison:** Compare multiple embedding models on the same query.
- **Reciprocal Rank Fusion:** Combine patent rankings across models.
- **Graph Explorer:** Explore the saved 2016 patent similarity graph.
- **Evaluation and Diagnostics:** Inspect index health, embedding dimensions, retrieval evaluation, and saved reports.

## 7. Dataset and Vector Database
The project uses patent data associated with the HUPD / 2016 patent collection.

The indexing experiments produce separate ChromaDB collections for the embedding models, allowing retrieval experiments to be performed independently.

The legacy MiniLM collection contains 67,303 indexed chunks in the local project database. This is a chunk count, not a patent count.

ChromaDB is used for persistent vector storage and nearest-neighbour retrieval.

**Note:** The indexed database is stored separately from the GitHub source repository because of its size. A fresh clone therefore requires the corresponding indexed data to be supplied or generated before the full application can run.

## 8. Technology Stack

Area | Technologies

Programming language | Python

Application | Streamlit

NLP and embeddings | Sentence Transformers

Embedding models | MiniLM, MPNet, BGE, E5, PatentSBERTa, PaECTER, Nomic

Vector database | ChromaDB

Reranking | Cross-Encoder

Retrieval fusion | Reciprocal Rank Fusion

LLM inference | Ollama, OpenAI-compatible APIs

Data processing | NumPy, Pandas

Visualization | Plotly

Patent data | HUPD / 2016 patent collection

## 9. Repository Structure

```
Patent_Intelligence_Project/
│
├── streamlit_app.py
│
├── src/
│   ├── rag_search.py
│   └── ...
│
├── data/
│   ├── chroma_db/
│   ├── patent_subset_runs/
│   └── patent_graph_2016/
│
├── notebooks/
│
├── pages/
│
├── reports/
│
├── results/
│
├── trials/
│   └── results/
│
├── system_images/
│
├── requirements.txt
├── requirements-demo.txt
├── requirements-deploy.txt
├── .gitignore
└── README.md
```
The repository contains the application, research experiments, supporting source code, evaluation outputs, and data-processing resources. Large datasets, local environments, and persistent vector databases are maintained separately from the source-controlled code.

## 10. Installation and Execution

### Prerequisites

- Python 3.12 or a compatible supported Python version
- Git
- Access to the project's indexed ChromaDB collections
- Sufficient memory and storage for the selected embedding models

### Clone the Repository

```
git clone <YOUR_REPOSITORY_URL>
cd Patent_Intelligence_Project
```

### Create a Virtual Environment

```
python3.12 -m venv .patent-intelligence
source .patent-intelligence/bin/activate
```

### Install Dependencies

```
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
```

### Configure the Data
Place the required indexed ChromaDB collections and graph data in their expected project directories.

The application expects model-specific collections under:

```
data/patent_subset_runs/<RUN_ID>/chroma/
```
The original MiniLM collection is maintained separately under:

```
data/chroma_db/
```

### Run the Application

```
streamlit run streamlit_app.py
```
Open the local Streamlit URL displayed in the terminal.

## 11. Configuration
The application supports configuration through environment variables.

Variable | Purpose

`PATENT_APP_RUN_ID` | Selects the indexed patent collection run

`PATENT_APP_DEVICE` | Selects the inference device

`PATENT_APP_LLM_PROVIDER` | Selects `ollama` or `openai`

`PATENT_APP_LLM_MODEL` | Specifies the LLM model

`PATENT_APP_OLLAMA_URL` | Configures the Ollama endpoint

`PATENT_APP_LLM_BASE_URL` | Configures an OpenAI-compatible endpoint

`PATENT_APP_LLM_API_KEY` | API key for a configured remote LLM provider
Keep API keys and credentials outside source control. Do not commit `.env`, Streamlit secrets, or other private configuration files.

## 12. Experiments and Evaluation
The project includes experiments investigating:

- Differences in semantic retrieval across embedding models.
- Patent retrieval using different text representations.
- Candidate retrieval depth.
- Cross-encoder reranking.
- Multi-model ranking and Reciprocal Rank Fusion.
- Retrieval performance and result overlap.
- Computational requirements of different embedding models.
Evaluation outputs and experiment results are maintained in the `trials/`, `results/`, and `reports/` directories.

Quantitative performance claims should be reported only alongside the corresponding evaluation setup, dataset, and retrieval metrics.

## 13. Limitations

- Retrieval quality depends on the coverage and quality of the indexed patent corpus.
- Semantic similarity does not establish technical equivalence.
- Similarity scores are ranking signals, not probabilities of patent overlap.
- Cross-encoder scores are not calibrated probabilities.
- LLM-generated research briefs may contain errors and require verification against original patent documents.
- The system does not replace professional patent searches or legal advice.
- Running several embedding models and rerankers can require substantial memory and computational resources.

## 14. Future Improvements

- Expand the indexed patent corpus and improve data coverage.
- Investigate hybrid retrieval combining keyword and dense vector search.
- Improve patent-specific chunking and document representation.
- Evaluate retrieval using labelled relevance judgements and standard ranking metrics.
- Investigate more effective query expansion and structured query representations.
- Improve LLM grounding, evidence attribution, and comparison quality.
- Explore more efficient indexing and inference strategies for larger patent collections.

---

**Project focus:** Applying NLP, representation learning, vector databases, information retrieval, and LLMs to make technical patent discovery more accessible and useful.
