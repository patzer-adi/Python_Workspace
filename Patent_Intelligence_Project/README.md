# Patent Intelligence Pipeline

A semantic patent search and analysis system that helps researchers and innovators discover patents related to a technical idea, investigate existing approaches, and compare relevant inventions using embedding models, vector retrieval, reranking, and large language models (LLMs).

The project explores how modern NLP, dense vector retrieval, patent-specific representation learning, and LLM-based analysis can be combined to build a practical patent intelligence system.

**Important:** This is a patent discovery and research-support tool, not a legal novelty assessment, patentability determination, or substitute for professional legal advice.

---

## 1. Project Objective

The goal is to develop a system that accepts a natural-language description of an invention or technical problem and retrieves patents with semantically related content.

Rather than relying exclusively on exact keyword matches, the system represents patent documents and user queries as dense numerical embeddings, allowing retrieval based on semantic similarity.

The system aims to:

- Retrieve patents relevant to a proposed technical idea.
- Identify related technical approaches and supporting patent passages.
- Compare retrieval quality across different embedding models.
- Fine-tune pretrained embedding models for patent-domain retrieval.
- Improve candidate rankings using cross-encoder reranking.
- Combine rankings from multiple embedding models using Reciprocal Rank Fusion.
- Generate evidence-grounded research briefs using an LLM.
- Support further investigation through patent details, metadata, and similarity graphs.

## 2. End-to-End Pipeline

The project consists of two main stages: offline patent indexing and online query processing.

### A. Offline Patent Indexing

The offline pipeline converts patent documents into searchable vector representations.

```text
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

Patent chunks can retain metadata such as:

- Patent number
- Title
- Section
- Chunk index
- CPC and IPC classifications, where available
- Publication date
- Patent decision information
The resulting embeddings and metadata are stored in persistent ChromaDB collections.

### **B. Online Query Processing**
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

## **3. Embedding Models**
A major experimental component of this project is comparing pretrained embedding models for patent retrieval.

### **Pretrained models**

| Model | Model identifier |
| --- | --- |
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` |
| MPNet | `sentence-transformers/all-mpnet-base-v2` |
| BGE | `BAAI/bge-base-en-v1.5` |
| E5 | `intfloat/e5-base-v2` |
| PatentSBERTa | `AI-Growth-Lab/PatentSBERTa` |
| PaECTER | `mpi-inno-comp/paecter` |
| Nomic | `nomic-ai/nomic-embed-text-v1.5` |

### **Fine-tuned patent embedding models**
Three BGE-based models were fine-tuned as part of the patent representation learning experiments.

| Model | Training approach | Hugging Face |
| --- | --- | --- |
| BGE Patent MNRL | Multiple Negatives Ranking Loss (MNRL) | [Model](https://huggingface.co/patzer-adi/bge-patent-mnrl) |
| BGE Similarity Contrastive | Similarity-based contrastive learning | [Model](https://huggingface.co/patzer-adi/bge-patent-similarity-contrastive) |
| BGE Hybrid Contrastive | Hybrid contrastive learning | [Model](https://huggingface.co/patzer-adi/bge-patent-hybrid-contrastive) |

All three fine-tuned model checkpoints are publicly available on Hugging Face.

These experiments investigate whether domain-oriented fine-tuning improves the representation of patent documents and retrieval performance compared with pretrained embedding models.

### **Loading a fine-tuned model**
The models can be loaded using Sentence Transformers:

```
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "patzer-adi/bge-patent-mnrl"
)

embeddings = model.encode(
    ["A technical description of an invention"],
    normalize_embeddings=True
)

print(embeddings.shape)
```

The similarity-contrastive and hybrid-contrastive models can be loaded by replacing the model identifier with their respective Hugging Face repository IDs.

## **4. Retrieval and Ranking**

### **Dense Vector Retrieval**
The user's query is converted into a vector representation and compared against indexed patent chunks using vector similarity search.

ChromaDB retrieves the nearest candidate chunks from the selected collection.

### **Patent-Level Grouping**
Since a patent may contain multiple relevant chunks, retrieved chunks are grouped by their patent number.

The application calculates a patent-level ranking score using:

`Patent Score = 0.5 × Best Chunk Score + 0.5 × Mean of Top 3 Chunk Scores`

This is a ranking heuristic designed to combine the strongest matching passage with evidence from other relevant passages.

It is not a calibrated probability of relevance.

### **Cross-Encoder Reranking**
The application optionally applies a cross-encoder to the retrieved query-document pairs.

Unlike bi-encoder retrieval, which compares independently generated vectors, a cross-encoder evaluates the query and candidate passage together.

The reranker used is:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

This provides a second-stage ranking mechanism for the initially retrieved candidates.

### **Reciprocal Rank Fusion (RRF)**
The project also implements multi-model comparison using Reciprocal Rank Fusion.

Different embedding models independently retrieve and rank patents. Their rankings are then combined using RRF.

The fusion score is calculated as:

`RRF(d) = Σ 1 / (k + rank(d))`

where `rank(d)` is the rank of a patent in an individual model's results and `k` is the RRF constant.

This allows patents appearing consistently near the top across different models to receive stronger combined rankings.

The application also provides model-wise comparisons and top-list overlap analysis.

## **5. LLM-Based Research Brief Generation**
After retrieving relevant patents, the application can generate a research brief grounded in the retrieved patent evidence.

The LLM receives the user's technical idea and selected retrieved patent information.

The generated brief is intended to help investigate:

- Common technical aspects
- Differences between the proposed idea and existing patents
- Distinctive technical approaches
- Potential areas of overlap
- Patent status and relevant evidence
- Further directions for investigation
The application supports a local Ollama-based LLM provider, with support for OpenAI-compatible endpoints.

The default configuration uses `llama3.2`.

LLM-generated conclusions must be checked against the original patent claims and supporting passages. They do not establish legal novelty or infringement.

## **6. Application Features**
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

## **7. Dataset and Vector Database**
The project uses patent data associated with the HUPD / 2016 patent collection.

The indexing experiments produce separate ChromaDB collections for the embedding models, allowing retrieval experiments to be performed independently.

The legacy MiniLM collection contains 67,303 indexed chunks in the local project database. This is a chunk count, not a patent count.

ChromaDB is used for persistent vector storage and nearest-neighbour retrieval.

**Dataset and storage note:** The indexed database and large generated datasets are maintained separately from the GitHub source repository because of their size. A fresh clone therefore requires the corresponding indexed data to be supplied or generated before the full application can run.

## **8. Technology Stack**

| Area | Technologies |
| --- | --- |
| Programming language | Python |
| Application | Streamlit |
| NLP and embeddings | Sentence Transformers, Transformers |
| Embedding models | MiniLM, MPNet, BGE, E5, PatentSBERTa, PaECTER, Nomic |
| Fine-tuning | BGE, contrastive learning, MNRL |
| Vector database | ChromaDB |
| Similarity search | FAISS |
| Reranking | Cross-Encoder |
| Retrieval fusion | Reciprocal Rank Fusion |
| LLM inference | Ollama, Gemini / Google GenAI, OpenAI-compatible APIs |
| Data processing | NumPy, Pandas |
| Visualization | Plotly |
| Patent data | HUPD / 2016 patent collection |
| Model hosting | Hugging Face |

## **9. Repository Structure**

```
Patent_Intelligence_Project/
│
├── streamlit_app.py
│
├── src/
│   ├── rag_search.py
│   ├── run_full_corpus_benchmark_v2.py
│   └── ...
│
├── data/
│   ├── chroma_db/
│   ├── patent_subset_runs/
│   └── patent_graph_2016/
│
├── experiments/
│   └── finetuning/
│
├── models/
│   └── fine_tuned_bge/
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

Large model weights are hosted separately on Hugging Face. Large datasets, local environments, generated embeddings, and persistent vector databases are excluded from the source-controlled repository.

## **10. Installation and Execution**

### **Prerequisites**

- Python 3.12 or a compatible supported Python version
- Git
- Access to the project's indexed ChromaDB collections
- Sufficient memory and storage for the selected embedding models

### **Clone the Repository**

```
git clone https://github.com/patzer-adi/Python_Workspace.git

cd Python_Workspace/Patent_Intelligence_Project
```

### **Create a Virtual Environment**

```
python3.12 -m venv .patent-intelligence

source .patent-intelligence/bin/activate
```

### **Install Dependencies**

```
python -m pip install --upgrade pip

python -m pip install -r requirements-demo.txt
```

### **Configure the Data**
Place the required indexed ChromaDB collections and graph data in their expected project directories.

The application expects model-specific collections under:

```
data/patent_subset_runs/<RUN_ID>/chroma/
```

The original MiniLM collection is maintained separately under:

```
data/chroma_db/
```

### **Run the Application**

```
streamlit run streamlit_app.py
```

Open the local Streamlit URL displayed in the terminal.

## **11. Configuration**
The application supports configuration through environment variables.

| Variable | Purpose |
| --- | --- |
| `PATENT_APP_RUN_ID` | Selects the indexed patent collection run |
| `PATENT_APP_DEVICE` | Selects the inference device |
| `PATENT_APP_LLM_PROVIDER` | Selects `ollama` or `openai` |
| `PATENT_APP_LLM_MODEL` | Specifies the LLM model |
| `PATENT_APP_OLLAMA_URL` | Configures the Ollama endpoint |
| `PATENT_APP_LLM_BASE_URL` | Configures an OpenAI-compatible endpoint |
| `PATENT_APP_LLM_API_KEY` | API key for a configured remote LLM provider |

Keep API keys and credentials outside source control. Do not commit `.env`, Streamlit secrets, or other private configuration files.

## **12. Experiments and Evaluation**
The project includes experiments investigating:

- Differences in semantic retrieval across embedding models.
- Patent retrieval using different text representations.
- Fine-tuning pretrained embedding models for patent-domain retrieval.
- Candidate retrieval depth.
- Cross-encoder reranking.
- Multi-model ranking and Reciprocal Rank Fusion.
- Retrieval performance and result overlap.
- Computational requirements of different embedding models.

### **Preliminary Embedding Model Benchmark**
A preliminary benchmark compared pretrained and fine-tuned embedding models on a sample of 1,500 patents drawn from a source collection of 25,247 rows.

The benchmark used:

- Patent titles and abstracts as the text representation.
- A maximum sequence length of 256.
- Query patents sampled from the same corpus.
- Self-matches excluded from retrieval results.
- CPC subclass overlap, based on the first four characters of the CPC label, as a relevance proxy.
The following results are reported at K = 20.

| Model | Hit@20 | Precision@20 | Recall@20 | MRR@20 | nDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PatentSBERTa | 0.7520 | 0.1799 | 0.2066 | 0.4563 | 0.2789 |
| PaECTER | 0.7993 | 0.2212 | 0.2679 | 0.5074 | 0.3430 |
| BGE-base-en-v1.5 | 0.7680 | 0.1943 | 0.2216 | 0.4642 | 0.2971 |
| E5-base-v2 | 0.7727 | 0.1993 | 0.2349 | 0.4791 | 0.3062 |
| all-mpnet-base-v2 | 0.7873 | 0.2112 | 0.2497 | 0.4881 | 0.3274 |
| all-MiniLM-L6-v2 | 0.7600 | 0.1986 | 0.2219 | 0.4605 | 0.2978 |
| BGE Patent MNRL | **0.8287** | **0.2711** | **0.3323** | **0.5554** | **0.4171** |
| BGE Similarity Contrastive | 0.7740 | 0.2443 | 0.2764 | 0.5001 | 0.3621 |
| BGE Hybrid Contrastive | 0.7853 | 0.2451 | 0.2743 | 0.4982 | 0.3554 |
| Nomic Embed v1.5 | Failed | - | - | - | - |

BGE Patent MNRL achieved the strongest reported results across these metrics in this preliminary experiment. The other two fine-tuned BGE models also completed successfully.

**Evaluation limitations:**

- This is a preliminary 1,500-patent sample benchmark, not a full-corpus evaluation.
- CPC subclass overlap is a weak metadata-based relevance proxy, not a human-annotated relevance judgement.
- Queries and candidate patents originate from the same source corpus; this is not a held-out generalisation test.
- Results depend on model-specific input formatting, prefixes, and truncation.
- The Nomic run failed due to a model/Transformers compatibility issue and is not included in the numerical comparison.
These results should be treated as exploratory evidence rather than a definitive ranking of embedding models.

The benchmark script and report are maintained in the project source and experiment outputs.

## **13. Limitations**

- Retrieval quality depends on the coverage and quality of the indexed patent corpus.
- Semantic similarity does not establish technical equivalence.
- Similarity scores are ranking signals, not probabilities of patent overlap.
- Cross-encoder scores are not calibrated probabilities.
- LLM-generated research briefs may contain errors and require verification against original patent documents.
- CPC-based benchmark relevance is only a proxy and does not replace expert relevance judgements.
- The system does not replace professional patent searches or legal advice.
- Running several embedding models and rerankers can require substantial memory and computational resources.

## **14. Future Improvements**

- Expand the indexed patent corpus and improve data coverage.
- Investigate hybrid retrieval combining keyword and dense vector search.
- Improve patent-specific chunking and document representation.
- Evaluate retrieval using labelled relevance judgements and standard ranking metrics.
- Conduct held-out evaluation to measure generalisation.
- Investigate more effective query expansion and structured query representations.
- Improve LLM grounding, evidence attribution, and comparison quality.
- Explore more efficient indexing and inference strategies for larger patent collections.

**Project focus:** Applying NLP, representation learning, vector databases, information retrieval, and LLMs to make technical patent discovery more accessible and useful.
