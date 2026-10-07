# Patent Intelligence System Project

## Streamlit app

The Streamlit app provides semantic search over the existing ChromaDB patent
chunk collection, with expandable patent results, retrieved evidence chunks,
and pipeline inspection. It is read-only: it opens `patent_rag_chunks` in
`data/chroma_db` and does not rebuild or modify the index.

The search flow is:

```text
query -> MiniLM embedding -> ChromaDB top-K -> optional cross-encoder rerank
	-> patent grouping -> evidence inspection
```

### Run locally

Use the repository's `venv` environment for both installation and runtime. In
the terminal, change to the repository root and run:

```bash
source venv/bin/activate
python --version
which python
python -m pip install -r requirements-demo.txt
python -m streamlit run streamlit_app.py
```

`which python` should point to this project's `venv/bin/python`. In VS Code,
select `venv/bin/python` as the project interpreter so the editor uses the same
environment. To leave the environment, run `deactivate`.

The demo dependencies (Streamlit, ChromaDB, and Sentence Transformers) are
listed in `requirements-demo.txt`. Install that file into `venv`, not system
Python. The checked-in `venv` currently uses Python 3.13. The larger
`requirements.txt` is for the broader notebook/research environment.

The first run may download the MiniLM model and, if enabled, the cross-encoder.
The sidebar lets you change the ChromaDB directory and collection name while
keeping the app read-only.

Search retrieves up to 50 candidate chunks by default and groups those chunks
by patent. The patent score combines the strongest chunk score with the mean of
the top three chunk scores for that patent, which is an inspection heuristic
rather than a calibrated relevance probability.
