
from pathlib import Path
import csv
import chromadb

root = Path.cwd()
run_dir = root / "data/patent_subset_runs/20261022_190054/chroma"

model_patents = {}

for model_dir in sorted(run_dir.iterdir()):
    if not model_dir.is_dir():
        continue

    client = chromadb.PersistentClient(path=str(model_dir))
    ids = set()

    for item in client.list_collections():
        name = item.name if hasattr(item, "name") else str(item)
        collection = client.get_collection(name=name)

        # Read IDs and metadata in batches; no embeddings are loaded.
        offset = 0
        batch_size = 5000

        while True:
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"]
            )
            returned_ids = result.get("ids", [])
            metadata = result.get("metadatas", [])

            if not returned_ids:
                break

            for meta in metadata:
                if meta and meta.get("patent_number"):
                    ids.add(str(meta["patent_number"]))

            offset += len(returned_ids)

            if len(returned_ids) < batch_size:
                break

    model_patents[model_dir.name] = ids

graph_path = root / "data/patent_graph_2016/patent_similarity_nodes.csv"

with graph_path.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    graph_patents = {
        str(row["patent_number"]).strip()
        for row in reader
        if row.get("patent_number")
    }

print("Graph distinct patent IDs:", len(graph_patents))

for model, patents in model_patents.items():
    overlap = patents & graph_patents
    print(
        f"{model:14} "
        f"distinct patents={len(patents):4} | "
        f"graph overlap={len(overlap):4} | "
        f"index-only={len(patents - graph_patents):4}"
    )

if model_patents:
    common = set.intersection(*model_patents.values())
    print("\nPatents common to all seven indexes:", len(common))
    print("Common index patents also in graph:", len(common & graph_patents))

