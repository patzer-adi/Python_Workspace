import json
import random
import csv
from pathlib import Path

random.seed(42)

files = list(Path("sample/2016").glob("*.json"))

print("Total JSON files:", len(files))

patents = []

for file in files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        title = (data.get("title") or "").strip()
        abstract = (data.get("abstract") or "").strip()

        if not title or not abstract:
            continue

        patents.append({
            "application_number": data.get("application_number"),
            "publication_number": data.get("publication_number"),
            "patent_number": data.get("patent_number"),
            "title": title,
            "abstract": abstract
        })

    except Exception:
        continue

# Remove exact duplicate Title + Abstract combinations
unique = {}

for patent in patents:
    key = (
        patent["title"].lower(),
        patent["abstract"].lower()
    )
    unique[key] = patent

patents = list(unique.values())

print("Valid unique patents:", len(patents))

# Select exactly 15 reproducibly
selected = random.sample(patents, 15)

output = Path("patent_sample_15.csv")

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "application_number",
            "publication_number",
            "patent_number",
            "title",
            "abstract"
        ]
    )

    writer.writeheader()
    writer.writerows(selected)

print("Saved:", output)
print("Selected patents:", len(selected))
