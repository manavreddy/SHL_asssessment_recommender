# build_vector_db.py

import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def catalog_item_to_text(item):

    parts = [
        item.get("name", ""),
        item.get("description", ""),
        item.get("test_type", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
        " ".join(item.get("languages", [])),
        item.get("remote", ""),
        item.get("adaptive", "")
    ]

    return " ".join(parts)


def main():

    root_dir = Path(__file__).resolve().parent

    catalog_path = root_dir / "data" / "catalog.json"

    index_path = root_dir / "data" / "catalog.index"

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    searchable_texts = [
        catalog_item_to_text(item)
        for item in catalog
    ]

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        searchable_texts,
        convert_to_numpy=True
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        str(index_path)
    )

    print("Vector database created successfully")


if __name__ == "__main__":
    main()