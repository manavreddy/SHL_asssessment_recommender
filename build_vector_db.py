# build_vector_db.py

import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def catalog_item_to_text(item):

    name = item.get("name", "")
    description = item.get("description", "")
    test_type = item.get("test_type", "")

    keys = " ".join(
        item.get("keys", [])
    )

    job_levels = " ".join(
        item.get("job_levels", [])
    )

    languages = " ".join(
        item.get("languages", [])
    )

    text_parts = [
        name,
        name,  # repeated intentionally
        description,
        keys,
        keys,  # repeated intentionally
        job_levels,
        test_type,
        languages,
    ]

    # --------------------------------
    # Semantic enrichment for OPQ
    # --------------------------------

    if "opq" in name.lower():

        text_parts.extend([
            "personality assessment",
            "behavioral assessment",
            "workplace behavior",
            "leadership style",
            "influencing style",
            "executive personality",
            "strategic leadership",
            "leadership benchmarking",
            "candidate selection"
        ])

    return " ".join(text_parts)


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
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        str(index_path)
    )

    print("Vector database created successfully")


if __name__ == "__main__":
    main()
