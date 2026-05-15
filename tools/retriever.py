# retriever.py

import json
import faiss
import numpy as np

from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_model = SentenceTransformer(MODEL_NAME)
_faiss_index = None
_catalog = None

def load_vector_database():

    global _faiss_index
    global _catalog

    if _faiss_index is not None and _catalog is not None:
        return _faiss_index, _catalog

    root_dir = Path(__file__).resolve().parent.parent
    index_path = root_dir / "data" / "catalog.index"

    catalog_path = root_dir / "data" / "catalog.json"

    _faiss_index = faiss.read_index(
        str(index_path)
    )

    with open(catalog_path, "r", encoding="utf-8") as f:
        _catalog = json.load(f)

    return _faiss_index, _catalog


def retrieve_assessments(
    query: str,
    top_k: int = 10
) -> List[Dict]:

    index, catalog = load_vector_database()

    model = _embedding_model

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_items = []

    for idx, distance in zip(indices[0], distances[0]):

        if idx == -1:
            continue

        item = catalog[idx]

        similarity_score = 1 / (1 + float(distance))

        retrieved_items.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "url": item.get("url"),
            "test_type": item.get("test_type"),
            "description": item.get("description"),
            "duration": item.get("duration"),
            "similarity_score": similarity_score
        })

    return retrieved_items