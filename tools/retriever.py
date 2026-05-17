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

    # Normalize query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)

    similarities, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_items = []

    for idx, similarity in zip(indices[0], similarities[0]):

        if idx == -1:
            continue

        item = catalog[idx]

        retrieved_items.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "url": item.get("url"),
            "test_type": item.get("test_type"),
            "description": item.get("description"),
            "duration": item.get("duration"),
            "keys": item.get("keys", []),
            "languages": item.get("languages", []),
            "job_levels": item.get("job_levels", []),
            "similarity_score": float(similarity)
        })

    # -----------------------------
    # Hybrid semantic reranking
    # -----------------------------

    query_lower = query.lower()

    for item in retrieved_items:

        score = item["similarity_score"]

        description = item.get(
            "description",
            ""
        ).lower()

        test_type = item.get(
            "test_type",
            ""
        ).lower()

        name = item.get(
            "name",
            ""
        ).lower()

        keys = " ".join(
            item.get("keys", [])
        ).lower()

        job_levels = " ".join(
            item.get("job_levels", [])
        ).lower()

        # --------------------------------
        # Leadership relevance
        # --------------------------------

        if any(term in query_lower for term in [
            "leadership",
            "executive",
            "senior leadership",
            "director",
            "cxo"
        ]):

            if "leadership" in description:
                score += 0.04

            if "leadership" in name:
                score += 0.03

            if "executive" in job_levels:
                score += 0.05

            if "director" in job_levels:
                score += 0.04

        # --------------------------------
        # Personality / behavioral signals
        # Strongly favor OPQ-family products
        # --------------------------------

        personality_behavior_terms = [
            "personality",
            "behavior",
            "behavioral",
            "workplace",
            "influencing",
            "leadership style",
            "interpersonal",
            "strategic thinking"
        ]

        if any(term in query_lower for term in personality_behavior_terms):

            if "personality" in description:
                score += 0.08

            if "behavior" in description:
                score += 0.06

            if "workplace" in description:
                score += 0.04

            if "influencing" in description:
                score += 0.04

            if "personality & behavior" in keys:
                score += 0.08

            if test_type == "p":
                score += 0.05

            # OPQ semantic boosting
            if "opq" in name:
                score += 0.12

            if "occupational personality questionnaire" in description:
                score += 0.10

        # --------------------------------
        # Selection / benchmarking signals
        # --------------------------------

        if any(term in query_lower for term in [
            "selection",
            "benchmark",
            "benchmarking",
            "compare candidates"
        ]):

            if "benchmark" in description:
                score += 0.05

            if "selection" in description:
                score += 0.04

        # --------------------------------
        # Cognitive / aptitude signals
        # --------------------------------

        cognitive_terms = [
            "cognitive",
            "reasoning",
            "numerical",
            "verbal",
            "aptitude",
            "problem solving"
        ]

        if any(term in query_lower for term in cognitive_terms):

            if any(term in name for term in [
                "verify",
                "ability",
                "aptitude"
            ]):
                score += 0.10

            if "ability & aptitude" in keys:
                score += 0.08

        # --------------------------------
        # Technical skill assessment signals
        # --------------------------------

        technical_terms = [
            "java",
            "python",
            "sql",
            "coding",
            "developer",
            "software",
            "programming"
        ]

        if any(term in query_lower for term in technical_terms):

            if "knowledge & skills" in keys:
                score += 0.08

            if test_type == "k":
                score += 0.05

        # --------------------------------
        # Penalize development-heavy reports
        # when user clearly wants selection
        # --------------------------------

        if any(term in query_lower for term in [
            "selection",
            "benchmark",
            "executive"
        ]):

            if "development" in description:
                score -= 0.05

        item["similarity_score"] = score

    # Final reranking
    retrieved_items.sort(
        key=lambda x: x["similarity_score"],
        reverse=True
    )

    return retrieved_items