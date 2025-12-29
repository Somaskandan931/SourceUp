import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = "C:/Users/somas/PycharmProjects/SourceUp/data/embeddings"
INDEX_FILE = f"{BASE_DIR}/suppliers.faiss"
META_FILE = f"{BASE_DIR}/suppliers_meta.csv"

_model = None
_index = None
_meta = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def load_index():
    global _index, _meta

    if _index is not None:
        return _index, _meta

    if not os.path.exists(INDEX_FILE):
        raise RuntimeError(
            "FAISS index not found. "
            "Run pipeline/run_all.py first."
        )

    if not os.path.exists(META_FILE):
        raise RuntimeError(
            "Metadata file not found. "
            "suppliers_meta.csv is required."
        )

    _index = faiss.read_index(INDEX_FILE)
    _meta = pd.read_csv(META_FILE)

    return _index, _meta

def retrieve(query: str, k: int = 10):
    model = get_model()
    index, meta = load_index()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, ids = index.search(query_embedding, k)

    results = meta.iloc[ids[0]].copy()
    results["score"] = distances[0]

    return results.to_dict("records")
