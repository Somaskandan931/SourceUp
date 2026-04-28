"""
Retriever Module — SourceUp
-----------------------------
Semantic search using FAISS + sentence-transformers.
All paths come from config.cfg — no hardcoded strings.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from config import cfg

_model = None
_index = None
_meta  = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def load_index():
    global _index, _meta
    if _index is not None:
        return _index, _meta

    index_path = str(cfg.FAISS_INDEX)
    meta_path  = str(cfg.FAISS_META)

    if not os.path.exists(index_path):
        raise RuntimeError(
            f"FAISS index not found: {index_path}\n"
            "Run:  python pipeline/run_all.py"
        )
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"Metadata not found: {meta_path}\n"
            "Run:  python pipeline/run_all.py"
        )

    _index = faiss.read_index(index_path)
    _meta  = pd.read_csv(meta_path)
    print(f"✅ FAISS index loaded ({_index.ntotal} vectors, {len(_meta)} suppliers)")
    return _index, _meta


def retrieve(query: str, k: int = 10) -> List[Dict]:
    """
    Semantic search — returns top-k supplier dicts with faiss_score and rank.
    """
    model         = get_model()
    index, meta   = load_index()
    query_emb     = model.encode([query], convert_to_numpy=True)
    distances, ids = index.search(query_emb, k)

    results = []
    for idx, (dist, sid) in enumerate(zip(distances[0], ids[0])):
        if sid < len(meta):
            s = meta.iloc[sid].to_dict()
            s["faiss_score"] = float(dist)
            s["rank"]        = idx + 1
            results.append(s)
    return results