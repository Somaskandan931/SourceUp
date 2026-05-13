"""
Retriever Module — SourceUp (SBERT Primary Retrieval)
-------------------------------------------------------
SBERT + FAISS is the PRIMARY retrieval backbone.
Retrieves top-k candidates for LTR re-ranking.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

import faiss
from sentence_transformers import SentenceTransformer
from config import cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None
_index = None
_meta = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading SBERT model 'all-MiniLM-L6-v2' for PRIMARY retrieval...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _model.max_seq_length = 128
        logger.info("✅ SBERT model loaded")
    return _model


def load_index():
    global _index, _meta
    if _index is not None:
        return _index, _meta

    index_path = str(cfg.FAISS_INDEX)
    meta_path = str(cfg.FAISS_META)

    if not os.path.exists(index_path):
        raise RuntimeError(f"FAISS index not found: {index_path}\nRun: python pipeline/run_all.py")

    _index = faiss.read_index(index_path)
    _meta = pd.read_csv(meta_path)
    _meta.columns = [c.strip().lower() for c in _meta.columns]
    logger.info(f"✅ FAISS index loaded: {_index.ntotal:,} vectors")
    return _index, _meta


def retrieve(query: str, k: int = 100) -> List[Dict]:
    """
    PRIMARY RETRIEVAL: SBERT + FAISS returns top-k candidates.
    These will be re-ranked by LTR (XGBRanker).
    """
    logger.info(f"🔍 SBERT retrieval: '{query}' (k={k})")

    model = get_model()
    index, meta = load_index()

    # Encode query
    query_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_emb)

    # Search
    distances, indices = index.search(query_emb, min(k, index.ntotal))

    # Convert L2 distance to similarity: similarity = 1 / (1 + distance)
    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(meta):
            s = meta.iloc[idx].to_dict()
            s["faiss_score"] = 1.0 / (1.0 + float(dist))  # This is SBERT similarity
            s["faiss_rank"] = rank + 1
            s["_index"] = int(idx)
            results.append(s)

    logger.info(f"✅ Retrieved {len(results)} candidates (top score: {results[0]['faiss_score']:.4f})")
    return results


def retrieve_bm25(query: str, k: int = 10) -> List[Dict]:
    """Kept for backward compatibility - returns empty."""
    return []


def retrieve_hybrid(query: str, k: int = 10, alpha: float = 0.7) -> List[Dict]:
    """Hybrid not needed - SBERT is primary. Returns SBERT results."""
    return retrieve(query, k)