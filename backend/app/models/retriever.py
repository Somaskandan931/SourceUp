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
from sklearn.preprocessing import normalize as sk_normalize
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
        # FIX 1: Upgraded from all-MiniLM-L6-v2 → all-mpnet-base-v2 (higher quality embeddings)
        logger.info("Loading SBERT model 'all-mpnet-base-v2' for PRIMARY retrieval...")
        _model = SentenceTransformer("all-mpnet-base-v2")
        _model.max_seq_length = 256  # FIX 1: Increased from 128 → 256
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
    _meta.columns = [c.strip().lower().replace(" ", "_") for c in _meta.columns]

    # FIX 4: Log index type to confirm IndexFlatIP (cosine) vs IndexFlatL2
    logger.info(f"Index type: {type(_index)}")

    # FIX 2: Enriched supplier_text with location and price to better match user query intent.
    # Also lowercased for consistency with query preprocessing.
    _meta["supplier_text"] = (
        _meta.get("supplier_name", pd.Series([""] * len(_meta))).fillna("").astype(str) + " " +
        _meta.get("category", pd.Series([""] * len(_meta))).fillna("").astype(str) + " " +
        _meta.get("description", pd.Series([""] * len(_meta))).fillna("").astype(str) + " " +
        _meta.get("location", pd.Series([""] * len(_meta))).fillna("").astype(str) + " " +
        _meta.get("price", pd.Series([""] * len(_meta))).fillna("").astype(str)
    ).str.lower().str.strip()  # FIX 2: lowercase + strip for alignment with query preprocessing

    logger.info(f"✅ FAISS index loaded: {_index.ntotal:,} vectors")
    return _index, _meta


def retrieve(query: str, k: int = 200) -> List[Dict]:  # FIX 7: Increased default k 100 → 200
    """
    PRIMARY RETRIEVAL: SBERT + FAISS returns top-k candidates.
    These will be re-ranked by LTR (XGBRanker).
    """
    logger.info(f"🔍 SBERT retrieval: '{query}' (k={k})")

    model = get_model()
    index, meta = load_index()

    # FIX 5: Normalize query text before encoding
    query = query.lower().strip()

    # FIX 5: Encode with normalize_embeddings=True for guaranteed unit vectors
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    # Double-check: apply sklearn L2 normalize to guarantee unit norm
    query_emb = sk_normalize(query_emb, norm='l2')

    # Note: faiss.normalize_L2 is now redundant given normalize_embeddings=True above,
    # but kept as a safety guarantee.
    faiss.normalize_L2(query_emb)

    # Search
    distances, indices = index.search(query_emb, min(k, index.ntotal))

    # FIX 3: Removed wrong 1/(1+dist) transformation.
    # For IndexFlatIP (cosine similarity), dist IS the similarity score directly.
    # If using IndexFlatL2, use: semantic_score = -float(dist)
    raw_scores = distances[0].astype(float)

    # FIX 5: Normalize semantic scores to [0, 1] so downstream features are on a consistent scale
    score_min = float(raw_scores.min()) if len(raw_scores) > 0 else 0.0
    score_max = float(raw_scores.max()) if len(raw_scores) > 0 else 1.0
    score_range = score_max - score_min if score_max > score_min else 1.0

    results = []
    for rank, (dist, idx) in enumerate(zip(raw_scores, indices[0])):
        if idx < len(meta):
            s = meta.iloc[idx].to_dict()
            # Normalized cosine similarity in [0, 1]
            s["semantic_score"] = float((dist - score_min) / score_range)
            s["faiss_score"] = s["semantic_score"]   # keep alias for backward compat
            s["faiss_rank"] = rank + 1
            s["_index"] = int(idx)
            # Guarantee all ranking-critical fields exist with safe defaults
            for field, default in [
                ("price", None), ("price_min", None),
                ("supplier_location", ""), ("location", ""),
                ("certifications", ""), ("years_with_gs", 0),
            ]:
                if s.get(field) is None:
                    s[field] = default
            results.append(s)

    if results:
        logger.info(f"✅ Retrieved {len(results)} candidates (top score: {results[0]['semantic_score']:.4f})")
    return results


def retrieve_bm25(query: str, k: int = 10) -> List[Dict]:
    """Kept for backward compatibility - returns empty."""
    return []


def retrieve_hybrid(query: str, k: int = 10, alpha: float = 0.7) -> List[Dict]:
    """Hybrid not needed - SBERT is primary. Returns SBERT results."""
    return retrieve(query, k)