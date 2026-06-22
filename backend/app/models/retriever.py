"""
Retriever Module — SourceUp (SBERT Primary Retrieval)
-------------------------------------------------------
SBERT + FAISS is the PRIMARY retrieval backbone.
Retrieves top-k candidates for LTR re-ranking.
"""

import os
import sys
import logging
from functools import lru_cache
import numpy as np
import pandas as pd
from typing import List, Dict

from pathlib import Path


def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))

import faiss
from sklearn.preprocessing import normalize as sk_normalize
from sentence_transformers import SentenceTransformer
from config import cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None
_index = None
_meta = None
_cross_encoder = None

# Priority 5: Query Expansion
# Simple synonym map so common SME procurement phrasing matches more
# supplier listings without requiring an exact wording match.
QUERY_SYNONYMS: Dict[str, List[str]] = {
    "food containers": ["packaging", "storage boxes", "plastic containers"],
    "packaging": ["containers", "boxes", "cartons"],
    "electronics": ["components", "circuit boards", "electronic parts"],
    "textiles": ["fabric", "cloth", "garments"],
    "machinery": ["equipment", "machines", "industrial equipment"],
    "furniture": ["home furniture", "office furniture", "wooden furniture"],
    "plastic": ["polymer", "PVC", "plastic products"],
    "steel": ["metal", "iron", "stainless steel"],
}


def expand_query(query: str) -> str:
    """
    Priority 5: Query Expansion.
    Appends related synonym terms to the query so retrieval isn't limited
    to exact keyword overlap. Returns the original query if no synonyms match.
    """
    q_lower = query.lower()
    expanded = query
    for phrase, synonyms in QUERY_SYNONYMS.items():
        if phrase in q_lower:
            expanded += " " + " ".join(synonyms)
    return expanded


def get_cross_encoder():
    """
    Priority 3: Cross Encoder reranking.
    Lazily loads a cross-encoder model used to rerank a small shortlist of
    FAISS candidates (NOT the full supplier base) for higher-precision
    ordering before LTR ranking.
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder 'cross-encoder/ms-marco-MiniLM-L-6-v2'...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("✅ CrossEncoder loaded")
    return _cross_encoder


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

    # FIX 2: Enriched supplier_text with product_name, location and price to
    # better match user query intent. Also lowercased for consistency with
    # query preprocessing. product_name is included FIRST since it's the
    # single most predictive field for matching procurement queries.
    _meta["supplier_text"] = (
        _meta.get("product_name", pd.Series([""] * len(_meta))).fillna("").astype(str) + " " +
        _meta.get("supplier_name", pd.Series([""] * len(_meta))).fillna("").astype(str) + " " +
        _meta.get("category", pd.Series([""] * len(_meta))).fillna("").astype(str) + " " +
        _meta.get("description", pd.Series([""] * len(_meta))).fillna("").astype(str) + " " +
        _meta.get("location", pd.Series([""] * len(_meta))).fillna("").astype(str) + " " +
        _meta.get("price", pd.Series([""] * len(_meta))).fillna("").astype(str)
    ).str.lower().str.strip()  # FIX 2: lowercase + strip for alignment with query preprocessing

    logger.info(f"✅ FAISS index loaded: {_index.ntotal:,} vectors")
    return _index, _meta


@lru_cache(maxsize=1000)
def _encode_query_cached(query: str) -> tuple:
    """
    Priority 8: Retrieval Cache.
    Caches SBERT embeddings for repeated queries (common in demos, autocomplete,
    and what-if simulations that re-issue the same product query). Returns a
    tuple so the result is hashable/cacheable; callers convert back to ndarray.
    """
    model = get_model()
    emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    emb = sk_normalize(emb, norm="l2")
    return tuple(emb[0].tolist())


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

    # Priority 5: Query Expansion — broaden the query with related terms
    # before embedding, so retrieval isn't limited to exact keyword overlap.
    expanded_query = expand_query(query)

    # Priority 8: Retrieval Cache — reuse cached embeddings for repeated queries.
    query_emb = np.array([_encode_query_cached(expanded_query)], dtype=np.float32)

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
            # Normalized cosine similarity in [0, 1] (relative to this query's
            # candidate pool only — two different queries are not comparable
            # on this scale).
            s["semantic_score"] = float((dist - score_min) / score_range)
            s["faiss_score"] = s["semantic_score"]   # keep alias for backward compat
            # Raw (pre-normalization) cosine similarity — IndexFlatIP returns
            # this directly in [-1, 1]. Unlike faiss_score, this is comparable
            # ACROSS queries and reflects absolute retrieval confidence rather
            # than within-query rank. Kept separately so feature_builder.py
            # can expose both signals instead of only the relative one.
            s["faiss_raw_score"] = float(dist)
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


def rerank_with_cross_encoder(query: str, candidates: List[Dict], top_n: int = 20) -> List[Dict]:
    """
    Priority 3: Cross Encoder reranking.

    Pipeline position: SBERT/FAISS retrieval (top ~100-200) -> Cross Encoder
    reranks ONLY this shortlist -> top_n passed on to the LTR ranker.

    Running a cross-encoder over the full supplier base (100k+) would be far
    too slow, so it's deliberately restricted to the candidates already
    returned by FAISS retrieval.
    """
    if not candidates:
        return candidates

    try:
        encoder = get_cross_encoder()
    except Exception as e:
        logger.warning(f"CrossEncoder unavailable, skipping rerank: {e}")
        return candidates

    pairs = [(query, c.get("supplier_text", "")) for c in candidates]
    try:
        ce_scores = encoder.predict(pairs)
    except Exception as e:
        logger.warning(f"CrossEncoder predict failed, skipping rerank: {e}")
        return candidates

    for c, s in zip(candidates, ce_scores):
        c["cross_encoder_score"] = float(s)

    reranked = sorted(candidates, key=lambda c: c["cross_encoder_score"], reverse=True)
    for rank, c in enumerate(reranked, 1):
        c["faiss_rank"] = rank

    logger.info(f"✅ Cross-encoder reranked {len(reranked)} candidates")
    return reranked[:top_n] if top_n else reranked


def retrieve_bm25(query: str, k: int = 10) -> List[Dict]:
    """Kept for backward compatibility - returns empty."""
    return []


def retrieve_hybrid(query: str, k: int = 10, alpha: float = 0.7) -> List[Dict]:
    """Hybrid not needed - SBERT is primary. Returns SBERT results."""
    return retrieve(query, k)