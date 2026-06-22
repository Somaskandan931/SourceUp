"""
SourceUp — Production Ranker (inference only)
-----------------------------------------------
Exposes:
  get_ranker()              → singleton SupplierRanker instance
  extract_features_batch()  → build feature DataFrame from supplier dicts
  apply_mmr()                → Maximal Marginal Relevance diversification
  FEATURE_COLS               → canonical 6-feature list
"""

import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

import sys
def _find_project_root(marker: str = "config.py") -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")

sys.path.insert(0, str(_find_project_root()))
from config import cfg
from rule_baseline import score_rule_based as _canonical_rule_scorer

# ── Canonical feature list (must match train_lambdarank.py) ──────────────────
# NOTE: faiss_rank intentionally excluded — correlation with relevance was
# 0.025 (near-zero), redundant with faiss_score (rank is a lossy derivative
# of the raw similarity score). Still computed below for the rule-based
# fallback scorer and MMR, just not selected for the ML model.
FEATURE_COLS = [
    "price_match",
    "price_ratio",
    "location_match",
    "cert_match",
    "faiss_score",
]


# ── Price parsing ──────────────────────────────────────────────────────────────

def parse_price(v) -> float:
    """
    Parse a price value that may be a plain number, a numeric string, a range
    string like '0.08 - 0.20' (takes the lower bound), or missing/NaN.
    Mirrors features/feature_builder.py's parse_price() so training-time and
    inference-time price parsing stay consistent.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    try:
        s = str(v).strip()
        return float(s.split("-")[0].strip()) if "-" in s else float(s)
    except (ValueError, TypeError):
        return 0.0


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_batch(
    suppliers: List[Dict[str, Any]],
    query: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build a feature DataFrame for a list of supplier dicts + a query dict.
    Returns a DataFrame with exactly FEATURE_COLS columns (float32).
    Missing source fields default to neutral / zero-contribution values.
    """
    records = []
    query_location = str(query.get("location", "")).lower().strip()
    query_cert     = str(query.get("certification", "")).lower().strip()
    max_price      = parse_price(query.get("max_price", 0))

    for s in suppliers:
        price     = parse_price(s.get("price"))
        location  = str(s.get("supplier_location") or s.get("location", "")).lower().strip()
        certs_raw = s.get("certifications") or s.get("certification") or ""
        certs     = str(certs_raw).lower()

        # price_match: 1 if within budget, 0 otherwise
        price_match = 1.0 if (max_price <= 0 or price <= max_price) else 0.0

        # price_ratio: price / max_price (capped at 2); 0 if no budget
        if max_price > 0 and price > 0:
            price_ratio = min(price / max_price, 2.0)
        else:
            price_ratio = 0.0

        # location_match
        location_match = (
            1.0 if (query_location and query_location in location) else 0.0
        )

        # cert_match
        cert_match = (
            1.0 if (query_cert and query_cert in certs) else 0.0
        )

        # faiss_score & faiss_rank (passed through from retriever)
        faiss_score = float(s.get("faiss_score") or s.get("score") or 0.0)
        faiss_rank  = float(s.get("faiss_rank") or 0.0)

        records.append({
            "price_match":    price_match,
            "price_ratio":    price_ratio,
            "location_match": location_match,
            "cert_match":     cert_match,
            "faiss_score":    faiss_score,
            "faiss_rank":     faiss_rank,
        })

    df = pd.DataFrame(records, columns=FEATURE_COLS).astype(np.float32)
    return df


# ── Ranker class ──────────────────────────────────────────────────────────────

class SupplierRanker:
    """
    Production inference ranker.
    Loads LightGBM (primary) or XGBRanker (fallback) from disk.
    Falls back to a rule-based scorer if no model file is found.
    """

    def __init__(self):
        self.model       = None
        self.model_type  = "rule-based"
        self.use_ml      = False
        self._xgb_feature_cols = None  # set by _load_model if XGB pkl has different feature list
        self._load_model()

    def _load_model(self):
        # Try LightGBM first (primary model from train_lambdarank.py)
        lgbm_path = cfg.LGBM_MODEL
        if lgbm_path.exists():
            try:
                with open(lgbm_path, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict):
                    self.model = payload.get("model", payload)
                else:
                    self.model = payload
                self.model_type = "lightgbm"
                self.use_ml     = True
                logger.info(f"✅ Loaded LightGBM ranker from {lgbm_path}")
                return
            except Exception as e:
                logger.warning(f"⚠️  LightGBM load failed ({e}), trying XGBoost …")

        # Fallback: XGBRanker
        xgb_path = cfg.XGB_MODEL
        if xgb_path.exists():
            try:
                with open(xgb_path, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict):
                    self.model = payload.get("model", payload)
                    # FIX: train_ranker.py trains XGBRanker on 7 features
                    # (FEATURE_COLS + years_normalized + is_manufacturer) and
                    # saves them as payload["feature_cols"]. If we load that
                    # model here but call predict() with only the 5-feature
                    # FEATURE_COLS from this file, XGBoost silently produces
                    # wrong scores (it maps by position, not name). Load the
                    # saved feature list from the pkl so predict() always gets
                    # the right columns in the right order.
                    saved_feats = payload.get("feature_cols")
                    if saved_feats and saved_feats != FEATURE_COLS:
                        logger.warning(
                            f"XGBRanker was trained on {len(saved_feats)} features "
                            f"{saved_feats}, but FEATURE_COLS here has {len(FEATURE_COLS)}. "
                            f"Using saved feature list for inference."
                        )
                        self._xgb_feature_cols = saved_feats
                    else:
                        self._xgb_feature_cols = None  # use default FEATURE_COLS
                else:
                    self.model = payload
                    self._xgb_feature_cols = None
                self.model_type = "xgboost"
                self.use_ml     = True
                logger.info(f"✅ Loaded XGBRanker from {xgb_path}")
                return
            except Exception as e:
                logger.warning(f"⚠️  XGBoost load failed ({e}), using rule-based ranker")

        logger.warning("⚠️  No trained model found — using rule-based ranker")

    def _rule_based_score(self, features: pd.DataFrame) -> np.ndarray:
        """Weighted linear scorer — the production fallback used when no
        trained model is available or model.predict() fails.

        FIX (this version): previously used its own weights (price_match
        0.35 / price_ratio -0.10 / location_match 0.20 / cert_match 0.20
        / faiss_score 0.20), different from the formula actually
        evaluated as "V3: No LTR" / "B3: Rule-Based" in ablation.py /
        baselines.py (0.40/0.30/0.25/0.05, no price_ratio term). That
        meant the live fallback ranker users could actually be served by
        was never the same scorer the paper's NDCG numbers describe.
        Now delegates to rule_baseline.score_rule_based() so production
        behavior matches what's evaluated. NOTE: this changes live
        ranking output when the fallback path is hit (no model file, or
        model.predict() raises) — re-validate before deploying if this
        fallback sees real traffic.
        """
        return _canonical_rule_scorer(features)

    def rank(
        self,
        suppliers: List[Dict[str, Any]],
        query: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Rank `suppliers` for `query`.
        Returns the same list sorted by descending score,
        with a 'score' key added/updated on each dict.
        """
        if not suppliers:
            return suppliers

        scores = self.score_batch(suppliers, query)

        for supplier, s in zip(suppliers, scores):
            supplier["score"] = float(s)

        return sorted(suppliers, key=lambda s: s.get("score", 0.0), reverse=True)

    def score_batch(
        self,
        suppliers: List[Dict[str, Any]],
        query: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute raw relevance scores for `suppliers` against `query`,
        without mutating the input dicts or sorting. Shared by `rank()`
        and the single-supplier `score()` convenience function below.
        """
        if not suppliers:
            return np.array([], dtype=np.float32)

        features = extract_features_batch(suppliers, query)

        if self.use_ml and self.model is not None:
            try:
                # Use saved feature list if it differs from the module default
                # (happens when an XGBRanker trained on more features is loaded).
                active_cols = self._xgb_feature_cols or FEATURE_COLS
                feat_vals = features[active_cols].values.astype(np.float32)
                return self.model.predict(feat_vals)
            except Exception as e:
                logger.warning(f"Model predict failed ({e}); falling back to rule-based")
                return self._rule_based_score(features)
        else:
            return self._rule_based_score(features)

    def score_one(self, supplier: Dict[str, Any], query: Dict[str, Any]) -> float:
        """Score a single supplier dict against a query dict."""
        return float(self.score_batch([supplier], query)[0])


# ── MMR diversification ───────────────────────────────────────────────────────

def apply_mmr(
    ranked: List[Dict[str, Any]],
    top_k: int = 10,
    lambda_param: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance re-ordering for diversity.
    Uses faiss_score + price as a simple 2D embedding proxy.
    Returns up to top_k items.
    """
    if not ranked or top_k <= 0:
        return ranked[:top_k]

    def _vec(s: Dict) -> np.ndarray:
        return np.array([
            float(s.get("faiss_score") or s.get("score") or 0.0),
            min(parse_price(s.get("price")) / 1e5, 1.0),   # normalise price
        ], dtype=np.float32)

    selected: List[Dict] = []
    remaining = list(ranked)

    while remaining and len(selected) < top_k:
        if not selected:
            # First pick: highest relevance score
            best = max(remaining, key=lambda s: float(s.get("score", 0.0)))
        else:
            sel_vecs = np.stack([_vec(s) for s in selected])

            def mmr_score(s: Dict) -> float:
                rel  = float(s.get("score", 0.0))
                v    = _vec(s)
                sims = [float(np.dot(v, sv) / (np.linalg.norm(v) * np.linalg.norm(sv) + 1e-9))
                        for sv in sel_vecs]
                max_sim = max(sims) if sims else 0.0
                return lambda_param * rel - (1 - lambda_param) * max_sim

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return selected


# ── Singleton factory ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_ranker() -> SupplierRanker:
    """Return a cached singleton SupplierRanker."""
    return SupplierRanker()


# ── Module-level convenience wrapper ──────────────────────────────────────────

def score(supplier: Dict[str, Any], query: Dict[str, Any]) -> float:
    """
    Score a single supplier dict against a query dict, using the cached
    singleton ranker. Convenience wrapper around get_ranker().score_one()
    for callers (e.g. backend.app.api.chat) that score suppliers one at a
    time inside a loop rather than batching via rank().
    """
    return get_ranker().score_one(supplier, query)