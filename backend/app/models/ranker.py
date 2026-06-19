"""
Ranker Module — SourceUp
--------------------------
Hybrid ML + rule-based scoring with ensemble support.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Literal
from config import cfg
from backend.app.utils.fields import get_field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

FEATURE_COLS = [
    "price_match", "price_ratio", "price_distance",
    "location_match", "cert_match", "years_normalized",
    "is_manufacturer", "is_trading_company",
    "faiss_score", "faiss_rank",
]


def parse_price(v) -> float:
    if v is None or (isinstance(v, float) and v != v):
        return 0.0
    try:
        s = str(v).strip()
        return float(s.split("-")[0].strip()) if "-" in s else float(s)
    except Exception:
        return 0.0


def extract_features(supplier: dict, query: dict) -> dict:
    # Use get_field to correctly resolve underscore-normalised keys
    raw_price = get_field(supplier, "price_min") or get_field(supplier, "price", default=0)
    supplier_price = parse_price(raw_price)

    qmax = query.get("max_price")
    if qmax is not None:
        qmax = float(qmax)
        price_match = 1.0 if supplier_price > 0 and supplier_price <= qmax else 0.0
        price_ratio = min(supplier_price / qmax, 2.0) if qmax > 0 else 1.0
        price_distance = abs(supplier_price - qmax) / qmax if qmax > 0 else 0.0
    else:
        price_match, price_ratio, price_distance = 0.5, 1.0, 0.0

    s_loc = str(get_field(supplier, "supplier_location") or get_field(supplier, "location", default="")).lower().strip()
    q_loc = str(query.get("location") or "").lower().strip()

    if not s_loc or not q_loc:
        location_match = 0.5
    elif q_loc in s_loc or s_loc in q_loc:
        location_match = 1.0
    else:
        location_match = 0.0

    s_cert = str(get_field(supplier, "certifications", default="") or "").lower()
    q_cert = str(query.get("certification") or "").lower()
    cert_match = (1.0 if q_cert in s_cert else 0.0) if q_cert else 0.5

    years = get_field(supplier, "years_with_gs", "years_on_platform", default=0) or 0
    try:
        years_normalized = min(float(years) / 10.0, 1.0)
    except Exception:
        years_normalized = 0.0

    biz = str(get_field(supplier, "business_type", default="") or "").lower()
    is_manufacturer = 1.0 if "manufacturer" in biz else 0.0
    is_trading_company = 1.0 if "trading company" in biz else 0.0

    return {
        "price_match": price_match,
        "price_ratio": price_ratio,
        "price_distance": price_distance,
        "location_match": location_match,
        "cert_match": cert_match,
        "years_normalized": years_normalized,
        "is_manufacturer": is_manufacturer,
        "is_trading_company": is_trading_company,
        "faiss_score": float(supplier.get("semantic_score") or supplier.get("faiss_score", 0.0)),
        "faiss_rank": int(supplier.get("faiss_rank") or supplier.get("rank", 999)),
        "supplier_price": supplier_price,
    }


def apply_constraint_penalty(score: float, supplier: dict, gamma: float = 0.5) -> float:
    if supplier.get("constraint_violated", False):
        return score - gamma * (gamma ** 2 + 1)
    return score


def extract_features_batch(suppliers: List[Dict], query: Dict) -> pd.DataFrame:
    return pd.DataFrame([extract_features(s, query) for s in suppliers])


def rescale_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(scores, 0.0, 1.0)
    batch_min = scores.min()
    batch_max = scores.max()
    if batch_max - batch_min > 1e-6:
        return 0.35 + (scores - batch_min) / (batch_max - batch_min) * 0.60
    return np.full_like(scores, 0.70)


def rule_based_score(feats: pd.DataFrame) -> np.ndarray:
    raw = (
        feats["price_match"] * 0.25 +
        (1 - feats["price_distance"]) * 0.10 +
        feats["location_match"] * 0.15 +
        feats["cert_match"] * 0.15 +
        feats["years_normalized"] * 0.10 +
        feats["is_manufacturer"] * 0.05 +
        feats["faiss_score"] * 0.20
    ).values
    return np.clip(raw, 0.0, 1.0)


class SupplierRanker:
    def __init__(self, model_type: Literal["lightgbm", "xgboost", "auto"] = "auto"):
        self.model = None
        self.model_type = None
        self.use_ml = False
        self._load(model_type)

    def _load(self, model_type: str):
        if model_type in ("auto", "lightgbm") and LGBM_AVAILABLE:
            path = str(cfg.LGBM_MODEL)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        self.model = pickle.load(f)
                    self.model_type = "lightgbm"
                    self.use_ml = True
                    logger.info(f"✅ LightGBM ranker loaded from {path}")
                    return
                except Exception as e:
                    logger.error(f"Failed to load LightGBM model: {e}")
        logger.warning("⚠️ No ML model found — using rule-based fallback")
        self.use_ml = False

    def rank(self, suppliers: List[Dict], query: Dict) -> List[Dict]:
        if not suppliers:
            return []

        feats = extract_features_batch(suppliers, query)

        if self.use_ml and self.model is not None:
            try:
                raw_scores = self.model.predict(feats[FEATURE_COLS].values)
                raw_scores = np.clip(raw_scores, 0, 1)
            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
                raw_scores = rule_based_score(feats)
        else:
            raw_scores = rule_based_score(feats)

        scores = rescale_scores(raw_scores)

        for i, s in enumerate(suppliers):
            raw_score = float(raw_scores[i])
            s["score"] = apply_constraint_penalty(float(scores[i]), s)
            s["raw_score"] = raw_score
            s["reasons"] = self._reasons(s, query, feats.iloc[i])

        return sorted(suppliers, key=lambda x: x["score"], reverse=True)

    def _reasons(self, s: dict, q: dict, f: pd.Series) -> List[str]:
        r = ["Relevant product match"]
        if f["price_match"] > 0:
            r.append("Price within budget")
        if f["location_match"] == 1.0:
            r.append("Exact location match")
        elif f["location_match"] > 0:
            r.append("Partial location match")
        if f["cert_match"] > 0:
            r.append(f"{q.get('certification','Required')} certification")
        yrs = get_field(s, "years_with_gs", "years_on_platform", default=0) or 0
        try:
            if int(float(yrs)) >= 5:
                r.append(f"{int(float(yrs))}+ years on platform")
        except Exception:
            pass
        if f["is_manufacturer"] > 0:
            r.append("Direct manufacturer")
        return r


_ranker = None


def get_ranker() -> SupplierRanker:
    global _ranker
    if _ranker is None:
        _ranker = SupplierRanker()
    return _ranker


def score(supplier: dict, query: dict) -> float:
    r = get_ranker()
    ranked = r.rank([supplier], query)
    return ranked[0]["score"] if ranked else 0.0
