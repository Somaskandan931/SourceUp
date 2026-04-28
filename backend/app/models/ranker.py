"""
Ranker Module — SourceUp
--------------------------
Hybrid ML + rule-based scoring. All paths from config.cfg.
Supports LightGBM LambdaRank, XGBoost, and rule-based fallback.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Literal
from config import cfg

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

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
    supplier_price  = parse_price(
        supplier.get("price min") or supplier.get("price", 0)
    )
    qmax = query.get("max_price")
    if qmax is not None:
        qmax = float(qmax)
        price_match    = 1.0 if supplier_price > 0 and supplier_price <= qmax else 0.0
        price_ratio    = supplier_price / qmax if qmax > 0 else 1.0
        price_distance = abs(supplier_price - qmax) / qmax if qmax > 0 else 0.0
    else:
        price_match, price_ratio, price_distance = 1.0, 1.0, 0.0

    s_loc = str(supplier.get("supplier location") or supplier.get("location", "")).lower()
    q_loc = str(query.get("location") or "").lower()
    if not s_loc or not q_loc:
        location_match = 0.5
    elif s_loc == q_loc:
        location_match = 1.0
    elif q_loc in s_loc or s_loc in q_loc:
        location_match = 0.5
    else:
        location_match = 0.0

    s_cert = str(supplier.get("certifications") or "").lower()
    q_cert = str(query.get("certification") or "").lower()
    cert_match = (1.0 if q_cert in s_cert else 0.0) if q_cert else 0.5

    years = supplier.get("years with gs", 0) or 0
    try:
        years_normalized = min(float(years) / 10.0, 1.0)
    except Exception:
        years_normalized = 0.0

    biz = str(supplier.get("business type") or "").lower()
    is_manufacturer    = 1.0 if "manufacturer" in biz else 0.0
    is_trading_company = 1.0 if "trading company" in biz else 0.0

    return {
        "price_match":        price_match,
        "price_ratio":        price_ratio,
        "price_distance":     price_distance,
        "location_match":     location_match,
        "cert_match":         cert_match,
        "years_normalized":   years_normalized,
        "is_manufacturer":    is_manufacturer,
        "is_trading_company": is_trading_company,
        "faiss_score":        float(supplier.get("faiss_score", 0.0)),
        "faiss_rank":         int(supplier.get("rank", 999)),
        "supplier_price":     supplier_price,
    }


def extract_features_batch(suppliers: List[Dict], query: Dict) -> pd.DataFrame:
    return pd.DataFrame([extract_features(s, query) for s in suppliers])


class SupplierRanker:
    def __init__(self, model_type: Literal["lightgbm", "xgboost", "auto"] = "auto"):
        self.model      = None
        self.model_type = None
        self.use_ml     = False
        self._load(model_type)

    def _load(self, model_type: str):
        if model_type in ("auto", "lightgbm") and LGBM_AVAILABLE:
            path = str(cfg.LGBM_MODEL)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                self.model_type = "lightgbm"
                self.use_ml = True
                print("✅ LightGBM ranker loaded")
                return
        if model_type in ("auto", "xgboost") and XGB_AVAILABLE:
            path = str(cfg.XGB_MODEL)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                self.model_type = "xgboost"
                self.use_ml = True
                print("✅ XGBoost ranker loaded")
                return
        print("⚠️  No ML model found — rule-based fallback active")

    def rank(self, suppliers: List[Dict], query: Dict) -> List[Dict]:
        if not suppliers:
            return []
        feats = extract_features_batch(suppliers, query)
        scores = (
            self.model.predict(feats[FEATURE_COLS].values)
            if self.use_ml else self._rule_score(feats)
        )
        if self.use_ml:
            scores = np.clip(scores, 0, None)
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
        method = f"ML ({self.model_type})" if self.use_ml else "rule-based"
        for i, s in enumerate(suppliers):
            s["score"]          = float(scores[i])
            s["scoring_method"] = method
            s["reasons"]        = self._reasons(s, query, feats.iloc[i])
        return sorted(suppliers, key=lambda x: x["score"], reverse=True)

    def _rule_score(self, feats: pd.DataFrame) -> np.ndarray:
        return (
            feats["price_match"]          * 0.35 +
            (1 - feats["price_distance"]) * 0.10 +
            feats["location_match"]       * 0.20 +
            feats["cert_match"]           * 0.20 +
            feats["years_normalized"]     * 0.05 +
            feats["is_manufacturer"]      * 0.05 +
            feats["faiss_score"]          * 0.05
        ).values

    def _reasons(self, s: dict, q: dict, f: pd.Series) -> List[str]:
        r = ["Relevant product match"]
        if f["price_match"] > 0:              r.append("Price within budget")
        if f["location_match"] == 1.0:        r.append("Exact location match")
        elif f["location_match"] > 0:         r.append("Partial location match")
        if f["cert_match"] > 0:               r.append(f"{q.get('certification','Required')} certification")
        yrs = s.get("years with gs", 0)
        try:
            if int(float(yrs)) >= 5:          r.append(f"{int(float(yrs))}+ years on platform")
        except Exception:
            pass
        if f["is_manufacturer"] > 0:          r.append("Direct manufacturer")
        elif f["is_trading_company"] > 0:     r.append("Trading company")
        return r


_ranker = None


def get_ranker() -> SupplierRanker:
    global _ranker
    if _ranker is None:
        _ranker = SupplierRanker()
    return _ranker


def score(supplier: dict, query: dict) -> float:
    """Backward-compatible single-supplier scoring."""
    r = get_ranker()
    if r.use_ml:
        ranked = r.rank([supplier], query)
        return ranked[0]["score"] if ranked else 0.0
    f = extract_features(supplier, query)
    return round(
        f["price_match"] * 0.35 +
        (1 - f["price_distance"]) * 0.10 +
        f["location_match"] * 0.20 +
        f["cert_match"] * 0.20 +
        f["years_normalized"] * 0.05 +
        f["is_manufacturer"] * 0.05 +
        f["faiss_score"] * 0.05, 3
    )