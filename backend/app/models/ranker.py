"""
Ranker Module - Hybrid ML + Rule-Based Scoring
-----------------------------------------------
Supports LightGBM, XGBoost, and rule-based fallback.
Handles GlobalSources data structure with price ranges.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Literal

# Optional ML imports - graceful degradation if not installed
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

# Paths
BASE_DIR = "C:/Users/somas/PycharmProjects/SourceUp"
MODELS_DIR = f"{BASE_DIR}/backend/app/models/embeddings"
LGBM_MODEL_PATH = f"{MODELS_DIR}/ranker_lightgbm.pkl"
XGB_MODEL_PATH = f"{MODELS_DIR}/ranker_xgboost.pkl"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_price(price_value) -> float:
    """
    Parse price from various formats:
    - "0.28 - 0.31" -> 0.28 (use min price)
    - "1.50" -> 1.50
    - 1.50 -> 1.50
    - None/NaN -> 0
    """
    if price_value is None or (isinstance(price_value, float) and price_value != price_value):
        return 0.0

    try:
        if isinstance(price_value, (int, float)):
            return float(price_value)

        price_str = str(price_value).strip()

        if '-' in price_str:
            parts = price_str.split('-')
            return float(parts[0].strip())

        return float(price_str)

    except (ValueError, AttributeError):
        return 0.0


# ============================================================================
# RULE-BASED SCORING (Fallback when ML is unavailable)
# ============================================================================

def score_rule_based(supplier: dict, query: dict) -> float:
    """
    Original rule-based scoring logic.
    Used as fallback when ML models are unavailable.

    FIXED: Handles None values properly.

    Args:
        supplier: Supplier data dictionary with GlobalSources columns
        query: Query parameters dictionary

    Returns:
        Float score between 0 and 1
    """
    total_score = 0.0

    # Price match (40% weight) - FIXED to handle None
    supplier_price = supplier.get("price min", None)
    if supplier_price is None or (isinstance(supplier_price, float) and supplier_price != supplier_price):
        supplier_price = parse_price(supplier.get("price", 0))
    else:
        supplier_price = float(supplier_price)

    query_max_price = query.get("max_price")  # FIXED: Can be None

    # FIXED: Check for None before comparison
    if query_max_price is not None and supplier_price > 0:
        if supplier_price <= query_max_price:
            total_score += 0.4

    # Location match (20% weight)
    supplier_location = str(supplier.get("supplier location") or supplier.get("location", "")).lower().strip()
    query_location = str(query.get("location", "")).lower().strip()

    if supplier_location and query_location:
        if supplier_location == query_location:
            total_score += 0.2
        elif query_location in supplier_location or supplier_location in query_location:
            total_score += 0.15

    # Certification match (30% weight)
    supplier_certs = str(supplier.get("certifications", "")).lower()
    query_cert = str(query.get("certification", "")).lower().strip()

    if query_cert and query_cert in supplier_certs:
        total_score += 0.3

    # Years with platform bonus (5% weight)
    years = supplier.get("years with gs", 0)
    if years:
        try:
            years_int = int(years)
            total_score += min(years_int / 10, 1.0) * 0.05
        except:
            pass

    # Business type bonus (5% weight)
    business_type = str(supplier.get("business type", "")).lower()
    if "manufacturer" in business_type:
        total_score += 0.05
    elif "trading company" in business_type:
        total_score += 0.02

    return round(total_score, 3)


# ============================================================================
# FEATURE ENGINEERING (For ML Models)
# ============================================================================

def extract_features(supplier: dict, query: dict) -> Dict[str, float]:
    """
    Extract numerical features for ML models.

    FIXED: Handles None values properly to avoid comparison errors.

    Returns a feature dictionary compatible with both LightGBM and XGBoost.
    """
    # ========================================================================
    # PRICE FEATURES - Fixed to handle None values
    # ========================================================================

    # Parse supplier price
    supplier_price = supplier.get("price min", None)
    if supplier_price is None or (isinstance(supplier_price, float) and np.isnan(supplier_price)):
        supplier_price = parse_price(supplier.get("price", 0))
    else:
        supplier_price = float(supplier_price)

    # Get query max price (can be None!)
    query_max_price = query.get("max_price")

    # Calculate price features with proper None handling
    if query_max_price is not None:
        # User specified a price filter
        query_max_price = float(query_max_price)

        if supplier_price > 0:
            price_match = 1.0 if supplier_price <= query_max_price else 0.0
            price_ratio = supplier_price / query_max_price if query_max_price > 0 else 1.0
            price_distance = abs(supplier_price - query_max_price) / query_max_price if query_max_price > 0 else 0.0
        else:
            # Supplier has no price
            price_match = 0.0
            price_ratio = 1.0
            price_distance = 1.0
    else:
        # No price filter - neutral values
        price_match = 1.0
        price_ratio = 1.0
        price_distance = 0.0

    # ========================================================================
    # LOCATION FEATURES - Fixed to handle None/empty values
    # ========================================================================

    supplier_location = supplier.get("supplier location") or supplier.get("location", "")
    query_location = query.get("location", "")

    # Convert None to empty string
    if supplier_location is None:
        supplier_location = ""
    if query_location is None:
        query_location = ""

    supplier_location = str(supplier_location).lower().strip()
    query_location = str(query_location).lower().strip()

    if not supplier_location or not query_location:
        location_match = 0.5  # Neutral when no location filter
    elif supplier_location == query_location:
        location_match = 1.0
    elif query_location in supplier_location or supplier_location in query_location:
        location_match = 0.5
    else:
        location_match = 0.0

    # ========================================================================
    # CERTIFICATION FEATURES - Fixed to handle None values
    # ========================================================================

    supplier_certs = supplier.get("certifications", "")
    query_cert = query.get("certification", "")

    # Convert None to empty string
    if supplier_certs is None:
        supplier_certs = ""
    if query_cert is None:
        query_cert = ""

    supplier_certs = str(supplier_certs).lower()
    query_cert = str(query_cert).lower().strip()

    if query_cert:
        cert_match = 1.0 if query_cert in supplier_certs else 0.0
    else:
        cert_match = 0.5  # Neutral when no cert filter

    # ========================================================================
    # EXPERIENCE FEATURES - Fixed to handle None values
    # ========================================================================

    years = supplier.get("years with gs", 0)
    if years is None:
        years = 0

    try:
        years_normalized = min(float(years) / 10.0, 1.0) if years else 0.0
    except (ValueError, TypeError):
        years_normalized = 0.0

    # ========================================================================
    # BUSINESS TYPE FEATURES - Fixed to handle None values
    # ========================================================================

    business_type = supplier.get("business type", "")
    if business_type is None:
        business_type = ""
    business_type = str(business_type).lower()

    is_manufacturer = 1.0 if "manufacturer" in business_type else 0.0
    is_trading_company = 1.0 if "trading company" in business_type else 0.0

    # ========================================================================
    # SEMANTIC SIMILARITY FROM FAISS
    # ========================================================================

    faiss_score = supplier.get("faiss_score", 0.0)
    faiss_rank = supplier.get("rank", 999)

    return {
        'price_match': price_match,
        'price_ratio': price_ratio,
        'price_distance': price_distance,
        'location_match': location_match,
        'cert_match': cert_match,
        'years_normalized': years_normalized,
        'is_manufacturer': is_manufacturer,
        'is_trading_company': is_trading_company,
        'faiss_score': faiss_score,
        'faiss_rank': faiss_rank,
        'supplier_price': supplier_price
    }


def extract_features_batch(suppliers: List[Dict], query: Dict) -> pd.DataFrame:
    """Extract features for multiple suppliers at once."""
    features = [extract_features(s, query) for s in suppliers]
    return pd.DataFrame(features)


# ============================================================================
# ML-BASED RANKING
# ============================================================================

class SupplierRanker:
    """
    Hybrid ranker supporting LightGBM, XGBoost, and rule-based fallback.
    """

    def __init__(self, model_type: Literal['lightgbm', 'xgboost', 'auto'] = 'auto'):
        """
        Initialize ranker.

        Args:
            model_type: 'lightgbm', 'xgboost', or 'auto' (tries both)
        """
        self.model = None
        self.model_type = None
        self.use_ml = False

        # Feature columns (must match training)
        self.feature_cols = [
            'price_match', 'price_ratio', 'price_distance',
            'location_match', 'cert_match', 'years_normalized',
            'is_manufacturer', 'is_trading_company',
            'faiss_score', 'faiss_rank'
        ]

        # Try to load model
        self._load_model(model_type)

    def _load_model(self, model_type: str):
        """Load trained ML model."""
        if model_type == 'auto':
            # Try LightGBM first, then XGBoost
            if self._try_load_lightgbm():
                return
            if self._try_load_xgboost():
                return
            print("⚠️ No trained ML models found. Using rule-based fallback.")

        elif model_type == 'lightgbm':
            if not self._try_load_lightgbm():
                print("⚠️ LightGBM model not found. Using rule-based fallback.")

        elif model_type == 'xgboost':
            if not self._try_load_xgboost():
                print("⚠️ XGBoost model not found. Using rule-based fallback.")

    def _try_load_lightgbm(self) -> bool:
        """Try to load LightGBM model."""
        if not LGBM_AVAILABLE:
            return False

        if not os.path.exists(LGBM_MODEL_PATH):
            return False

        try:
            with open(LGBM_MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            self.model_type = 'lightgbm'
            self.use_ml = True
            print("✅ Loaded LightGBM ranking model")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load LightGBM model: {e}")
            return False

    def _try_load_xgboost(self) -> bool:
        """Try to load XGBoost model."""
        if not XGB_AVAILABLE:
            return False

        if not os.path.exists(XGB_MODEL_PATH):
            return False

        try:
            with open(XGB_MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            self.model_type = 'xgboost'
            self.use_ml = True
            print("✅ Loaded XGBoost ranking model")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load XGBoost model: {e}")
            return False

    def rank(self, suppliers: List[Dict], query: Dict) -> List[Dict]:
        """
        Rank suppliers using ML or rule-based scoring.

        Args:
            suppliers: List of supplier dictionaries
            query: Query parameters

        Returns:
            Ranked list with scores and reasons
        """
        if not suppliers:
            return []

        # Extract features
        features_df = extract_features_batch(suppliers, query)

        # Score using ML or rules
        if self.use_ml and self.model is not None:
            scores = self._ml_score(features_df)
            method = f"ML ({self.model_type})"
        else:
            scores = self._rule_score(features_df)
            method = "Rule-based"

        # Add scores and reasons
        ranked_suppliers = []
        for i, supplier in enumerate(suppliers):
            supplier['score'] = float(scores[i])
            supplier['scoring_method'] = method
            supplier['reasons'] = self._generate_reasons(
                supplier,
                query,
                features_df.iloc[i]
            )
            ranked_suppliers.append(supplier)

        # Sort by score descending
        ranked_suppliers.sort(key=lambda x: x['score'], reverse=True)

        return ranked_suppliers

    def _ml_score(self, features_df: pd.DataFrame) -> np.ndarray:
        """Score using trained ML model."""
        X = features_df[self.feature_cols].values

        # Predict scores
        scores = self.model.predict(X)

        # Normalize to 0-1 range
        scores = np.clip(scores, 0, None)  # Remove negative scores
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    def _rule_score(self, features_df: pd.DataFrame) -> np.ndarray:
        """Rule-based scoring using engineered features."""
        scores = (
            features_df['price_match'] * 0.35 +
            (1 - features_df['price_distance']) * 0.10 +
            features_df['location_match'] * 0.20 +
            features_df['cert_match'] * 0.20 +
            features_df['years_normalized'] * 0.05 +
            features_df['is_manufacturer'] * 0.05 +
            features_df['faiss_score'] * 0.05
        )
        return scores.values

    def _generate_reasons(self, supplier: Dict, query: Dict, features: pd.Series) -> List[str]:
        """Generate human-readable reasons for ranking."""
        reasons = []

        # Always include semantic match
        reasons.append("Relevant product match")

        # Price
        if features['price_match'] > 0:
            reasons.append("Price within budget")

        # Location
        if features['location_match'] == 1.0:
            reasons.append("Exact location match")
        elif features['location_match'] > 0:
            reasons.append("Partial location match")

        # Certification
        if features['cert_match'] > 0:
            cert = query.get('certification', 'Required')
            reasons.append(f"{cert} certification")

        # Experience
        years = supplier.get("years with gs", 0)
        if years:
            try:
                years_int = int(years)
                if years_int >= 5:
                    reasons.append(f"{years_int}+ years on platform")
            except:
                pass

        # Business type
        if features['is_manufacturer'] > 0:
            reasons.append("Direct manufacturer")
        elif features['is_trading_company'] > 0:
            reasons.append("Trading company")

        return reasons


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_ranker_instance = None

def get_ranker(model_type: Literal['lightgbm', 'xgboost', 'auto'] = 'auto') -> SupplierRanker:
    """Get or create ranker singleton."""
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = SupplierRanker(model_type=model_type)
    return _ranker_instance


# ============================================================================
# BACKWARD COMPATIBLE API
# ============================================================================

def score(supplier: dict, query: dict) -> float:
    """
    Legacy function for backward compatibility.

    Delegates to ML ranker if available, otherwise uses rule-based scoring.
    Your existing code calling score() will continue to work.

    Args:
        supplier: Supplier dictionary
        query: Query dictionary

    Returns:
        Score between 0 and 1
    """
    ranker = get_ranker()

    # If ML is available, use it
    if ranker.use_ml:
        ranked = ranker.rank([supplier], query)
        return ranked[0]['score'] if ranked else 0.0
    else:
        # Fallback to original rule-based scoring
        return score_rule_based(supplier, query)