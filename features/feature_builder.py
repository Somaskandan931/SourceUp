"""
Feature Builder — SourceUp Supplier Ranking
--------------------------------------------
Constructs the learning-to-rank training dataset from:
  1. The cleaned supplier database (suppliers_clean.csv)
  2. A set of representative sourcing queries (search_query.csv)
  3. The pre-built FAISS index (suppliers.faiss + suppliers_meta.csv)

For each (query, supplier) pair, it computes 10 features and assigns
a weakly-supervised relevance label (0–5 scale) from the composite
score heuristic used in App.java.

Output: data/training/ranking_data.csv
  Columns: query_id, query_text, [10 features], relevance, composite_score

Run this ONCE before training:
    python features/features/feature_builder.py

IEEE justification for weak supervision:
    Labels are derived from the same scoring function used in production
    (price, delivery, reliability). This is the standard approach for
    bootstrapping L2R datasets without explicit human judgments.
    The label_noise_analysis.py experiment validates robustness.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — all from config.cfg (no hardcoded paths)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import cfg

CLEAN_DATA  = str(cfg.CLEAN_DATA)
QUERY_FILE  = str(cfg.QUERY_FILE)
INDEX_FILE  = str(cfg.FAISS_INDEX)
META_FILE   = str(cfg.FAISS_META)
OUTPUT_FILE = str(cfg.TRAINING_DATA)
SCHEMA_FILE = str(cfg.SCHEMA_FILE)

cfg.ensure_dirs()

# ---------------------------------------------------------------------------
# FAISS / SBERT imports (optional — degrades gracefully)
# ---------------------------------------------------------------------------
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss / sentence_transformers not available — "
          "FAISS scores will be approximated from text overlap")

# ---------------------------------------------------------------------------
# Schema column mappings
# Adjust these to match your test_output.csv canonical column names
# ---------------------------------------------------------------------------
COL_PRICE          = "price"
COL_MOQ            = "min order qty"
COL_LOCATION       = "location"
COL_CERTIFICATIONS = "certifications"
COL_YEARS          = "years on platform"
COL_BIZ_TYPE       = "business type"
COL_COMPOSITE      = "composite score"
COL_PRODUCT        = "product name"
COL_COMPANY        = "company name"
COL_RELIABILITY    = "reliability score"
COL_DELIVERY_DAYS  = "est delivery days"
COL_DELIVERY_SCORE = "delivery score"
COL_PRICE_SCORE    = "price score"

# Metro cities for location scoring
METRO_CITIES = {
    "mumbai": 0.95, "delhi": 0.92, "new delhi": 0.92, "chennai": 0.90,
    "bangalore": 0.88, "bengaluru": 0.88, "hyderabad": 0.87,
    "kolkata": 0.85, "pune": 0.85, "ahmedabad": 0.83,
    "noida": 0.82,  "thane": 0.82, "surat": 0.80, "vadodara": 0.76,
}

# Recognised certification keywords
KNOWN_CERTS = {"iso", "fda", "ce", "rohs", "ul", "bis", "gst", "iec"}


# ============================================================================
# LOADERS
# ============================================================================

def load_schema() -> list:
    """Load canonical column names from test_output.csv."""
    if os.path.exists(SCHEMA_FILE):
        return [c.strip().lower() for c in pd.read_csv(SCHEMA_FILE, nrows=0).columns]
    return []


def load_suppliers() -> pd.DataFrame:
    """Load cleaned supplier database."""
    if not os.path.exists(CLEAN_DATA):
        raise FileNotFoundError(
            f"Cleaned data not found: {CLEAN_DATA}\n"
            "Run: python pipeline/run_all.py"
        )
    df = pd.read_csv(CLEAN_DATA)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"✅ Loaded {len(df)} suppliers from suppliers_clean.csv")
    return df


def load_queries() -> list:
    """Load query list from search_query.csv."""
    if not os.path.exists(QUERY_FILE):
        print(f"⚠️  Query file not found: {QUERY_FILE}")
        print("   Using synthetic queries for demo purposes.")
        return _synthetic_queries()
    df_q = pd.read_csv(QUERY_FILE)
    queries = df_q.iloc[:, 0].dropna().str.strip().tolist()
    queries = [q for q in queries if q]
    print(f"✅ Loaded {len(queries)} queries from search_query.csv")
    return queries


def _synthetic_queries() -> list:
    """Fallback queries for testing when search_query.csv is absent."""
    return [
        "plastic containers food grade",
        "steel pipes industrial grade",
        "cotton fabric wholesale",
        "electrical cables copper",
        "ceramic tiles floor",
        "solar panels photovoltaic",
        "packaging materials corrugated",
        "leather bags genuine",
        "wooden furniture office",
        "chemical solvents industrial",
    ]


def load_faiss():
    """Load FAISS index and metadata."""
    if not FAISS_AVAILABLE:
        return None, None
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        print(f"⚠️  FAISS index not found — run: python pipeline/run_all.py")
        return None, None
    index   = faiss.read_index(INDEX_FILE)
    df_meta = pd.read_csv(META_FILE)
    df_meta.columns = [c.strip().lower() for c in df_meta.columns]
    print(f"✅ FAISS index loaded ({index.ntotal} vectors)")
    return index, df_meta


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def safe_float(val, default: float = 0.0) -> float:
    """Convert to float safely."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        s = str(val).strip()
        if "-" in s and not s.startswith("-"):
            return float(s.split("-")[0].strip())
        return float(s.replace(",", ""))
    except Exception:
        return default


def parse_location_score(location: str) -> float:
    """Delivery score from city lookup."""
    if not isinstance(location, str):
        return 0.65
    loc_l = location.lower()
    for city, score in METRO_CITIES.items():
        if city in loc_l:
            return score
    return 0.65


def has_certification(cert_string: str, required_cert: str) -> float:
    """
    Returns 1.0 if cert_string contains required_cert,
            0.5 if no cert is required,
            0.0 if cert is required but missing.
    """
    if not required_cert or str(required_cert).strip() == "":
        return 0.5
    if not isinstance(cert_string, str):
        return 0.0
    return 1.0 if required_cert.lower() in cert_string.lower() else 0.0


def compute_faiss_scores(query: str, index, df_meta: pd.DataFrame,
                          top_k: int = 50) -> pd.DataFrame:
    """
    Retrieve top-k FAISS neighbours for a query.
    Returns df with columns: meta_idx, faiss_score, faiss_rank.
    """
    if index is None or df_meta is None:
        return pd.DataFrame(columns=["meta_idx", "faiss_score", "faiss_rank"])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_emb, min(top_k, index.ntotal))

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx < 0:
            continue
        score = 1.0 / (1.0 + dist)
        results.append({
            "meta_idx":    idx,
            "faiss_score": score,
            "faiss_rank":  rank + 1,
        })
    return pd.DataFrame(results)


def extract_features(supplier_row: pd.Series, query: str,
                      max_price: float = None, target_location: str = None,
                      required_cert: str = None,
                      faiss_score: float = 0.5,
                      faiss_rank: int = 500) -> dict:
    """Compute the 10-feature vector for a (query, supplier) pair."""
    # Price features
    price = safe_float(supplier_row.get(COL_PRICE, 0))
    if max_price and max_price > 0 and price > 0:
        price_match    = 1.0 if price <= max_price else 0.0
        price_ratio    = min(price / max_price, 2.0)
        price_distance = abs(price - max_price) / max_price
    else:
        price_match    = 0.5
        price_ratio    = 1.0
        price_distance = 0.0

    # Location features
    location = str(supplier_row.get(COL_LOCATION, "India"))
    if target_location and target_location.strip():
        location_match = (
            1.0 if target_location.lower() in location.lower()
            else parse_location_score(location) * 0.5
        )
    else:
        location_match = 0.5

    # Certification
    cert_string = str(supplier_row.get(COL_CERTIFICATIONS, ""))
    cert_match  = has_certification(cert_string, required_cert or "")

    # Supplier quality
    years_raw        = safe_float(supplier_row.get(COL_YEARS, 0))
    years_normalized = min(years_raw / 10.0, 1.0)

    biz_type = str(supplier_row.get(COL_BIZ_TYPE, "")).lower()
    is_manufacturer    = 1.0 if "manufacturer" in biz_type else 0.0
    is_trading_company = 1.0 if "trading" in biz_type else 0.0

    return {
        "price_match":        price_match,
        "price_ratio":        price_ratio,
        "price_distance":     price_distance,
        "location_match":     location_match,
        "cert_match":         cert_match,
        "years_normalized":   years_normalized,
        "is_manufacturer":    is_manufacturer,
        "is_trading_company": is_trading_company,
        "faiss_score":        faiss_score,
        "faiss_rank":         faiss_rank,
    }


def compute_relevance_label(supplier_row: pd.Series, features: dict,
                              gamma: float = 0.3) -> float:
    """
    Derive relevance label from composite_score in the data, or
    recompute it from raw features if the column is absent.
    Scale: 0–5 (integer labels for LambdaRank).
    """
    if COL_COMPOSITE in supplier_row.index:
        composite = safe_float(supplier_row[COL_COMPOSITE], -1)
        if composite >= 0:
            violation = (
                (1.0 - features["price_match"])    * 0.5 +
                (1.0 - features["location_match"]) * 0.3 +
                (1.0 - features["cert_match"])     * 0.2
            )
            violation  = 0.0 if features["price_match"] == 0.5 else violation
            penalised  = composite - gamma * violation
            return max(0.0, min(5.0, penalised * 5.0))

    price_s     = safe_float(supplier_row.get(COL_PRICE_SCORE, 0.5))
    delivery_s  = safe_float(supplier_row.get(COL_DELIVERY_SCORE, 0.65))
    reliability = safe_float(supplier_row.get(COL_RELIABILITY, 0.5))
    score       = 0.35 * price_s + 0.25 * delivery_s + 0.40 * reliability
    return max(0.0, min(1.0, score)) * 5.0


# ============================================================================
# QUERY SAMPLING
# ============================================================================

CERT_OPTIONS     = ["ISO", "FDA", "CE", "RoHS", ""]
LOCATION_OPTIONS = ["India", "Mumbai", "Delhi", "Chennai", "Bengaluru", ""]


def sample_constraints(query: str, seed: int = None) -> dict:
    """Randomly sample constraint parameters to create diverse training pairs."""
    rng = np.random.default_rng(seed)
    max_price = float(rng.choice([None, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0,
                                   500.0, 1000.0]))
    location  = str(rng.choice(LOCATION_OPTIONS)) if rng.random() < 0.40 else ""
    cert      = str(rng.choice(CERT_OPTIONS[:-1])) if rng.random() < 0.30 else ""
    return {"max_price": max_price, "target_location": location, "required_cert": cert}


# ============================================================================
# MAIN BUILDER
# ============================================================================

def build_training_data(top_k_faiss: int = 50,
                         queries_per_constraint: int = 3,
                         gamma: float = 0.3) -> pd.DataFrame:
    """
    Build the full (query, supplier, features, label) training dataset.
    """
    print("=" * 65)
    print("🏗️  SourceUp — Feature Builder")
    print("=" * 65)

    df_suppliers = load_suppliers()
    queries      = load_queries()
    faiss_index, df_meta = load_faiss()

    all_rows = []
    query_id = 0

    for base_query in queries:
        print(f"\n  Query: {base_query}")
        df_faiss = compute_faiss_scores(base_query, faiss_index, df_meta, top_k_faiss)

        if df_faiss.empty:
            sample_size = min(top_k_faiss, len(df_suppliers))
            candidate_indices = np.random.default_rng(hash(base_query) % 2**31).choice(
                len(df_suppliers), sample_size, replace=False
            )
            df_faiss = pd.DataFrame({
                "meta_idx":    candidate_indices,
                "faiss_score": np.random.uniform(0.3, 0.8, sample_size),
                "faiss_rank":  np.arange(1, sample_size + 1),
            })

        for variation in range(queries_per_constraint):
            constraints = sample_constraints(
                base_query,
                seed=hash(f"{base_query}_{variation}") % 2**31
            )
            query_id += 1

            for _, faiss_row in df_faiss.iterrows():
                idx     = int(faiss_row["meta_idx"])
                f_score = float(faiss_row["faiss_score"])
                f_rank  = int(faiss_row["faiss_rank"])

                if df_meta is not None and idx < len(df_meta):
                    supplier = df_meta.iloc[idx]
                elif idx < len(df_suppliers):
                    supplier = df_suppliers.iloc[idx]
                else:
                    continue

                features  = extract_features(
                    supplier, base_query,
                    max_price       = constraints.get("max_price"),
                    target_location = constraints.get("target_location", ""),
                    required_cert   = constraints.get("required_cert", ""),
                    faiss_score     = f_score,
                    faiss_rank      = f_rank,
                )
                relevance = compute_relevance_label(supplier, features, gamma)

                all_rows.append({
                    "query_id":        query_id,
                    "query_text":      base_query,
                    "max_price":       constraints.get("max_price", ""),
                    "target_location": constraints.get("target_location", ""),
                    "required_cert":   constraints.get("required_cert", ""),
                    "supplier_idx":    idx,
                    **features,
                    "relevance":       round(relevance, 4),
                    "composite_score": safe_float(supplier.get(COL_COMPOSITE, 0), 0),
                })

        if len(all_rows) % 500 == 0 and all_rows:
            print(f"    {len(all_rows)} pairs generated so far...")

    if not all_rows:
        raise RuntimeError(
            "No training rows generated. "
            "Check that suppliers_clean.csv and search_query.csv exist."
        )

    df_out = pd.DataFrame(all_rows)

    # Validate and clip
    feature_cols = [
        "price_match", "price_ratio", "price_distance",
        "location_match", "cert_match", "years_normalized",
        "is_manufacturer", "is_trading_company",
        "faiss_score", "faiss_rank", "relevance"
    ]
    for col in feature_cols:
        if col not in df_out.columns:
            raise ValueError(f"Missing column in output: {col}")

    df_out["relevance"] = df_out["relevance"].clip(0, 5)
    for col in ["price_match", "location_match", "cert_match",
                "faiss_score", "years_normalized"]:
        df_out[col] = df_out[col].clip(0.0, 1.0)

    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Training dataset saved: {OUTPUT_FILE}")
    print(f"   Total pairs:    {len(df_out)}")
    print(f"   Unique queries: {df_out['query_id'].nunique()}")
    print(f"   Relevance distribution:\n"
          f"{df_out['relevance'].round().astype(int).value_counts().sort_index()}")

    return df_out


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SourceUp training features")
    parser.add_argument("--top-k",      type=int,   default=50,
                        help="FAISS top-k per query (default: 50)")
    parser.add_argument("--variations", type=int,   default=3,
                        help="Constraint variations per query (default: 3)")
    parser.add_argument("--gamma",      type=float, default=0.3,
                        help="Constraint penalty weight for labels (default: 0.3)")
    args = parser.parse_args()

    build_training_data(
        top_k_faiss=args.top_k,
        queries_per_constraint=args.variations,
        gamma=args.gamma,
    )
    print("\n🎯 Feature building complete.")
