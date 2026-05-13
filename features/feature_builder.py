# features/feature_builder.py - CORRECTED IMPORTS

"""
Feature Builder — SourceUp Supplier Ranking (SBERT-Primary)
------------------------------------------------------------
Builds training data ONLY from SBERT-retrieved candidates.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Fix: Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])  # Goes up to SourceUp/
sys.path.insert(0, PROJECT_ROOT)

from config import cfg

# Import SBERT retriever - fix path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Goes up to project root
from backend.app.models.retriever import retrieve

CLEAN_DATA = str(cfg.CLEAN_DATA)
QUERY_FILE = str(cfg.QUERY_FILE)
OUTPUT_FILE = str(cfg.TRAINING_DATA)

cfg.ensure_dirs()


def load_suppliers() -> pd.DataFrame:
    if not os.path.exists(CLEAN_DATA):
        raise FileNotFoundError(f"Cleaned data not found: {CLEAN_DATA}")
    df = pd.read_csv(CLEAN_DATA)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    print(f"✅ Loaded {len(df):,} suppliers")
    return df


def load_queries() -> list:
    if not os.path.exists(QUERY_FILE):
        print(f"⚠️  Query file not found, using defaults")
        return ["plastic containers", "steel pipes", "cotton fabric", "electronic components"]
    df_q = pd.read_csv(QUERY_FILE)
    return df_q.iloc[:, 0].dropna().str.strip().tolist()


def parse_price(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    try:
        s = str(v).strip()
        return float(s.split("-")[0].strip()) if "-" in s else float(s)
    except:
        return 0.0


def compute_relevance(sbert_score: float, price_match: float, location_match: float,
                       cert_match: float, years_norm: float) -> float:
    """60% SBERT + 25% rule + 15% constraint fit."""
    semantic = sbert_score * 0.60
    rule = (price_match * 0.35 + location_match * 0.20 + cert_match * 0.20 + years_norm * 0.15 + 0.10)
    constraint = price_match * 0.4 + location_match * 0.3 + cert_match * 0.3
    return max(0.0, min(1.0, semantic + 0.25 * rule + 0.15 * constraint))


def build_training_data(top_k: int = 50, queries_per_constraint: int = 3, gamma: float = 0.3, hard_negative_ratio: float = 1.0) -> pd.DataFrame:
    """
    Build training dataset using SBERT as PRIMARY retrieval.

    Args:
        top_k: Number of candidates to retrieve per query (default: 50)
        queries_per_constraint: Number of constraint variations per query (default: 3)
        gamma: Not used (kept for compatibility)
        hard_negative_ratio: Not used (kept for compatibility)
    """
    print("=" * 65)
    print("🏗️ Feature Builder (SBERT-Primary Retrieval)")
    print("=" * 65)
    print(f"   top_k: {top_k}")
    print(f"   queries_per_constraint: {queries_per_constraint}")

    df_suppliers = load_suppliers()
    queries = load_queries()

    all_rows = []
    query_id = 0

    for base_query in queries[:10]:  # Limit to 10 queries for faster run
        print(f"\n  Query: {base_query}")

        # PRIMARY: Retrieve via SBERT
        candidates = retrieve(base_query, k=top_k)

        if not candidates:
            print(f"    ⚠️ No candidates retrieved, skipping")
            continue

        print(f"    ✅ Retrieved {len(candidates)} candidates via SBERT")

        for variation in range(queries_per_constraint):
            query_id += 1
            rng = np.random.default_rng(seed=hash(f"{base_query}_{variation}") % 2**31)

            max_price = rng.choice([None, 50, 100, 500])
            target_loc = rng.choice(["", "Mumbai", "Delhi", "Chennai"])
            req_cert = rng.choice(["", "ISO", "FDA"])

            for rank, supplier in enumerate(candidates):
                sbert_score = supplier.get("faiss_score", 0.5)
                price = parse_price(supplier.get("price min") or supplier.get("price", 0))

                # Price features
                if max_price and max_price > 0 and price > 0:
                    price_match = 1.0 if price <= max_price else 0.0
                    price_ratio = min(price / max_price, 2.0)
                    price_distance = abs(price - max_price) / max_price
                else:
                    price_match, price_ratio, price_distance = 0.5, 1.0, 0.0

                # Location
                s_loc = str(supplier.get("supplier location") or supplier.get("location", "")).lower()
                if target_loc:
                    location_match = 1.0 if target_loc.lower() in s_loc else 0.0
                else:
                    location_match = 0.5

                # Certification
                s_cert = str(supplier.get("certifications", "")).lower()
                if req_cert:
                    cert_match = 1.0 if req_cert.lower() in s_cert else 0.0
                else:
                    cert_match = 0.5

                # Years
                years = supplier.get("years with gs", 0) or 0
                try:
                    years_norm = min(float(years) / 10.0, 1.0)
                except:
                    years_norm = 0.0

                biz = str(supplier.get("business type", "")).lower()
                is_manuf = 1.0 if "manufacturer" in biz else 0.0
                is_trading = 1.0 if "trading" in biz else 0.0

                relevance = compute_relevance(sbert_score, price_match, location_match, cert_match, years_norm)

                all_rows.append({
                    "query_id": query_id,
                    "query_text": base_query,
                    "supplier_idx": rank,
                    "price_match": price_match,
                    "price_ratio": price_ratio,
                    "price_distance": price_distance,
                    "location_match": location_match,
                    "cert_match": cert_match,
                    "years_normalized": years_norm,
                    "is_manufacturer": is_manuf,
                    "is_trading_company": is_trading,
                    "faiss_score": sbert_score,
                    "faiss_rank": rank + 1,
                    "relevance": relevance,
                })

        print(f"    📊 Generated {len(candidates) * queries_per_constraint} training pairs")

    if not all_rows:
        raise RuntimeError("No training rows generated")

    df_out = pd.DataFrame(all_rows)

    # Bucket continuous relevance into 0-5 labels
    scores = df_out["relevance"].values
    percentiles = np.percentile(scores, [10, 25, 45, 65, 85])

    def bucket_label(score):
        if score >= percentiles[4]: return 5
        if score >= percentiles[3]: return 4
        if score >= percentiles[2]: return 3
        if score >= percentiles[1]: return 2
        if score >= percentiles[0]: return 1
        return 0

    df_out["relevance"] = df_out["relevance"].apply(bucket_label)

    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Training dataset saved: {OUTPUT_FILE}")
    print(f"   Total pairs: {len(df_out):,}")
    print(f"   Unique queries: {df_out['query_id'].nunique():,}")
    print(f"   Label distribution:\n{df_out['relevance'].value_counts().sort_index()}")

    return df_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--queries-per-constraint", type=int, default=3)
    args = parser.parse_args()

    build_training_data(top_k=args.top_k, queries_per_constraint=args.queries_per_constraint)
    print("\n🎯 Feature building complete!")