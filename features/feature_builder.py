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
from backend.app.utils.fields import get_field
from backend.app.models.retriever import retrieve

CLEAN_DATA = str(cfg.CLEAN_DATA)
QUERY_FILE = str(cfg.QUERY_FILE)
OUTPUT_FILE = str(cfg.TRAINING_DATA)

cfg.ensure_dirs()

# ── Tier definitions ──────────────────────────────────────────────────────────
METRO_CITIES = {
    "mumbai", "delhi", "chennai", "bangalore", "bengaluru",
    "kolkata", "hyderabad", "pune", "ahmedabad",
}

TIER2_CITIES = {
    "surat", "jaipur", "lucknow", "kanpur", "nagpur",
    "indore", "bhopal", "visakhapatnam", "patna", "vadodara",
    "ludhiana", "agra", "nashik", "faridabad", "meerut",
    "rajkot", "varanasi", "amritsar", "allahabad", "coimbatore",
}

# Constraint location pool includes Metro + Tier-2 cities
CONSTRAINT_LOCATIONS = [""] + sorted(METRO_CITIES | TIER2_CITIES)


def _classify_tier(location: str) -> str:
    """Return 'metro', 'tier2', or 'international' for a location string."""
    loc = str(location).strip().lower()
    if any(city in loc for city in METRO_CITIES):
        return "metro"
    if any(city in loc for city in TIER2_CITIES):
        return "tier2"
    return "international"


def load_suppliers() -> pd.DataFrame:
    if not os.path.exists(CLEAN_DATA):
        raise FileNotFoundError(f"Cleaned data not found: {CLEAN_DATA}")
    df = pd.read_csv(CLEAN_DATA)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Classify each supplier into a tier for stratified sampling
    loc_col = "location" if "location" in df.columns else "supplier_location"
    df["_tier"] = df[loc_col].fillna("").apply(_classify_tier)

    counts = df["_tier"].value_counts()
    print(f"✅ Loaded {len(df):,} suppliers")
    print(f"   Tier breakdown — metro: {counts.get('metro', 0):,}  "
          f"tier2: {counts.get('tier2', 0):,}  "
          f"international: {counts.get('international', 0):,}")
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


def _stratified_hard_negatives(
    df_suppliers: pd.DataFrame,
    n: int,
    seed: int = 42,
) -> list[dict]:
    """
    Sample hard negatives with guaranteed Indian representation.

    Target allocation (mirrors real-world fairness goals):
      40 % metro Indian
      40 % Tier-2 Indian
      20 % international

    Falls back gracefully when a tier has fewer rows than the quota.
    """
    rng = np.random.default_rng(seed=seed)

    n_metro = max(1, int(n * 0.40))
    n_tier2 = max(1, int(n * 0.40))
    n_intl  = max(1, n - n_metro - n_tier2)

    def _sample(tier: str, k: int) -> pd.DataFrame:
        pool = df_suppliers[df_suppliers["_tier"] == tier]
        k = min(k, len(pool))
        return pool.sample(n=k, random_state=int(rng.integers(0, 2**31))) if k > 0 else pd.DataFrame()

    parts = pd.concat([
        _sample("metro",         n_metro),
        _sample("tier2",         n_tier2),
        _sample("international", n_intl),
    ], ignore_index=True)

    # Top-up if any tier was under-represented
    shortfall = n - len(parts)
    if shortfall > 0:
        extra = df_suppliers.sample(
            n=shortfall,
            random_state=int(rng.integers(0, 2**31)),
        )
        parts = pd.concat([parts, extra], ignore_index=True)

    return parts.to_dict("records")


def _location_sampling_weights(df_suppliers: pd.DataFrame, loc_col: str) -> np.ndarray:
    """
    Weight target_loc draws by real supplier density per city instead of
    uniform 1/30. Uniform sampling picks an essentially-unmatchable city
    ~97% of the time (since suppliers cluster in a handful of metros),
    which crashes joint feasibility (price∧location∧cert) to <1% and
    starves CD-LambdaRank of feasible-vs-infeasible pairs to learn from.

    "" (no location constraint) gets a fixed floor probability — most real
    procurement queries don't pin an exact city.
    """
    locs = df_suppliers[loc_col].fillna("").str.lower()
    NO_CONSTRAINT_PROB = 0.35
    cities = CONSTRAINT_LOCATIONS[1:]  # exclude ""
    counts = np.array([locs.str.contains(c, regex=False).sum() for c in cities], dtype=float)
    counts = np.where(counts.sum() > 0, counts, 1.0)  # avoid all-zero edge case
    city_weights = (1 - NO_CONSTRAINT_PROB) * counts / counts.sum()
    return np.concatenate([[NO_CONSTRAINT_PROB], city_weights])


def compute_relevance(
    sbert_score,
    price_match,
    location_match,
    cert_match,
    years_norm,
):

    hidden_factor = np.random.normal(0, 0.30)

    score = (
        0.25 * sbert_score +
        0.20 * price_match +
        0.20 * location_match +
        0.20 * cert_match +
        0.10 * years_norm +
        0.05 * hidden_factor
    )

    if np.random.rand() < 0.15:
        score += np.random.normal(0, 0.15)

    score = float(np.clip(score, 0, 1))

    if score >= 0.75:
        return 3
    elif score >= 0.55:
        return 2
    elif score >= 0.35:
        return 1
    else:
        return 0


def build_training_data(
    top_k: int = 50,
    queries_per_constraint: int = 5,
    gamma: float = 0.3,          # kept for CLI compatibility, not used
    hard_negative_ratio: float = 1.0,  # kept for CLI compatibility, not used
) -> pd.DataFrame:
    """
    Build training dataset using SBERT as PRIMARY retrieval.

    Hard negatives are drawn via STRATIFIED sampling (40 % metro /
    40 % Tier-2 / 20 % international) instead of uniform random sampling,
    so Tier-2 Indian suppliers are meaningfully represented in training.

    Constraint location pool is expanded to include all Tier-2 cities,
    ensuring the model sees Tier-2 location_match signals during training.

    Args:
        top_k: Number of candidates to retrieve per query (default: 50)
        queries_per_constraint: Number of constraint variations per query (default: 5)
        gamma: Not used (kept for compatibility)
        hard_negative_ratio: Not used (kept for compatibility)
    """
    print("=" * 65)
    print("🏗️ Feature Builder (SBERT-Primary Retrieval)")
    print("=" * 65)
    print(f"   top_k:                   {top_k}")
    print(f"   queries_per_constraint:  {queries_per_constraint}")
    print(f"   hard-negative strategy:  stratified (40% metro / 40% tier2 / 20% intl)")
    print(f"   constraint locations:    {len(CONSTRAINT_LOCATIONS)} cities (metro + tier2)")

    df_suppliers = load_suppliers()
    queries = load_queries()

    loc_col = "location" if "location" in df_suppliers.columns else "supplier_location"
    location_weights = _location_sampling_weights(df_suppliers, loc_col)

    all_rows = []
    query_id = 0

    for base_query in queries:  # Use all queries — minimum 200 needed for IEEE-level evaluation
        print(f"\n  Query: {base_query}")

        # Add query noise to prevent overfitting on clean queries
        noise_words = ["cheap", "bulk", "supplier", "export", "manufacturer"]
        rng_noise = np.random.default_rng(seed=hash(base_query) % 2**31)
        noisy_query = base_query + " " + rng_noise.choice(noise_words)

        candidates = retrieve(noisy_query, k=top_k)
        # Inject retrieval noise
        rng = np.random.default_rng()

        for c in candidates :
            if "faiss_score" in c :
                c["faiss_score"] += rng.normal( 0, 0.10 )

        # Stratified hard negatives — guaranteed Indian city representation
        hard_negatives = _stratified_hard_negatives(
            df_suppliers,
            n=top_k // 2,
            seed=hash(base_query) % 2**31,
        )
        candidates = candidates + hard_negatives

        # Shuffle to remove position bias
        np.random.shuffle(candidates)

        if not candidates:
            print(f"    ⚠️ No candidates retrieved, skipping")
            continue

        print(f"    ✅ Retrieved {len(candidates)} candidates via SBERT + stratified negatives")

        for variation in range(queries_per_constraint):
            query_id += 1
            rng = np.random.default_rng(seed=hash(f"{base_query}_{variation}") % 2**31)

            constraint_type = rng.choice(
                [
                    "price_only",
                    "location_only",
                    "cert_only",
                    "price_location",
                    "price_cert",
                    "location_cert",
                    "all_constraints",
                    "none"
                ],
                p=[
                    0.20,  # price only
                    0.15,  # location only
                    0.10,  # cert only
                    0.15,  # price + location
                    0.10,  # price + cert
                    0.10,  # location + cert
                    0.05,  # all constraints
                    0.15  # unconstrained query
                ]
            )

            max_price = None
            target_loc = ""
            req_cert = ""

            if constraint_type in (
                    "price_only",
                    "price_location",
                    "price_cert",
                    "all_constraints"
            ) :
                max_price = rng.choice(
                    [50, 100, 200, 500, 1000],
                    p=[0.10, 0.20, 0.30, 0.30, 0.10]
                )

            if constraint_type in (
                    "location_only",
                    "price_location",
                    "location_cert",
                    "all_constraints"
            ) :
                target_loc = rng.choice(
                    CONSTRAINT_LOCATIONS,
                    p=location_weights
                )

            if constraint_type in (
                    "cert_only",
                    "price_cert",
                    "location_cert",
                    "all_constraints"
            ) :
                req_cert = rng.choice(
                    ["ISO", "FDA", "CE", "GMP"]
                )

            for rank, supplier in enumerate(candidates):
                # Hard negatives won't have faiss_score — assign low random score
                if "faiss_score" in supplier:
                    sbert_score = supplier["faiss_score"]
                elif "semantic_score" in supplier:
                    sbert_score = supplier["semantic_score"]
                else:
                    sbert_score = np.random.uniform(0.1, 0.4)

                price = parse_price(get_field(supplier, "price_min") or supplier.get("price", 0))

                # Price features
                if max_price and max_price > 0 and price > 0:
                    price_match = 1.0 if price <= max_price else np.random.uniform(0.0, 0.3)
                    price_ratio = min(price / max_price, 2.0)
                    price_distance = abs(price - max_price) / max_price
                else:
                    price_match, price_ratio, price_distance = 0.5, 1.0, 0.0

                # Location — read from both possible key names
                s_loc = str(get_field(supplier, "supplier_location") or get_field(supplier, "location", default="")).lower()
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

                # Years — cold-start fix: a brand-new supplier with no tenure
                # data should be treated as *neutral* (0.5), not penalized
                # (0.0). Otherwise every new supplier gets buried regardless
                # of how well they actually match the query.
                years = get_field(supplier, "years_with_gs", "years_on_platform", default=None)
                try:
                    if years is None or str(years).strip() in ("", "0", "nan", "none"):
                        years_norm = 0.5  # missing tenure → neutral, not 0
                    else:
                        years_norm = min(float(years) / 10.0, 1.0)
                        if years_norm < 0.1:
                            years_norm = max(years_norm, 0.3)  # cold-start floor
                except (ValueError, TypeError):
                    years_norm = 0.5

                biz = str(get_field(supplier, "business_type", default="")).lower()
                is_manuf = 1.0 if "manufacturer" in biz else 0.0
                is_trading = 1.0 if "trading" in biz else 0.0

                relevance = compute_relevance(
                    sbert_score, price_match, location_match, cert_match, years_norm
                )

                # Carry real city name through so fairness.py gets actual location data.
                raw_location = get_field(supplier, "supplier_location") or get_field(supplier, "location", default="")

                constraint_score = (
                                           price_match +
                                           location_match +
                                           cert_match
                                   ) / 3.0

                all_rows.append( {
                    "query_id" : query_id,
                    "query_text" : base_query,
                    "supplier_idx" : rank,
                    "supplier_name" : supplier.get( "supplier_name" ) or supplier.get( "supplier name", "" ),
                    "location" : str( raw_location ).strip(),

                    "price_match" : price_match,
                    "price_ratio" : price_ratio,
                    "price_distance" : price_distance,

                    "location_match" : location_match,
                    "cert_match" : cert_match,

                    "constraint_score" : constraint_score,

                    "years_normalized" : years_norm,

                    "is_manufacturer" : is_manuf,
                    "is_trading_company" : is_trading,

                    "faiss_score" : sbert_score,
                    "faiss_rank" : rank + 1,

                    "relevance" : relevance,
                } )

        print(f"    📊 Generated {len(candidates) * queries_per_constraint} training pairs")

    if not all_rows:
        raise RuntimeError("No training rows generated")

    df_out = pd.DataFrame(all_rows)
    df_out["relevance"] = df_out["relevance"].clip(0, 3).astype(int)

    # Tier column for downstream fairness analysis
    df_out["tier"] = df_out["location"].apply(_classify_tier)

    df_out.to_csv(OUTPUT_FILE, index=False)
    joint_feasible = (
            (df_out["price_match"] >= 0.5) &
            (df_out["location_match"] >= 0.5) &
            (df_out["cert_match"] >= 0.5)
    )

    print(
        f"   Joint feasibility: "
        f"{joint_feasible.sum():,} / {len( df_out ):,} "
        f"({100 * joint_feasible.mean():.1f}%)"
    )

    print(f"\n✅ Training dataset saved: {OUTPUT_FILE}")
    print(f"   Total pairs:      {len(df_out):,}")
    print(f"   Unique queries:   {df_out['query_id'].nunique():,}")
    print(f"   Label distribution:\n{df_out['relevance'].value_counts().sort_index()}")
    print(f"   Tier distribution:\n{df_out['tier'].value_counts()}")

    return df_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--queries-per-constraint", type=int, default=5)
    args = parser.parse_args()

    build_training_data(top_k=args.top_k, queries_per_constraint=args.queries_per_constraint)
    print("\n🎯 Feature building complete!")