"""
Feature Builder — SourceUp Supplier Ranking (SBERT-Primary)
------------------------------------------------------------
Builds training data ONLY from SBERT-retrieved candidates.

Relevance labels are produced via weak supervision: see
backend/app/training/weak_label_generator.py for the labeling
methodology, and configs/weak_labels.yaml for the heuristic weights.
"""

import os
import sys
import re
import warnings
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


def _stable_hash(s: str) -> int:
    """
    Deterministic string -> int hash, safe for use as an RNG seed.

    WHY THIS EXISTS: this module previously used Python's built-in
    hash(str) to derive per-query RNG seeds (e.g. hash(base_query) % 2**31).
    Python randomizes str hashing per-process by default (PYTHONHASHSEED
    is a random salt set at interpreter startup) specifically to prevent
    hash-flooding attacks — it is NOT reproducible across runs, which
    silently defeated the entire point of "seeding" these generators.
    This was the root cause of run-to-run differences in retrieved
    candidates, sampled hard negatives, and injected FAISS noise (e.g.
    "vacuum seal food packaging bulk" vs "...export" between identical
    pipeline runs), which in turn produced misleading swings in CVR,
    NDCG, and stability metrics that had nothing to do with the actual
    label-generation or model changes being tested.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (2**31)

def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))
from config import cfg

from backend.app.utils.fields import get_field
from backend.app.models.retriever import retrieve
from backend.app.training.weak_label_generator import (
    compute_weak_label,
    load_weak_label_config,
    save_weak_label_metadata,
    rebucket_by_quantile,
)

# Weak-label weights/thresholds/noise are now sourced from
# configs/weak_labels.yaml instead of being hardcoded here.
_WEAK_LABEL_CONFIG = load_weak_label_config()

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


def build_training_data(
    top_k: int = 50,
    queries_per_constraint: int = 5,
    gamma: float = 0.3,          # kept for CLI compatibility, not used
    hard_negative_ratio: float = 1.0,  # kept for CLI compatibility, not used
    seed: int = 42,
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

    # Seed the global numpy RNG so the few remaining fallback paths that
    # use np.random.* directly (rather than a per-query _stable_hash seed)
    # are also reproducible run-to-run. Combined with the _stable_hash()
    # fix above (Python's built-in hash() is randomized per-process and
    # was silently making every "seeded" RNG in this file non-reproducible),
    # this makes the whole training-data build deterministic for a fixed
    # `seed`, which matters for the paper's reported metrics being stable
    # across reruns rather than swinging with sampling noise.
    np.random.seed(seed)

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
        rng_noise = np.random.default_rng(seed=_stable_hash(base_query))
        noisy_query = base_query + " " + rng_noise.choice(noise_words)

        candidates = retrieve(noisy_query, k=top_k)
        # Inject retrieval noise — deterministically seeded so repeated runs
        # of the pipeline are reproducible (previously np.random.default_rng()
        # with no seed, so this noise differed every run).
        rng = np.random.default_rng(seed=_stable_hash(base_query + "_faissnoise"))

        for c in candidates :
            if "faiss_score" in c :
                c["faiss_score"] += rng.normal( 0, 0.10 )

        # Stratified hard negatives — guaranteed Indian city representation
        hard_negatives = _stratified_hard_negatives(
            df_suppliers,
            n=top_k // 2,
            seed=_stable_hash(base_query),
        )
        candidates = candidates + hard_negatives

        # Shuffle to remove position bias — deterministically seeded (was
        # the bare global np.random.shuffle(), also non-reproducible).
        rng_shuffle = np.random.default_rng(seed=_stable_hash(base_query + "_shuffle"))
        rng_shuffle.shuffle(candidates)

        if not candidates:
            print(f"    ⚠️ No candidates retrieved, skipping")
            continue

        print(f"    ✅ Retrieved {len(candidates)} candidates via SBERT + stratified negatives")

        for variation in range(queries_per_constraint):
            query_id += 1
            rng = np.random.default_rng(seed=_stable_hash(f"{base_query}_{variation}"))

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

                # Raw (pre-normalization) FAISS cosine similarity — comparable
                # across queries, unlike faiss_score which is normalized
                # relative to this query's candidate pool only. Hard negatives
                # don't come from retrieve(), so default to a low fixed value
                # rather than 0, which would look like a perfect-opposite match.
                faiss_raw_score = float(supplier.get("faiss_raw_score", 0.1))

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
                s_cert_raw = str(supplier.get("certifications", ""))
                s_cert = s_cert_raw.lower()
                if req_cert:
                    cert_match = 1.0 if req_cert.lower() in s_cert else 0.0
                else:
                    cert_match = 0.5

                # Certification count — number of distinct certifications listed
                # (comma/semicolon/slash separated). A supplier with more
                # certifications is generally lower-risk/more credible,
                # independent of whether THIS query's specific cert matched.
                if s_cert_raw and s_cert.strip() not in ("", "nan", "none"):
                    cert_count = len([
                        c for c in re.split(r"[,;/]", s_cert_raw) if c.strip()
                    ])
                else:
                    cert_count = 0
                cert_count_norm = min(cert_count / 5.0, 1.0)  # cap at 5+ certs

                # Supplier rating — from CANONICAL_COLUMNS "rating" field.
                # Cold-start: missing rating treated as neutral (0.5 on a
                # 0-1 scale), same convention as years_normalized below,
                # rather than penalizing suppliers with no review history.
                rating_raw = get_field(supplier, "rating", default=None)
                try:
                    if rating_raw is None or str(rating_raw).strip().lower() in ("", "nan", "none", "0"):
                        rating_norm = 0.5
                    else:
                        # Ratings are typically on a 0-5 scale in the scraped data
                        rating_norm = min(max(float(rating_raw) / 5.0, 0.0), 1.0)
                except (ValueError, TypeError):
                    rating_norm = 0.5

                # Category overlap — token overlap between the query text and
                # the supplier's category/product_name fields. Cheap proxy for
                # "is this supplier even in the right product category" that
                # doesn't require the category_l1..l4 hierarchy (which isn't
                # present in CANONICAL_COLUMNS / suppliers_clean.csv).
                query_tokens = set(base_query.lower().split())
                supplier_text = " ".join([
                    str(get_field(supplier, "category", default="")),
                    str(get_field(supplier, "product_name", default="")),
                ]).lower()
                supplier_tokens = set(supplier_text.split())
                if query_tokens and supplier_tokens:
                    category_overlap_score = len(query_tokens & supplier_tokens) / len(query_tokens)
                else:
                    category_overlap_score = 0.0

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

                weak_label_score, weak_label = compute_weak_label(
                    sbert_score, price_match, location_match, cert_match, years_norm,
                    config=_WEAK_LABEL_CONFIG,
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
                    "product_name" : get_field( supplier, "product_name", default="" ),
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
                    "faiss_raw_score" : faiss_raw_score,

                    # New candidate-quality features (not yet in FEATURE_COLS —
                    # add to train_lambdarank.py's FEATURE_COLS deliberately,
                    # after checking SHAP importance, rather than assuming
                    # they help).
                    "supplier_rating" : rating_norm,
                    "certification_count" : cert_count_norm,
                    "category_overlap_score" : category_overlap_score,

                    # Continuous weak-label score (pre-bucketing) — kept for
                    # debugging/reproducibility per the weak-supervision
                    # methodology (see backend/app/training/weak_label_generator.py).
                    "weak_label_score" : weak_label_score,

                    # NOTE: column kept as "relevance" for backward
                    # compatibility with downstream training/eval code
                    # (train_lambdarank.py, eval/*.py, etc.), but the value
                    # is now produced by the weak-supervision label
                    # generator rather than human annotation.
                    "relevance" : weak_label,
                } )

        print(f"    📊 Generated {len(candidates) * queries_per_constraint} training pairs")

    if not all_rows:
        raise RuntimeError("No training rows generated")

    df_out = pd.DataFrame(all_rows)
    df_out["relevance"] = df_out["relevance"].clip(0, 5).astype(int)

    # --- Quantile rebucketing (v3.0 label-skew fix, v3.1 feasibility gate)
    # compute_weak_label() bucketed each row above against fixed absolute
    # thresholds from weak_labels.yaml, which assumed weak_label_score is
    # roughly uniform over [0,1]. It isn't (see weak_label_generator.py's
    # rebucket_by_quantile docstring for the full diagnosis) — most rows
    # land well below the absolute label_4/label_5 cutoffs regardless of
    # how relevant they are *relative to the rest of this dataset*.
    # Recompute the final "relevance" column from the empirical quantiles
    # of weak_label_score instead, so labels 4-5 actually get populated.
    # The original absolute-threshold "relevance" above is kept as
    # "relevance_absolute_threshold" for comparison/debugging — it is NOT
    # used for training.
    #
    # v3.1: also pass feasibility_count (how many of price_match/
    # location_match/cert_match are >= 0.5) so rebucket_by_quantile can
    # cap label tier by constraint satisfaction — fixes the CVR increase
    # (~0.37 -> ~0.45-0.47, confirmed reproducible) that v3.0's quantile
    # fix introduced by letting semantically-strong/constraint-violating
    # rows into labels 4-5.
    df_out["relevance_absolute_threshold"] = df_out["relevance"]
    feasibility_count = (
        (df_out["price_match"] >= 0.5).astype(int) +
        (df_out["location_match"] >= 0.5).astype(int) +
        (df_out["cert_match"] >= 0.5).astype(int)
    )
    df_out["relevance"], _resolved_cutoffs = rebucket_by_quantile(
        df_out["weak_label_score"],
        config=_WEAK_LABEL_CONFIG,
        feasibility_count=feasibility_count,
    )

    print("   Relevance label distribution (quantile-rebucketed + feasibility-gated):")
    print(df_out["relevance"].value_counts().sort_index().to_string())
    print(f"   Resolved score cutoffs for this run: {_resolved_cutoffs}")
    print(f"   Feasibility count distribution (rows satisfying N of 3 constraints):")
    print(feasibility_count.value_counts().sort_index().to_string())

    # Tier column for downstream fairness analysis
    df_out["tier"] = df_out["location"].apply(_classify_tier)

    df_out.to_csv(OUTPUT_FILE, index=False)

    metadata_path = save_weak_label_metadata(resolved_cutoffs=_resolved_cutoffs)
    print(f"   Weak-label metadata saved: {metadata_path}")

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