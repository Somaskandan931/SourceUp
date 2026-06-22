"""
SourceUp — Full Pipeline Orchestrator (UPDATED with SBERT-Primary + XGBRanker)
--------------------------------------
Runs the complete data pipeline from raw scraping to model training
and evaluation. NOW with SBERT as PRIMARY retrieval and XGBRanker.

Usage:
    python pipeline/run_all.py --full                   # Run everything
    python pipeline/run_all.py --train-xgbranker       # Train XGBRanker (primary)
    python pipeline/run_all.py --train-lambdarank      # Train LightGBM (backup)
    python pipeline/run_all.py --run-analysis          # Run evaluation analysis
    python pipeline/run_all.py --shap-analysis         # Run SHAP analysis
    python pipeline/run_all.py --limit 1000            # Limit dataset size

Note: location assignment (step 4) always runs automatically — it used
to be behind --assign-locations, but clean_normalize.py (step 2) wipes
the location-tier column on every run, so making it optional meant any
run without that flag silently lost the fairness fix.
"""

import sys
import os
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

from pathlib import Path


def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))
from config import cfg
from rule_baseline import score_rule_based as _canonical_rule_scorer


FEATURE_COLS = [
    "price_match", "price_ratio",
    "location_match", "cert_match",
    "faiss_score",
    # FIX: this list previously restored years_normalized/is_manufacturer,
    # on the theory that rule_based_score() below used both and gave the
    # baseline an unfair edge. That was only half right — rule_based_score()
    # only ever read years_normalized, never is_manufacturer — and it was
    # never re-verified with SHAP after the restore. Meanwhile 8 other files
    # in this codebase (ranker.py, shap_analysis.py, ablation.py,
    # baselines.py, fairness.py, sensitivity.py, stability.py,
    # label_noise_analysis.py, train_lambdarank.py's FEATURE_COLS_6) all
    # independently confirmed zero SHAP importance for both columns and use
    # this same 6-feature set as the production schema. Restoring two extra
    # columns here just to chase the baseline — without re-running SHAP to
    # check whether they're actually informative on this data — risked
    # reintroducing the exact price_distance-style duplicate/instability
    # problem the other NOTE below describes. Reverted to the 6-feature
    # production set for consistency with the rest of the codebase; the
    # years_normalized asymmetry vs. rule_based_score() is fixed at the
    # source instead (see rule_based_score() below), so the comparison is
    # apples-to-apples without inflating the model's own feature set.
    # NOTE: price_distance removed — for price/max_price <= 2 (the vast
    # majority of rows) it equals abs(price_ratio - 1) exactly, a pure
    # deterministic transform of price_ratio. Keeping both caused the model
    # to split arbitrarily between two copies of the same signal, which is
    # why SHAP rank order for price features flipped between training runs.
]


# ============================================================================
# REPLACED ndcg_at_k function (NaN-safe)
# ============================================================================
def ndcg_at_k(y_true, y_pred, query_ids, k=10, verbose=False):
    df_ndcg = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "qid": query_ids})
    scores = []
    skipped_single_level = 0
    skipped_constant_pred = 0
    skipped_too_short = 0

    for qid, group in df_ndcg.groupby("qid"):
        t = group["y_true"].values
        p = group["y_pred"].values

        # Fix NaNs / Inf in both arrays
        p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
        t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)

        if len(t) < 2:
            skipped_too_short += 1
            continue

        # Skip queries with only one relevance level (NDCG undefined)
        if len(np.unique(t)) <= 1:
            skipped_single_level += 1
            continue

        # Skip if all predictions identical — no ranking signal
        if np.all(p == p[0]):
            skipped_constant_pred += 1
            continue

        try:
            score = ndcg_score([t], [p], k=k)
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue

    # FIX: this function silently drops queries with a single relevance
    # level or constant predictions, while a model's own internal
    # eval_metric (e.g. XGBoost's "ndcg@10" printed during training)
    # typically does NOT drop them — it can score them as 0 instead. That
    # mismatch is exactly why the per-round training log and the final
    # printed NDCG@10 in this script can disagree by a wide margin even on
    # the same test set. Printing the skip counts makes that explicit
    # instead of leaving a confusing, unexplained gap between the two
    # numbers.
    if verbose:
        total = len(df_ndcg["qid"].unique())
        print(f"      ndcg_at_k: scored {len(scores)}/{total} queries "
              f"(skipped {skipped_single_level} single-relevance-level, "
              f"{skipped_constant_pred} constant-prediction, "
              f"{skipped_too_short} too-short)")

    return np.mean(scores) if scores else 0.0


def safe_div(a, b):
    """Safe division that returns 0 when denominator is 0."""
    return a / b if b != 0 else 0.0


def compute_rule_score(row):
    """NaN-safe rule-based scoring function.

    DEPRECATED / UNUSED: not called anywhere in this file (confirmed via
    grep) — rule_based_score() below is what's actually invoked for the
    pipeline's "Rule-Based Baseline NDCG@10" line. Left in place rather
    than deleted in case something outside this file imports it, but do
    not add new call sites; use rule_baseline.score_rule_based() instead.
    """
    score = 0.0

    if 'price' in row and 'budget' in row:
        # FIX 2: epsilon prevents division-by-zero NaN
        budget_safe = float(row['budget']) + 1e-6
        score += (1 - safe_div(float(row['price']), budget_safe))

    if 'cert_match' in row:
        score += 0.3 * row['cert_match']

    if 'years_on_platform' in row:
        score += 0.1 * row['years_on_platform']

    return score


def rule_based_score(df):
    """NaN-safe rule-based scoring function.

    FIX (this version): this function previously used its OWN weights —
    0.45 price_match / 0.35 faiss_score / 0.20 cert_match, with NO
    location_match term at all — different from every other script in
    this repo that reports a "Rule-Based Baseline." That's why this
    pipeline's own printed "Rule-Based Baseline NDCG@10: 0.8549" didn't
    match ablation.py/baselines.py/sensitivity.py's "0.8720" for what
    looked like the same number. Now delegates to
    rule_baseline.score_rule_based(), the single canonical formula
    (0.40 price_match / 0.30 location_match / 0.25 cert_match / 0.05
    faiss_score) shared by every script that needs a no-ML baseline.
    Re-running this pipeline will now print a different (higher) number
    for the rule-based baseline than before — that's expected and is the
    point of the fix; the old 0.8549 was never comparable to the other
    scripts' 0.8720 to begin with.
    """
    score = _canonical_rule_scorer(df)
    return score


def train_xgbranker_inline(df_train, df_val):
    """Train XGBRanker (primary ranking model).

    NOTE: `df_val` here is an early-stopping validation split carved out of
    the training data — NOT the final held-out test set. Previously this
    function received the final test set directly and used it for
    early stopping, which means the reported "test" NDCG@10 was computed
    on a set the model had already been tuned against (number of boosting
    rounds chosen to do well on it). That's a leak: it can bias the final
    metric in either direction and makes it incomparable to the
    rule-based baseline, which never gets to see the test set at all.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("   ❌ XGBoost not installed. Run: pip install xgboost")
        return None

    print("\n🚀 Training XGBRanker (Primary Ranking Model)")
    print("=" * 50)

    # Build query groups
    train_groups = df_train.groupby('query_id').size().values
    val_groups = df_val.groupby('query_id').size().values

    train_qid = np.concatenate([[i] * size for i, size in enumerate(train_groups)])
    val_qid = np.concatenate([[i] * size for i, size in enumerate(val_groups)])

    # FIX: no early stopping was configured, so the model trained all 300
    # rounds regardless of validation NDCG. The printed per-round NDCG@10
    # oscillated between ~0.785 and ~0.789 without ever clearly converging
    # (e.g. round 61: 0.78935, round 299: 0.78760) — the final model is
    # whichever round happened to land last, not the best one. Adding
    # early stopping keeps the best-scoring iteration instead.
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.5,
        min_child_weight=5,
        tree_method="hist",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=1
    )

    print(f"   Train: {len(df_train):,} rows, {len(train_groups)} queries")
    print(f"   Val:   {len(df_val):,} rows, {len(val_groups)} queries (early stopping only)")

    model.fit(
        df_train[FEATURE_COLS].values.astype(np.float32),
        df_train['relevance'].values,
        qid=train_qid,
        eval_set=[(df_val[FEATURE_COLS].values.astype(np.float32), df_val['relevance'].values)],
        eval_qid=[val_qid],
        verbose=True
    )

    return model


def train_lambdarank_via_module(rounds: int = 500) -> bool:
    """Train LightGBM LambdaRank by invoking train_lambdarank.py as a
    subprocess, instead of maintaining a second, divergent inline trainer.

    FIX: run_all.py previously had its own train_lambdarank_inline()
    function — different hyperparameters, different sample-weighting
    logic, no train/test leakage protections beyond what's duplicated
    here, and (until the FEATURE_COLS fix above) a different feature
    schema than every eval script. train_lambdarank.py is the
    paper-authoritative trainer: FIX-1 through FIX-11 in its docstring
    document a real data-leakage fix, a vectorised CD-LambdaRank
    objective, a feasibility sanity check, and the documented finding
    (FIX-8/FIX-9) that Standard LambdaRank — not CD-LambdaRank — is the
    production model. Calling it here as a subprocess (rather than
    importing main()) avoids any argparse/global-state collision between
    the two CLIs while guaranteeing run_all.py and the standalone script
    can never train two different models under the same filename again.

    Writes ranker_lightgbm.pkl (production, Standard LambdaRank) and
    ranker_lightgbm_standard.pkl (same model, explicit name used by
    ablation.py's V1b arm) directly via train_lambdarank.py's own save
    logic — this function does not touch cfg.LGBM_MODEL itself.
    """
    import subprocess

    # FIX: run_all.py lives at pipeline/run_all.py while train_lambdarank.py
    # lives at backend/app/models/train_lambdarank.py — NOT the same folder.
    # Path(__file__).parent pointed at pipeline/, which doesn't contain
    # train_lambdarank.py at all (confirmed by a real run: "can't open
    # file ...\pipeline\train_lambdarank.py: No such file or directory").
    # Resolve relative to the project root (already located via
    # _find_project_root() at import time) so this works regardless of
    # which directory run_all.py itself is invoked from.
    script_path = _find_project_root() / "backend" / "app" / "models" / "train_lambdarank.py"
    if not script_path.exists():
        print(f"   ❌ train_lambdarank.py not found at: {script_path}")
        print(f"      If it has moved, update this path in train_lambdarank_via_module().")
        return False
    print("\n🌳 Training LightGBM — Standard LambdaRank (Production Model)")
    print("=" * 50)
    print(f"   Delegating to: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path), "--rounds", str(rounds)],
        cwd=str(script_path.parent),
    )

    if result.returncode != 0:
        print(f"   ❌ train_lambdarank.py exited with code {result.returncode}")
        return False

    if not cfg.LGBM_MODEL.exists():
        print(f"   ❌ Expected output not found: {cfg.LGBM_MODEL}")
        return False

    print(f"   ✅ LightGBM training complete: {cfg.LGBM_MODEL}")
    if cfg.LGBM_MODEL_STANDARD.exists():
        print(f"   ✅ Also available as: {cfg.LGBM_MODEL_STANDARD}")
    return True


def assign_locations_to_suppliers():
    """Intelligently assign locations to suppliers without location data."""
    print("\n📍 Assigning Smart Locations to Suppliers")
    print("=" * 50)

    METRO_CITIES = [
        "Mumbai", "Delhi", "New Delhi", "Chennai", "Bangalore",
        "Bengaluru", "Hyderabad", "Kolkata", "Pune", "Ahmedabad"
    ]

    TIER2_CITIES = [
        "Noida", "Gurgaon", "Chandigarh", "Jaipur", "Lucknow",
        "Kanpur", "Nagpur", "Indore", "Bhopal", "Surat",
        "Vadodara", "Coimbatore", "Kochi", "Visakhapatnam", "Patna"
    ]

    TIER3_CITIES = [
        "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot",
        "Varanasi", "Amritsar", "Allahabad", "Ranchi", "Jodhpur",
        "Guwahati", "Mysore", "Tiruchirappalli", "Salem", "Dehradun"
    ]

    if not cfg.CLEAN_DATA.exists():
        print("❌ Clean data not found. Run pipeline without --skip-features first.")
        return False

    df = pd.read_csv(str(cfg.CLEAN_DATA))
    print(f"📊 Loaded {len(df):,} suppliers")

    rng = np.random.default_rng(42)

    def get_product_category(product_name):
        product = str(product_name).lower()
        if any(kw in product for kw in ['electronic', 'computer', 'phone', 'mobile', 'laptop', 'tech', 'led', 'usb']):
            return 'electronics'
        if any(kw in product for kw in ['fabric', 'cloth', 'garment', 'textile', 'apparel', 'shirt', 'jeans']):
            return 'textile'
        if any(kw in product for kw in ['auto', 'car', 'vehicle', 'truck', 'bike', 'motorcycle', 'filter', 'brake']):
            return 'automotive'
        if any(kw in product for kw in ['chemical', 'pharma', 'drug', 'medicine', 'oil', 'solvent']):
            return 'chemical'
        if any(kw in product for kw in ['machine', 'equipment', 'industrial', 'tool', 'motor']):
            return 'industrial'
        if any(kw in product for kw in ['packaging', 'box', 'carton', 'tape', 'bag', 'pouch']):
            return 'packaging'
        return 'general'

    def assign_city(row):
        category = get_product_category(row.get('product name', ''))
        category_cities = {
            'electronics': ['Bangalore', 'Hyderabad', 'Pune', 'Chennai'],
            'textile': ['Surat', 'Lucknow', 'Jaipur', 'Delhi'],
            'automotive': ['Chennai', 'Pune', 'Delhi NCR', 'Mumbai'],
            'chemical': ['Ahmedabad', 'Hyderabad', 'Mumbai', 'Vadodara'],
            'industrial': ['Mumbai', 'Chennai', 'Pune', 'Ahmedabad'],
            'packaging': ['Delhi NCR', 'Mumbai', 'Ahmedabad', 'Chennai'],
            'general': METRO_CITIES + TIER2_CITIES
        }
        cities = category_cities.get(category, METRO_CITIES + TIER2_CITIES)

        try:
            price_str = str(row.get('price', ''))
            import re
            price_nums = re.findall(r'\d+', price_str)
            if price_nums and float(price_nums[0]) > 100:
                metro_options = [c for c in cities if c in METRO_CITIES]
                if metro_options:
                    return rng.choice(metro_options)
        except:
            pass

        # FIX 3: Weighted random — 40% Metro, 40% Tier-2, 20% Tier-3 for diversity
        rand = rng.random()
        if rand < 0.4:
            opts = [c for c in cities if c in METRO_CITIES] or METRO_CITIES
        elif rand < 0.8:
            opts = [c for c in cities if c in TIER2_CITIES] or TIER2_CITIES
        else:
            opts = [c for c in cities if c in TIER3_CITIES] or TIER3_CITIES
        return rng.choice(opts)

    locations = []
    for _, row in df.iterrows():
        loc = assign_city(row)
        locations.append(loc)

    df['location'] = locations

    def get_tier(city):
        if any(m in city for m in METRO_CITIES):
            return 'Metro'
        elif any(t in city for t in TIER2_CITIES):
            return 'Tier-2'
        return 'Tier-3'

    df['city_tier'] = df['location'].apply(get_tier)

    print(f"\n📊 City tier distribution:")
    tier_counts = df['city_tier'].value_counts()
    for tier in ['Metro', 'Tier-2', 'Tier-3']:
        count = tier_counts.get(tier, 0)
        pct = count / len(df) * 100
        print(f"   {tier}: {count:,} ({pct:.1f}%)")

    df.to_csv(str(cfg.CLEAN_DATA), index=False)
    print(f"\n✅ Locations assigned and saved to {cfg.CLEAN_DATA}")
    return True


def add_fairness_weights_to_training():
    """Add fairness weights to training data."""
    print("\n⚖️ Adding Fairness Weights to Training Data")
    print("=" * 50)

    if not cfg.TRAINING_DATA.exists():
        print("❌ Training data not found. Run feature builder first.")
        return False

    df = pd.read_csv(str(cfg.TRAINING_DATA))
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Check if training data has supplier_idx to merge with clean data
    if 'supplier_idx' not in df.columns:
        print("⚠️ Training data missing 'supplier_idx' column - cannot add fairness weights")
        print("   Fairness weights will be skipped")
        return False

    clean_df = pd.read_csv(str(cfg.CLEAN_DATA))
    clean_df.columns = [c.strip().lower().replace(' ', '_') for c in clean_df.columns]

    if 'location' in clean_df.columns and 'city_tier' in clean_df.columns:
        # Create location mapping by index
        clean_df['supplier_idx'] = clean_df.index
        location_map = clean_df[['supplier_idx', 'location', 'city_tier']].drop_duplicates()

        # Merge
        df = df.merge(location_map, on='supplier_idx', how='left')

        # Fill missing city_tier with 'Other'
        if 'city_tier' in df.columns:
            df['city_tier'] = df['city_tier'].fillna('Other')
        else:
            print("⚠️ city_tier column not found after merge - skipping fairness weights")
            return False

        tier_counts = df['city_tier'].value_counts()
        print(f"\n📊 Training city tier distribution:")
        for tier in ['Metro', 'Tier-2', 'Tier-3', 'Other']:
            count = tier_counts.get(tier, 0)
            pct = count / len(df) * 100 if len(df) > 0 else 0
            print(f"   {tier}: {count:,} ({pct:.1f}%)")

        # ====================================================================
        # REPLACED fairness weight logic (correct direction)
        # ====================================================================
        max_count = tier_counts.max()

        def get_weight(tier):
            return max_count / tier_counts[tier]

        df['fairness_weight'] = df['city_tier'].apply(get_weight)

        print(f"\n   Fairness weights applied:")
        for tier in df['city_tier'].unique():
            if tier in tier_counts:
                print(f"      {tier}: {get_weight(tier):.2f}")

        output_path = str(cfg.TRAINING_DATA)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Fairness weights added to {output_path}")

        import pickle
        weights_path = cfg.MODELS_DIR / 'fairness_weights.pkl'
        os.makedirs(str(cfg.MODELS_DIR), exist_ok=True)
        with open(str(weights_path), 'wb') as f:
            pickle.dump({
                'has_fairness_weights': True,
                'tier_counts': tier_counts.to_dict(),
                'weights_applied': True
            }, f)
        print(f"✅ Fairness weights saved to {weights_path}")

        return True
    else:
        print("⚠️ Could not merge location data - missing columns")
        print(f"   Clean data columns: {clean_df.columns.tolist()[:10]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="SourceUp Full Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/run_all.py --full                     # Run everything
  python pipeline/run_all.py --train-xgbranker          # Only train XGBRanker
  python pipeline/run_all.py --train-lambdarank         # Only train LightGBM
  python pipeline/run_all.py --run-analysis             # Only run analysis
        """
    )

    parser.add_argument("--skip-features", action="store_true",
                        help="Skip LTR training data build")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit dataset size for faster runs")
    parser.add_argument("--assign-locations", action="store_true",
                        help="[DEPRECATED, now always runs] Assign smart locations to suppliers")
    parser.add_argument("--train-xgbranker", action="store_true",
                        help="Train XGBRanker (primary ranking model)")
    parser.add_argument("--train-lambdarank", action="store_true",
                        help="Train LightGBM LambdaRank (backup model)")
    parser.add_argument("--run-analysis", action="store_true",
                        help="Run evaluation analysis suite")
    parser.add_argument("--shap-analysis", action="store_true",
                        help="Run SHAP explainability analysis")
    parser.add_argument("--full", action="store_true",
                        help="Run complete pipeline (data + training + analysis + SHAP)")
    parser.add_argument("--neg-ratio", type=float, default=1.0,
                        help="Hard negative to positive ratio (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.3,
                        help="Constraint penalty weight (default: 0.3)")

    args = parser.parse_args()

    # --full enables all steps
    if args.full:
        args.assign_locations = True
        args.train_xgbranker = True
        args.train_lambdarank = True
        args.run_analysis = True
        args.shap_analysis = True

    cfg.ensure_dirs()

    print("=" * 65)
    print("🚀 SourceUp — Full Pipeline (SBERT-Primary + XGBRanker)")
    print(f"   Root: {cfg.ROOT}")
    if args.limit:
        print(f"   LIMIT         : {args.limit:,}")
    print(f"   Neg Ratio     : {args.neg_ratio}")
    print(f"   Gamma         : {args.gamma}")
    print(f"   Assign Loc    : Yes (always runs — see step 4)")
    print(f"   XGBRanker     : {'Yes' if args.train_xgbranker else 'No'}")
    print(f"   LightGBM      : {'Yes' if args.train_lambdarank else 'No'}")
    print(f"   Analysis      : {'Yes' if args.run_analysis else 'No'}")
    print(f"   SHAP          : {'Yes' if args.shap_analysis else 'No'}")
    print("=" * 65)

    # ========================================================================
    # Step 1: Validate & Merge
    # ========================================================================
    print("\n[1/7] Validate & Merge")
    try:
        from pipeline.validate_merge import validate_and_merge
        validate_and_merge(limit=args.limit)
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        return

    # ========================================================================
    # Step 2: Clean & Normalise
    # ========================================================================
    print("\n[2/7] Clean & Normalise")
    try:
        from pipeline.clean_normalize import clean
        clean()
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        return

    # ========================================================================
    # Step 3: FAISS Embedding
    # ========================================================================
    print("\n[3/7] FAISS Embedding")
    if os.path.exists(str(cfg.FAISS_INDEX)) and not args.full:
        print(f"   ⏭️  FAISS index already exists — skipping re-embedding")
        print(f"   📦 Index: {cfg.FAISS_INDEX}")
        print(f"   💡 To force rebuild: python pipeline/run_all.py --full")
    else:
        try:
            from pipeline.incremental_faiss import incremental_update
            incremental_update()
        except Exception as e:
            print(f"❌ Step 3 failed: {e}")
            return

    # ========================================================================
    # Step 4: Assign Locations (Fixes Fairness Issue)
    # ========================================================================
    # NOTE: This step now ALWAYS runs (previously gated behind
    # --assign-locations). Step 2 (Clean & Normalise) regenerates
    # suppliers_clean.csv from the raw merge on every run, which has no
    # location-tier column. Without re-running this step every time, all
    # suppliers silently fall back to "international" tier — that's the
    # bug that caused tier counts to collapse to (metro=488, tier2=149,
    # international=870,825) in a real run. Assigning locations is cheap
    # and idempotent, so there is no reason to make it optional.
    print("\n[4/7] Assign Smart Locations")
    assign_locations_to_suppliers()

    # ========================================================================
    # Step 5: Feature Builder (SBERT-Primary Retrieval)
    # ========================================================================
    if not args.skip_features:
        print("\n[5/7] Feature Builder (SBERT-Primary Retrieval)")
        feature_build_succeeded = False
        try:
            from features.feature_builder import build_training_data
            build_training_data(
                top_k=50,
                queries_per_constraint=3,
                gamma=args.gamma,
                hard_negative_ratio=args.neg_ratio
            )
            feature_build_succeeded = True
        except TypeError as e:
            print(f"   Trying fallback parameters...")
            try:
                from features.feature_builder import build_training_data
                build_training_data(top_k=50, queries_per_constraint=3)
                feature_build_succeeded = True
            except Exception as e2:
                print(f"❌ Feature builder failed: {e2}")
        except Exception as e:
            print(f"❌ Feature builder failed: {e}")

        # NOTE: previously this just printed a warning and let the pipeline
        # continue into training/analysis using whatever ranking_data.csv
        # already existed on disk — silently training and evaluating on
        # stale data while still reporting "Pipeline Complete!" at the end.
        # Abort instead, so a real failure can't be missed.
        if not feature_build_succeeded:
            print("\n❌ Feature builder did not complete successfully.")
            print("   Aborting — refusing to train or evaluate on stale/missing training data.")
            return

        if cfg.TRAINING_DATA.exists():
            add_fairness_weights_to_training()
        else:
            print("\n⚠️ Training data not created yet - skipping fairness weights")
    else:
        print("\n[5/7] Feature Builder — SKIPPED")

    # ========================================================================
    # Step 6: Train Models (XGBRanker Primary + LightGBM Backup)
    # ========================================================================

    # Check if training data exists
    if not cfg.TRAINING_DATA.exists():
        print("\n❌ Training data not found. Run without --skip-features first.")
        print("   Try: python features/feature_builder.py")
        return

    # Load data for training
    df = pd.read_csv(str(cfg.TRAINING_DATA))
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df['relevance'] = df['relevance'].round().clip(0, 5).astype(int)

    print(f"\n📊 Loaded {len(df):,} rows, {df['query_id'].nunique()} queries")

    # Display label distribution
    label_counts = df['relevance'].value_counts().sort_index()
    print("   Label distribution:")
    for label in range(6):
        count = label_counts.get(label, 0)
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2) if pct > 0 else "▪"
        print(f"      Label {label}: {count:6,} ({pct:5.1f}%) {bar}")

    # Split by query (no leakage)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df['query_id']))

    df_train_full = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # FIX: previously df_test was used for both early stopping (inside
    # train_xgbranker_inline / train_lambdarank_inline) AND the final
    # reported NDCG@10. That means the number of boosting rounds was
    # chosen specifically to do well on the same data the final metric
    # was computed on — a leak that makes the reported "test" NDCG
    # optimistic/unreliable and not comparable to the rule-based baseline,
    # which never gets tuned against anything. Carving a separate
    # validation split out of the training data fixes this: df_test now
    # stays untouched until the very end.
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx2, val_idx2 = next(val_splitter.split(df_train_full, groups=df_train_full['query_id']))

    df_train = df_train_full.iloc[train_idx2].reset_index(drop=True)
    df_val = df_train_full.iloc[val_idx2].reset_index(drop=True)

    print(f"\n   Train: {len(df_train):,} rows ({df_train['query_id'].nunique()} queries)")
    print(f"   Val:   {len(df_val):,} rows ({df_val['query_id'].nunique()} queries) — early stopping only")
    print(f"   Test:  {len(df_test):,} rows ({df_test['query_id'].nunique()} queries) — held out, used once for final metrics")

    # ====================================================================
    # DATA CLEANING BEFORE EVALUATION (NaN/Inf handling)
    # ====================================================================
    # FIX: this cleaning block previously only applied to df_test. df_train
    # (and the new df_val) went into model.fit() with whatever NaNs/Infs
    # were already in the feature columns — including the now-restored
    # years_normalized/is_manufacturer columns, which the FIX-1 sanitize
    # list below already accounted for but never actually got applied to.
    # Cleaning all three splits identically removes that inconsistency.
    def _clean_split(d):
        d = d.copy()
        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna(subset=['relevance'])
        d = d.fillna(0)
        for _col in ["price_match", "faiss_score", "cert_match", "years_normalized",
                     "price_distance", "location_match", "is_manufacturer"]:
            if _col in d.columns:
                d[_col] = d[_col].fillna(0).replace([np.inf, -np.inf], 0)
        return d

    df_train = _clean_split(df_train)
    df_val = _clean_split(df_val)
    df_test = _clean_split(df_test)

    # Pre-compute rule-based baseline for comparison
    rule_pred = rule_based_score(df_test)
    rule_pred = np.nan_to_num(rule_pred, nan=0.0, posinf=1.0, neginf=0.0)
    assert not np.isnan(rule_pred).any(), "NaNs remain in rule_pred after sanitization"

    print(f"   NaNs in rule_pred: {np.isnan(rule_pred).sum()} (should be 0)")

    rule_ndcg = ndcg_at_k(df_test['relevance'], rule_pred, df_test['query_id'], k=10, verbose=True)
    print(f"\n📊 Rule-Based Baseline NDCG@10: {rule_ndcg:.4f}")

    # --------------------------------------------------------------------
    # Train XGBRanker (Primary)
    # --------------------------------------------------------------------
    xgb_model = None
    if args.train_xgbranker:
        print("\n[6a/7] XGBRanker Training (Primary Ranking Model)")
        xgb_model = train_xgbranker_inline(df_train, df_val)

        if xgb_model:
            xgb_pred = xgb_model.predict(df_test[FEATURE_COLS].values.astype(np.float32))
            xgb_pred = np.nan_to_num(xgb_pred, nan=0.0, posinf=1.0, neginf=0.0)
            xgb_ndcg = ndcg_at_k(df_test['relevance'], xgb_pred, df_test['query_id'], k=10, verbose=True)

            print("\n" + "=" * 65)
            print("📊 XGBRanker RESULTS (held-out test set, never used for early stopping)")
            print("=" * 65)
            print(f"   Rule-Based Baseline NDCG@10:  {rule_ndcg:.4f}")
            print(f"   XGBRANKER NDCG@10:             {xgb_ndcg:.4f}")
            print(f"   Improvement:                   {xgb_ndcg - rule_ndcg:+.4f}")

            # Save XGBRanker model
            import pickle
            os.makedirs(str(cfg.MODELS_DIR), exist_ok=True)
            xgb_path = cfg.MODELS_DIR / "xgb_ranker.pkl"
            with open(xgb_path, "wb") as f:
                pickle.dump({"model": xgb_model, "feature_cols": FEATURE_COLS}, f)
            print(f"\n💾 XGBRanker saved: {xgb_path}")

            # FIX: previously also wrote this XGBRanker object to
            # cfg.LGBM_MODEL "for backward compatibility". That path is
            # also where the LightGBM branch below (and train_lambdarank.py)
            # saves its Booster, and where 6 eval scripts (stability.py,
            # fairness.py, sensitivity.py, baselines.py, ablation.py) load
            # from expecting a LightGBM object. Whichever of the two
            # trainers ran last silently won and corrupted the others.
            # xgb_ranker.pkl above is this model's correct, dedicated home.
            print(f"   (NOT written to {cfg.LGBM_MODEL} — that path is "
                  f"reserved for the LightGBM model. Use xgb_ranker.pkl to "
                  f"load XGBRanker elsewhere, e.g. eval/shap_analysis.py.)")
    else:
        print("\n[6a/7] XGBRanker Training — SKIPPED (use --train-xgbranker or --full)")

    # --------------------------------------------------------------------
    # Train LightGBM LambdaRank (Backup/Comparison)
    # --------------------------------------------------------------------
    if args.train_lambdarank:
        print("\n[6b/7] LightGBM LambdaRank Training (Standard LambdaRank, Production Model)")
        # FIX: train_lambdarank.py owns its own load/split/normalise/save
        # pipeline (leak-fixed — see its FIX-1) and writes cfg.LGBM_MODEL /
        # cfg.LGBM_MODEL_STANDARD itself, so df_train/df_val/df_test from
        # run_all.py's own split are intentionally NOT passed through here.
        # The two scripts would otherwise need byte-identical splitting,
        # cleaning, and normalisation logic to produce comparable numbers,
        # and any future edit to one without the other would silently
        # reintroduce the train/test divergence this whole fix is about.
        lgbm_ok = train_lambdarank_via_module(rounds=500)

        if lgbm_ok:
            with open(str(cfg.LGBM_MODEL), "rb") as f:
                lgb_booster = pickle.load(f)
            lgb_pred = lgb_booster.predict(df_test[FEATURE_COLS].values.astype(np.float32))
            lgb_pred = np.nan_to_num(lgb_pred, nan=0.0, posinf=1.0, neginf=0.0)
            lgb_ndcg = ndcg_at_k(df_test['relevance'], lgb_pred, df_test['query_id'], k=10, verbose=True)

            print("\n" + "=" * 65)
            print("📊 LightGBM LambdaRank RESULTS (run_all.py's own held-out test set)")
            print("=" * 65)
            print(f"   Rule-Based Baseline NDCG@10:  {rule_ndcg:.4f}")
            print(f"   LAMBDARANK NDCG@10:           {lgb_ndcg:.4f}")
            print(f"   Improvement:                  {lgb_ndcg - rule_ndcg:+.4f}")
            print(f"   NOTE: this NDCG@10 is computed on run_all.py's test split, which")
            print(f"   may differ from train_lambdarank.py's own internal test split")
            print(f"   (different random seed / source dataframe). For the paper's")
            print(f"   authoritative number, use train_lambdarank.py's own printed")
            print(f"   evaluate_all() output, not this comparison line.")
    else:
        print("\n[6b/7] LightGBM Training — SKIPPED (use --train-lambdarank or --full)")

    # ========================================================================
    # Step 7: Evaluation Analysis
    # ========================================================================
    if args.run_analysis:
        print("\n[7/7] Running Analysis Suite")

        analyses = [
            ("Ablation Study", "eval.ablation", "run_ablation"),
            ("Baseline Comparison", "eval.baselines", "run_baselines"),
            ("Fairness Analysis", "eval.fairness", "run_fairness_analysis"),
            ("Label Noise Analysis", "eval.label_noise_analysis", "run_label_noise_analysis"),
            ("Sensitivity Analysis", "eval.sensitivity", "run_sensitivity"),
            ("Stability Analysis", "eval.stability", "run_stability"),
        ]

        for name, module_path, func_name in analyses:
            try:
                print(f"\n  ▶ {name}...")
                import importlib
                try:
                    mod = importlib.import_module(module_path)
                    getattr(mod, func_name)()
                    print(f"  ✅ {name} complete")
                except ImportError:
                    alt = f"eval.{module_path.split('.')[-1]}"
                    alt_mod = importlib.import_module(alt)
                    getattr(alt_mod, func_name)()
                    print(f"  ✅ {name} complete (fallback)")
            except ImportError:
                print(f"  ⚠️  {name} skipped — module not found")
            except Exception as e:
                print(f"  ⚠️  {name} failed: {e}")
    else:
        print("\n[7/7] Analysis Suite — SKIPPED (use --run-analysis or --full)")

    # ========================================================================
    # Step 7b: SHAP Analysis
    # ========================================================================
    if args.shap_analysis:
        print("\n[7b] SHAP Explainability Analysis")
        try:
            import shap
            try:
                from eval.shap_analysis import run_shap_analysis
                run_shap_analysis()
                print("  ✅ SHAP analysis complete")
            except ImportError:
                print("  ⚠️  eval.shap_analysis module not found")
        except ImportError:
            print("  ⚠️  SHAP not installed. Run: pip install shap")
    else:
        print("\n[7b] SHAP Analysis — SKIPPED (use --shap-analysis or --full)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 65)
    print("✅ Pipeline Complete!")
    print("=" * 65)
    print(f"\n📂 Output locations:")
    print(f"   Clean data    : {cfg.CLEAN_DATA}")
    print(f"   FAISS index   : {cfg.FAISS_INDEX}")
    print(f"   Training data : {cfg.TRAINING_DATA}")
    if args.train_xgbranker:
        print(f"   XGBRanker     : {cfg.MODELS_DIR / 'xgb_ranker.pkl'}")
    if args.train_lambdarank:
        print(f"   LightGBM      : {cfg.LGBM_MODEL}")
    print("=" * 65)

    print("\n📋 Quick commands:")
    print("   Full pipeline:        python pipeline/run_all.py --full")
    print("   Train XGBRanker only: python pipeline/run_all.py --train-xgbranker")
    print("   Train LightGBM only:  python pipeline/run_all.py --train-lambdarank")
    print("   Analysis only:        python pipeline/run_all.py --run-analysis")
    print()


if __name__ == "__main__":
    main()