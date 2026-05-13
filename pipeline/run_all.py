"""
SourceUp — Full Pipeline Orchestrator (UPDATED with SBERT-Primary + XGBRanker)
--------------------------------------
Runs the complete data pipeline from raw scraping to model training
and evaluation. NOW with SBERT as PRIMARY retrieval and XGBRanker.

Usage:
    python pipeline/run_all.py --full                   # Run everything
    python pipeline/run_all.py --assign-locations      # Assign locations to suppliers
    python pipeline/run_all.py --train-xgbranker       # Train XGBRanker (primary)
    python pipeline/run_all.py --train-lambdarank      # Train LightGBM (backup)
    python pipeline/run_all.py --run-analysis          # Run evaluation analysis
    python pipeline/run_all.py --shap-analysis         # Run SHAP analysis
    python pipeline/run_all.py --limit 1000            # Limit dataset size
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg


FEATURE_COLS = [
    "price_match", "price_ratio", "price_distance",
    "location_match", "cert_match", "years_normalized",
    "is_manufacturer", "is_trading_company",
    "faiss_score", "faiss_rank",
]


def ndcg_at_k(y_true, y_pred, query_ids, k=10):
    """Calculate NDCG@k."""
    scores = []
    for qid in query_ids.unique():
        mask = query_ids == qid
        if mask.sum() < 2:
            continue
        t = y_true[mask].values.reshape(1, -1)
        p = y_pred[mask].reshape(1, -1)
        scores.append(ndcg_score(t, p, k=k))
    return np.mean(scores) if scores else 0.0


def rule_based_score(df):
    """Rule-based scoring function."""
    return (
        df["price_match"].values * 0.35 +
        (1 - df["price_distance"].values) * 0.10 +
        df["location_match"].values * 0.20 +
        df["cert_match"].values * 0.20 +
        df["years_normalized"].values * 0.05 +
        df["is_manufacturer"].values * 0.05 +
        df["faiss_score"].values * 0.05
    )


def train_xgbranker_inline(df_train, df_test):
    """Train XGBRanker (primary ranking model)."""
    try:
        import xgboost as xgb
    except ImportError:
        print("   ❌ XGBoost not installed. Run: pip install xgboost")
        return None

    print("\n🚀 Training XGBRanker (Primary Ranking Model)")
    print("=" * 50)

    # Build query groups
    train_groups = df_train.groupby('query_id').size().values
    test_groups = df_test.groupby('query_id').size().values

    train_qid = np.concatenate([[i] * size for i, size in enumerate(train_groups)])
    test_qid = np.concatenate([[i] * size for i, size in enumerate(test_groups)])

    # XGBRanker parameters (optimized for ranking)
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
        random_state=42,
        verbosity=1
    )

    print(f"   Train: {len(df_train):,} rows, {len(train_groups)} queries")
    print(f"   Test:  {len(df_test):,} rows, {len(test_groups)} queries")

    model.fit(
        df_train[FEATURE_COLS].values.astype(np.float32),
        df_train['relevance'].values,
        qid=train_qid,
        eval_set=[(df_test[FEATURE_COLS].values.astype(np.float32), df_test['relevance'].values)],
        eval_qid=[test_qid],
        verbose=True
    )

    return model


def train_lambdarank_inline(df_train, df_test):
    """Train LightGBM LambdaRank (backup/comparison model)."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("   ❌ LightGBM not installed. Run: pip install lightgbm")
        return None

    print("\n🌳 Training LightGBM LambdaRank (Backup Model)")
    print("=" * 50)

    train_groups = df_train.groupby('query_id').size().values
    test_groups = df_test.groupby('query_id').size().values

    # Calculate sample weights for balanced training
    label_counts = df_train['relevance'].value_counts()
    sample_weights = np.ones(len(df_train))
    for label in [1, 2, 3, 4, 5]:
        if label in label_counts:
            weight = label_counts[0] / (label_counts[label] * 5) if label_counts[0] > 0 else 1.0
            sample_weights[df_train['relevance'] == label] = min(weight, 5.0)

    # Add fairness weights if available
    if 'fairness_weight' in df_train.columns:
        sample_weights = sample_weights * df_train['fairness_weight'].values

    model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[10],
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.5,
        reg_alpha=0.5,
        min_child_samples=10,
        random_state=42,
        verbose=1
    )

    print(f"   Train: {len(df_train):,} rows, {len(train_groups)} queries")
    print(f"   Test:  {len(df_test):,} rows, {len(test_groups)} queries")

    model.fit(
        df_train[FEATURE_COLS].values.astype(np.float32),
        df_train['relevance'].values,
        group=train_groups,
        sample_weight=sample_weights,
        eval_set=[(df_test[FEATURE_COLS].values.astype(np.float32), df_test['relevance'].values)],
        eval_group=[test_groups],
        eval_metric='ndcg@10',
        callbacks=[lgb.early_stopping(20, verbose=True)]
    )

    return model


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

        return rng.choice(cities)

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

        df['fairness_weight'] = 1.0
        if 'Tier-2' in tier_counts and tier_counts.get('Metro', 0) > 0:
            weight = tier_counts['Metro'] / max(tier_counts['Tier-2'], 1)
            df.loc[df['city_tier'] == 'Tier-2', 'fairness_weight'] = min(weight, 5.0)
            print(f"\n   Tier-2 weight multiplier: {min(weight, 5.0):.2f}")

        if 'Tier-3' in tier_counts and tier_counts.get('Metro', 0) > 0:
            weight = tier_counts['Metro'] / max(tier_counts['Tier-3'], 1)
            df.loc[df['city_tier'] == 'Tier-3', 'fairness_weight'] = min(weight, 5.0)
            print(f"   Tier-3 weight multiplier: {min(weight, 5.0):.2f}")

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
  python pipeline/run_all.py --assign-locations         # Only assign locations
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
                        help="Assign smart locations to suppliers (fix fairness)")
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
    print(f"   Assign Loc    : {'Yes' if args.assign_locations else 'No'}")
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
    try:
        from pipeline.incremental_faiss import incremental_update
        incremental_update()
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        return

    # ========================================================================
    # Step 4: Assign Locations (Fixes Fairness Issue)
    # ========================================================================
    if args.assign_locations:
        print("\n[4/7] Assign Smart Locations")
        assign_locations_to_suppliers()
    else:
        print("\n[4/7] Assign Locations — SKIPPED (use --assign-locations or --full)")

    # ========================================================================
    # Step 5: Feature Builder (SBERT-Primary Retrieval)
    # ========================================================================
    if not args.skip_features:
        print("\n[5/7] Feature Builder (SBERT-Primary Retrieval)")
        try:
            from features.feature_builder import build_training_data
            # FIXED: Use correct parameter names for your feature_builder.py
            build_training_data(
                top_k=50,                    # Changed from top_k_faiss to top_k
                queries_per_constraint=3,    # Changed from queries_per_constraint
                gamma=args.gamma,
                hard_negative_ratio=args.neg_ratio
            )
        except TypeError as e:
            # Try fallback with fewer parameters
            print(f"   Trying fallback parameters...")
            try:
                from features.feature_builder import build_training_data
                build_training_data(top_k=50, queries_per_constraint=3)
            except Exception as e2:
                print(f"⚠️  Feature builder failed: {e2}")
        except Exception as e:
            print(f"⚠️  Feature builder failed: {e}")

        # Add fairness weights to training data (only if training data was created)
        if args.assign_locations and cfg.TRAINING_DATA.exists():
            add_fairness_weights_to_training()
        elif args.assign_locations:
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

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print(f"\n   Train: {len(df_train):,} rows ({df_train['query_id'].nunique()} queries)")
    print(f"   Test:  {len(df_test):,} rows ({df_test['query_id'].nunique()} queries)")

    # Pre-compute rule-based baseline for comparison
    rule_pred = rule_based_score(df_test)
    rule_ndcg = ndcg_at_k(df_test['relevance'], rule_pred, df_test['query_id'], k=10)
    print(f"\n📊 Rule-Based Baseline NDCG@10: {rule_ndcg:.4f}")

    # --------------------------------------------------------------------
    # Train XGBRanker (Primary)
    # --------------------------------------------------------------------
    xgb_model = None
    if args.train_xgbranker:
        print("\n[6a/7] XGBRanker Training (Primary Ranking Model)")
        xgb_model = train_xgbranker_inline(df_train, df_test)

        if xgb_model:
            xgb_pred = xgb_model.predict(df_test[FEATURE_COLS].values.astype(np.float32))
            xgb_ndcg = ndcg_at_k(df_test['relevance'], xgb_pred, df_test['query_id'], k=10)

            print("\n" + "=" * 65)
            print("📊 XGBRanker RESULTS")
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

            # Also save to LGBM path for backward compatibility
            with open(str(cfg.LGBM_MODEL), "wb") as f:
                pickle.dump(xgb_model, f)
            print(f"💾 Also saved to: {cfg.LGBM_MODEL} (compatibility)")
    else:
        print("\n[6a/7] XGBRanker Training — SKIPPED (use --train-xgbranker or --full)")

    # --------------------------------------------------------------------
    # Train LightGBM LambdaRank (Backup/Comparison)
    # --------------------------------------------------------------------
    if args.train_lambdarank:
        print("\n[6b/7] LightGBM LambdaRank Training (Backup Model)")
        lgb_model = train_lambdarank_inline(df_train, df_test)

        if lgb_model:
            lgb_pred = lgb_model.predict(df_test[FEATURE_COLS].values.astype(np.float32))
            lgb_ndcg = ndcg_at_k(df_test['relevance'], lgb_pred, df_test['query_id'], k=10)

            print("\n" + "=" * 65)
            print("📊 LightGBM LambdaRank RESULTS")
            print("=" * 65)
            print(f"   Rule-Based Baseline NDCG@10:  {rule_ndcg:.4f}")
            print(f"   LAMBDARANK NDCG@10:           {lgb_ndcg:.4f}")
            print(f"   Improvement:                  {lgb_ndcg - rule_ndcg:+.4f}")

            # Save LightGBM model
            import pickle
            os.makedirs(str(cfg.MODELS_DIR), exist_ok=True)
            with open(str(cfg.LGBM_MODEL), "wb") as f:
                pickle.dump(lgb_model.booster_, f)
            print(f"\n💾 LightGBM saved: {cfg.LGBM_MODEL}")
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
    print("   Fix fairness only:    python pipeline/run_all.py --assign-locations")
    print("   Train XGBRanker only: python pipeline/run_all.py --train-xgbranker")
    print("   Train LightGBM only:  python pipeline/run_all.py --train-lambdarank")
    print("   Analysis only:        python pipeline/run_all.py --run-analysis")
    print()


if __name__ == "__main__":
    main()