"""
Train XGBRanker - Primary LTR Model for SourceUp
------------------------------------------------
Trains XGBRanker with pairwise ranking objective.
This is the primary ranking model for Stage 2.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import cfg

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("❌ XGBoost not installed. Run: pip install xgboost")
    sys.exit(1)

FEATURE_COLS = [
    "price_match", "price_ratio", "price_distance",
    "location_match", "cert_match", "years_normalized",
    "is_manufacturer", "is_trading_company",
    "faiss_score", "faiss_rank",
]

XGB_PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.5,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}


def ndcg_at_k(y_true, y_pred, query_ids, k=10):
    import pandas as pd
    df_n = pd.DataFrame({'t': y_true, 'p': y_pred, 'q': query_ids})
    scores = []
    for qid, grp in df_n.groupby('q'):
        if len(grp) < 2:
            continue
        t = np.nan_to_num(grp['t'].values, nan=0.0).reshape(1, -1)
        p = np.nan_to_num(grp['p'].values, nan=0.0).reshape(1, -1)
        if len(np.unique(t)) <= 1 or np.all(p == p[0][0]):
            continue
        try:
            scores.append(ndcg_score(t, p, k=k))
        except Exception:
            continue
    return np.mean(scores) if scores else 0.0


def main():
    print("=" * 65)
    print("🚀 Training XGBRanker (Pairwise Ranking)")
    print("=" * 65)

    # Load data
    if not os.path.exists(str(cfg.TRAINING_DATA)):
        print(f"❌ Training data not found: {cfg.TRAINING_DATA}")
        print("   Run: python features/feature_builder.py")
        return

    df = pd.read_csv(str(cfg.TRAINING_DATA))
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df["relevance"] = df["relevance"].round().clip(0, 5).astype(int)

    # ── Drop raw location / tier columns — they must never reach the model ──
    # location_match (a derived binary feature) is fine; the raw city name is not.
    # Keeping raw strings would let the model learn a Metro-city proxy signal,
    # causing counterfactual unfairness (Experiment 4 in fairness.py).
    cols_to_drop = [c for c in ["location", "tier", "supplier_name", "query_text"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"   ⚠️  Dropped non-feature columns: {cols_to_drop}")

    print(f"📊 Loaded: {len(df):,} rows, {df['query_id'].nunique()} queries")
    print(f"   Label distribution:\n{df['relevance'].value_counts().sort_index()}")

    # ── Normalize numeric features to [0, 1] ─────────────────────────────
    numeric_cols = ['price_ratio', 'price_distance', 'years_normalized', 'faiss_score', 'faiss_rank']
    for col in numeric_cols:
        if col in df.columns:
            col_min, col_max = df[col].min(), df[col].max()
            if col_max > col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)

    # ── Impute missing feature values ────────────────────────────────────
    for col in FEATURE_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Split by query (no leakage)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["query_id"]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)

    # ── Verify no query overlap ───────────────────────────────────────────
    overlap = set(df_train["query_id"].unique()) & set(df_val["query_id"].unique())
    if overlap:
        print(f"   ❌ DATA LEAKAGE: {len(overlap)} queries appear in both train and val!")
    else:
        print(f"   ✅ No query overlap between train and val")

    # Build query groups for XGBRanker
    train_groups = df_train.groupby("query_id").size().values
    val_groups = df_val.groupby("query_id").size().values

    train_qid = np.concatenate([[i] * size for i, size in enumerate(train_groups)])
    val_qid = np.concatenate([[i] * size for i, size in enumerate(val_groups)])

    print(f"\n   Train: {len(df_train):,} rows, {len(train_groups)} queries")
    print(f"   Val:   {len(df_val):,} rows, {len(val_groups)} queries")

    # Prepare data
    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    X_val   = df_val[FEATURE_COLS].values.astype(np.float32)
    y_train = df_train["relevance"].values
    y_val   = df_val["relevance"].values

    # FIX 4: Add tiny noise to break feature ties → prevents "no further splits" in LightGBM
    X_train += np.random.normal(0, 1e-6, X_train.shape).astype(np.float32)
    X_val   += np.random.normal(0, 1e-6, X_val.shape).astype(np.float32)

    # Train XGBRanker
    print("\n🔧 Training XGBRanker...")
    model = xgb.XGBRanker(**XGB_PARAMS)

    model.fit(
        X_train, y_train,
        qid=train_qid,
        eval_set=[(X_val, y_val)],
        eval_qid=[val_qid],
        verbose=True
    )

    # Evaluate
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    train_ndcg = ndcg_at_k(y_train, pred_train, df_train["query_id"])
    val_ndcg = ndcg_at_k(y_val, pred_val, df_val["query_id"])

    # Rule-based baseline
    rule_pred = (
        df_val["price_match"] * 0.35 +
        (1 - df_val["price_distance"]) * 0.10 +
        df_val["location_match"] * 0.20 +
        df_val["cert_match"] * 0.20 +
        df_val["years_normalized"] * 0.05 +
        df_val["is_manufacturer"] * 0.05 +
        df_val["faiss_score"] * 0.05
    ).values
    rule_ndcg = ndcg_at_k(y_val, rule_pred, df_val["query_id"])

    print("\n" + "=" * 65)
    print("📊 RESULTS")
    print("=" * 65)
    print(f"   Rule-Based NDCG@10:     {rule_ndcg:.4f}")
    print(f"   XGBRanker Train NDCG@10: {train_ndcg:.4f}")
    print(f"   XGBRanker Val NDCG@10:   {val_ndcg:.4f}")
    print(f"   Improvement:             {val_ndcg - rule_ndcg:+.4f}")

    # Save model
    os.makedirs(str(cfg.MODELS_DIR), exist_ok=True)

    # Save as XGBRanker (primary)
    xgb_path = cfg.MODELS_DIR / "xgb_ranker.pkl"
    with open(xgb_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)
    print(f"\n💾 XGBRanker saved: {xgb_path}")

    # Also save to LGBM path for backward compatibility
    with open(str(cfg.LGBM_MODEL), "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Also saved to: {cfg.LGBM_MODEL} (compatibility)")

    return model


if __name__ == "__main__":
    main()