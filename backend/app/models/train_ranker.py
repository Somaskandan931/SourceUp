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
    scores = []
    for qid in query_ids.unique():
        mask = query_ids == qid
        if mask.sum() < 2:
            continue
        t = y_true[mask].values.reshape(1, -1)
        p = y_pred[mask].reshape(1, -1)
        scores.append(ndcg_score(t, p, k=k))
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

    print(f"📊 Loaded: {len(df):,} rows, {df['query_id'].nunique()} queries")
    print(f"   Label distribution:\n{df['relevance'].value_counts().sort_index()}")

    # Split by query (no leakage)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["query_id"]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    # Build query groups for XGBRanker
    train_groups = df_train.groupby("query_id").size().values
    val_groups = df_val.groupby("query_id").size().values

    train_qid = np.concatenate([[i] * size for i, size in enumerate(train_groups)])
    val_qid = np.concatenate([[i] * size for i, size in enumerate(val_groups)])

    print(f"\n   Train: {len(df_train):,} rows, {len(train_groups)} queries")
    print(f"   Val:   {len(df_val):,} rows, {len(val_groups)} queries")

    # Prepare data
    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    X_val = df_val[FEATURE_COLS].values.astype(np.float32)
    y_train = df_train["relevance"].values
    y_val = df_val["relevance"].values

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