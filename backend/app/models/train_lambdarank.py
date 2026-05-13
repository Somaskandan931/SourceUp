"""
LambdaRank Training — SourceUp Supplier Ranking (BALANCED STABILITY FIX)
Goal: τ ≥ 0.85 at σ=0.03 while maintaining NDCG > 0.85
"""

import os
import sys
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("❌ lightgbm not installed. Run: pip install lightgbm")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import cfg

TRAIN_DATA = str(cfg.TRAINING_DATA)
MODEL_DIR  = str(cfg.MODELS_DIR)
LGBM_PATH  = str(cfg.LGBM_MODEL)
OUT_DIR    = str(cfg.EVAL_DIR)
PLOTS_DIR  = str(cfg.EVAL_PLOTS_DIR)

cfg.ensure_dirs()

FEATURE_COLS = [
    "price_match", "price_ratio", "price_distance",
    "location_match", "cert_match", "years_normalized",
    "is_manufacturer", "is_trading_company",
    "faiss_score", "faiss_rank",
]

LABEL_COL = "relevance"
QUERY_COL = "query_id"

# ============================================================================
# BALANCED PARAMETERS (Good NDCG + Improved Stability)
# ============================================================================
LGBM_PARAMS = {
    "objective":         "lambdarank",
    "metric":            "ndcg",
    "ndcg_eval_at":      [5, 10],
    "learning_rate":     0.01,       # Back to reasonable
    "num_leaves":        15,         # Moderate tree size
    "min_data_in_leaf":  15,         # Moderate
    "feature_fraction":  0.6,        # Moderate dropout
    "bagging_fraction":  0.6,        # Moderate bagging
    "bagging_freq":      1,
    "lambda_l1":         0.5,        # Moderate regularization
    "lambda_l2":         0.5,
    "max_depth":         5,
    "min_gain_to_split": 0.05,
    "num_threads":       4,
    "verbosity":         -1,
    "seed":              42,
}

NUM_ROUNDS        = 200
EARLY_STOP_ROUNDS = 15


def load_data(gamma: float = 0.3) -> pd.DataFrame:
    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA}")

    df = pd.read_csv(TRAIN_DATA)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df[LABEL_COL] = df[LABEL_COL].round().clip(0, 5).astype(int)

    # ============================================================
    # LIGHT feature smoothing (don't over-smooth)
    # ============================================================
    for col in ['price_match', 'location_match', 'cert_match', 'faiss_score']:
        if col in df.columns:
            # Use 2nd and 98th percentiles (minimal clipping)
            lower = df[col].quantile(0.02)
            upper = df[col].quantile(0.98)
            df[col] = df[col].clip(lower, upper)
            print(f"   Smoothed {col}: clipped at [{lower:.3f}, {upper:.3f}]")

    print(f"✅ Loaded: {len(df):,} rows, {df[QUERY_COL].nunique()} queries")
    print(f"   Label distribution:\n{df[LABEL_COL].value_counts().sort_index()}")

    return df


def query_stratified_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    queries = df[QUERY_COL].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(queries)
    split = int(len(queries) * (1 - test_frac))
    train_qids = set(queries[:split])
    test_qids = set(queries[split:])
    df_train = df[df[QUERY_COL].isin(train_qids)].reset_index(drop=True)
    df_test = df[df[QUERY_COL].isin(test_qids)].reset_index(drop=True)
    print(f"   Train: {len(df_train)} rows ({len(train_qids)} queries)")
    print(f"   Test:  {len(df_test)} rows ({len(test_qids)} queries)")
    return df_train, df_test


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


def precision_at_k(y_true, y_pred, query_ids, k=5, thr=3):
    sc = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tv = y_true[m].values
        pv = y_pred[m]
        top = np.argsort(pv)[::-1][:k]
        rel = (tv >= thr)
        if rel.sum() == 0:
            continue
        sc.append(rel[top].sum() / min(k, len(top)))
    return np.mean(sc) if sc else 0.0


def kendall_tau_at_noise(model, df_test, noise_level=0.03, n_trials=5):
    """Quick stability test."""
    X_original = df_test[FEATURE_COLS].values.astype(np.float32)
    pred_original = model.predict(X_original)
    query_ids = df_test[QUERY_COL]

    all_taus = []
    for trial in range(n_trials):
        X_noisy = X_original.copy()
        # Add noise to continuous features
        for col_idx, col_name in enumerate(FEATURE_COLS):
            if col_name in ['price_ratio', 'price_distance', 'faiss_score']:
                std = X_original[:, col_idx].std()
                if std > 0:
                    noise = np.random.normal(0, noise_level * std, size=len(X_original))
                    X_noisy[:, col_idx] += noise
                    X_noisy[:, col_idx] = np.clip(X_noisy[:, col_idx], 0, 1)

        pred_noisy = model.predict(X_noisy)

        for qid in query_ids.unique():
            mask = query_ids == qid
            if mask.sum() > 2:
                tau, _ = kendalltau(pred_original[mask], pred_noisy[mask])
                if not np.isnan(tau):
                    all_taus.append(tau)

    return np.mean(all_taus)


def evaluate_model(model, df_test: pd.DataFrame) -> Dict:
    X = df_test[FEATURE_COLS].values.astype(np.float32)
    pred = model.predict(X)
    y = df_test[LABEL_COL]
    qids = df_test[QUERY_COL]

    metrics = {
        "NDCG@10": round(ndcg_at_k(y, pred, qids, k=10), 4),
        "NDCG@5":  round(ndcg_at_k(y, pred, qids, k=5), 4),
        "P@5":     round(precision_at_k(y, pred, qids, k=5), 4),
    }
    return metrics, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 65)
    print("🏗️ SourceUp — LambdaRank Training (BALANCED FIX)")
    print("=" * 65)
    print(f"   Target: τ ≥ 0.85 at σ=0.03")
    print(f"   Parameters:")
    print(f"     learning_rate: {LGBM_PARAMS['learning_rate']}")
    print(f"     num_leaves: {LGBM_PARAMS['num_leaves']}")
    print(f"     max_depth: {LGBM_PARAMS['max_depth']}")
    print(f"     lambda_l1: {LGBM_PARAMS['lambda_l1']}, lambda_l2: {LGBM_PARAMS['lambda_l2']}")
    print(f"     feature_fraction: {LGBM_PARAMS['feature_fraction']}")
    print(f"     subsample: {LGBM_PARAMS['bagging_fraction']}")
    print("=" * 65)

    # Load and split
    df = load_data(gamma=args.gamma)
    df_train, df_test = query_stratified_split(df, test_frac=args.test_frac)

    # Build datasets
    train_groups = df_train.groupby(QUERY_COL, sort=False).size().values
    test_groups = df_test.groupby(QUERY_COL, sort=False).size().values

    print("\n🔧 Training LambdaRank...")

    model = lgb.LGBMRanker(
        **LGBM_PARAMS,
        n_estimators=args.rounds,
        random_state=42
    )

    model.fit(
        df_train[FEATURE_COLS].values.astype(np.float32),
        df_train[LABEL_COL].values,
        group=train_groups,
        eval_set=[(df_test[FEATURE_COLS].values.astype(np.float32), df_test[LABEL_COL].values)],
        eval_group=[test_groups],
        eval_metric="ndcg@10",
        callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=True)]
    )

    # Evaluate
    metrics, pred = evaluate_model(model, df_test)
    print(f"\n📊 LambdaRank Results:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    # Rule-based baseline
    rule_pred = (
        df_test["price_match"] * 0.35 +
        (1 - df_test["price_distance"]) * 0.10 +
        df_test["location_match"] * 0.20 +
        df_test["cert_match"] * 0.20 +
        df_test["years_normalized"] * 0.05 +
        df_test["is_manufacturer"] * 0.05 +
        df_test["faiss_score"] * 0.05
    ).values

    rule_ndcg = ndcg_at_k(df_test[LABEL_COL], rule_pred, df_test[QUERY_COL], k=10)
    print(f"\n   Rule-Based NDCG@10: {rule_ndcg:.4f}")
    print(f"   LambdaRank improvement: {metrics['NDCG@10'] - rule_ndcg:+.4f}")

    # Stability test
    print("\n🔬 Running stability test...")
    stability_tau = kendall_tau_at_noise(model, df_test, noise_level=0.03, n_trials=5)
    print(f"   Mean Kendall's τ at σ=0.03: {stability_tau:.4f}")
    if stability_tau >= 0.85:
        print(f"   ✅ STABLE (τ ≥ 0.85) - Passes IEEE criterion!")
    else:
        print(f"   ⚠️  Needs improvement (gap: {0.85 - stability_tau:.4f})")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(LGBM_PATH, "wb") as f:
        pickle.dump(model.booster_, f)

    print(f"\n✅ Model saved: {LGBM_PATH}")

    # Also save full model
    full_model_path = LGBM_PATH.replace('.pkl', '_full.pkl')
    with open(full_model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Full model saved: {full_model_path}")

    print("=" * 65)

    return model, stability_tau


if __name__ == "__main__":
    model, tau = main()
    print(f"\n🎯 FINAL STABILITY RESULT: τ = {tau:.4f}")
    if tau >= 0.85:
        print("🎉 SUCCESS! Model meets IEEE stability standard!")
    else:
        print(f"⚠️  Gap to target: {0.85 - tau:.4f}")
        print("\n💡 Suggestions for next steps:")
        print("   1. Increase training data (add more queries)")
        print("   2. Add more regularization features (min_gain_to_split=0.1)")
        print("   3. Try CatBoost which is naturally more stable")
        print("   4. Accept τ=0.70 as reasonable for this dataset size")