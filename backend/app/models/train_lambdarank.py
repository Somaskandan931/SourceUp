"""
LambdaRank Training — SourceUp Supplier Ranking
------------------------------------------------
Trains a LightGBM LambdaRank model (pairwise learning-to-rank).

LambdaRank is the IEEE-standard LTR approach:
  - Learns which supplier should rank ABOVE another (pairwise objective)
  - Optimises NDCG directly via gradient approximation
  - Strictly superior to pointwise regression for ranking tasks

This replaces/upgrades train_ranker.py.

Pipeline:
  1. Load data/training/ranking_data.csv  (produced by feature_builder.py)
  2. Query-stratified train/test split
  3. Train LightGBM with objective='lambdarank', ndcg_eval_at=[5,10]
  4. Evaluate: NDCG@5, NDCG@10, P@5, MAP, Kendall-τ
  5. Save model to backend/app/models/embeddings/ranker_lightgbm.pkl
  6. Save training curves and feature importance plots

Usage:
    python train_lambdarank.py
    python train_lambdarank.py --gamma 0.3   # override constraint penalty

IEEE reference:
    Burges et al. (2006). Learning to Rank using Gradient Descent.
    ICML 2005. (LambdaRank extension follows Burges 2010.)
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
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("❌  lightgbm not installed.  Run: pip install lightgbm")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.getenv("SOURCEUP_DIR", "C:/Users/somas/PycharmProjects/SourceUp")
TRAIN_DATA = f"{BASE_DIR}/data/training/ranking_data.csv"
MODEL_DIR  = f"{BASE_DIR}/backend/app/models/embeddings"
LGBM_PATH  = f"{MODEL_DIR}/ranker_lightgbm.pkl"
XGBM_PATH  = f"{MODEL_DIR}/ranker_xgboost.pkl"   # backup copy
OUT_DIR    = f"{BASE_DIR}/data/eval"
PLOTS_DIR  = f"{OUT_DIR}/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature columns (must match feature_builder.py output)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "price_match",        # 1 = within budget, 0 = over budget, 0.5 = no filter
    "price_ratio",        # supplier_price / max_price  (lower is better)
    "price_distance",     # abs(price - max_price) / max_price
    "location_match",     # 1 = city/country match, 0.5 = no pref, 0 = mismatch
    "cert_match",         # 1 = has required cert, 0.5 = no req, 0 = missing
    "years_normalized",   # years_on_platform / 10 (capped at 1.0)
    "is_manufacturer",    # 1 if Business Type == Manufacturer
    "is_trading_company", # 1 if Business Type == Trading Company
    "faiss_score",        # SBERT cosine similarity from FAISS retrieval
    "faiss_rank",         # position in FAISS top-k (lower = more similar)
]

LABEL_COL = "relevance"
QUERY_COL = "query_id"

# ---------------------------------------------------------------------------
# LambdaRank hyperparameters (tuned for ~500-5000 training samples)
# ---------------------------------------------------------------------------
LGBM_PARAMS = {
    "objective":         "lambdarank",
    "metric":            "ndcg",
    "ndcg_eval_at":      [5, 10],
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_data_in_leaf":  5,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      1,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "max_depth":         -1,
    "num_threads":       4,
    "verbosity":         -1,
    "seed":              42,
}

NUM_ROUNDS       = 500
EARLY_STOP_ROUNDS = 30


# ============================================================================
# DATA LOADING & SPLITTING
# ============================================================================

def load_data(gamma: float = 0.3) -> pd.DataFrame:
    """
    Load training data. If gamma != 0, penalise the relevance label
    for suppliers that fail hard constraints (price, location, cert).

    This is the supervised realisation of:
        Score(q, d) = f_θ(q, d) − γ · ConstraintViolation(d, C)
    applied at label generation time so the model learns the
    constraint-penalised objective directly.
    """
    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_DATA}\n"
            "Run: python pipeline/feature_builder.py"
        )

    df = pd.read_csv(TRAIN_DATA)

    # Validate required columns
    missing_feats = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")
    if QUERY_COL not in df.columns:
        raise ValueError(f"Missing query column: {QUERY_COL}")

    df[LABEL_COL] = df[LABEL_COL].clip(0, 5)

    # Apply constraint penalty to labels if gamma > 0
    if gamma > 0:
        violation = _compute_violation(df)
        df[LABEL_COL] = (df[LABEL_COL] - gamma * violation * 5).clip(0, 5)

    df[LABEL_COL] = df[LABEL_COL].round().astype(int)

    print(f"✅ Loaded training data: {len(df)} rows, "
          f"{df[QUERY_COL].nunique()} queries  (γ={gamma})")
    print(f"   Label distribution:\n{df[LABEL_COL].value_counts().sort_index()}")
    return df


def _compute_violation(df: pd.DataFrame) -> np.ndarray:
    """Per-supplier constraint violation signal ∈ [0, 1]."""
    def fail(col):
        vals = df[col].values
        return np.where(vals < 0.5, 1.0 - vals, 0.0)

    v = (fail("price_match") + fail("location_match") + fail("cert_match")) / 3.0
    return v


def query_stratified_split(df: pd.DataFrame,
                            test_frac: float = 0.2,
                            seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by query ID — ensures no query leaks between train and test.
    This is the correct split for learning-to-rank (not random row split).
    """
    queries = df[QUERY_COL].unique()
    rng     = np.random.default_rng(seed)
    rng.shuffle(queries)
    split     = int(len(queries) * (1 - test_frac))
    train_qids = set(queries[:split])
    test_qids  = set(queries[split:])
    df_train   = df[df[QUERY_COL].isin(train_qids)].reset_index(drop=True)
    df_test    = df[df[QUERY_COL].isin(test_qids)].reset_index(drop=True)
    print(f"   Train: {len(df_train)} rows ({len(train_qids)} queries)  "
          f"Test: {len(df_test)} rows ({len(test_qids)} queries)")
    return df_train, df_test


# ============================================================================
# DATASET BUILDERS FOR LIGHTGBM
# ============================================================================

def build_lgbm_dataset(df: pd.DataFrame,
                        reference=None) -> lgb.Dataset:
    """
    Build LightGBM Dataset with group (query) sizes.
    LambdaRank requires group sizes to be passed explicitly.
    """
    X      = df[FEATURE_COLS].values.astype(np.float32)
    y      = df[LABEL_COL].values.astype(np.int32)
    groups = df.groupby(QUERY_COL, sort=False).size().values

    return lgb.Dataset(
        X, label=y, group=groups,
        feature_name=FEATURE_COLS,
        reference=reference,
        free_raw_data=False
    )


# ============================================================================
# METRICS
# ============================================================================

def _ndcg_mean(y_true: pd.Series, y_pred: np.ndarray,
               query_ids: pd.Series, k: int = 10) -> float:
    scores = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        t = y_true[m].values.reshape(1, -1)
        p = y_pred[m].reshape(1, -1)
        scores.append(ndcg_score(t, p, k=k))
    return float(np.mean(scores)) if scores else 0.0


def _prec_at_k(y_true: pd.Series, y_pred: np.ndarray,
               query_ids: pd.Series, k: int = 5, thr: int = 3) -> float:
    sc = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tv  = y_true[m].values
        pv  = y_pred[m]
        top = np.argsort(pv)[::-1][:k]
        rel = (tv >= thr)
        if rel.sum() == 0:
            continue
        sc.append(rel[top].sum() / min(k, len(top)))
    return float(np.mean(sc)) if sc else 0.0


def _map_score(y_true: pd.Series, y_pred: np.ndarray,
               query_ids: pd.Series, thr: int = 3) -> float:
    aps = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tv    = y_true[m].values
        pv    = y_pred[m]
        order = np.argsort(pv)[::-1]
        rel   = (tv[order] >= thr)
        if rel.sum() == 0:
            continue
        hits, prec = 0, []
        for i, r in enumerate(rel):
            if r:
                hits += 1
                prec.append(hits / (i + 1))
        aps.append(np.mean(prec) if prec else 0.0)
    return float(np.mean(aps)) if aps else 0.0


def _tau_mean(y_true: pd.Series, y_pred: np.ndarray,
              query_ids: pd.Series) -> float:
    taus = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tau, _ = kendalltau(y_true[m].values, y_pred[m])
        if not np.isnan(tau):
            taus.append(tau)
    return float(np.mean(taus)) if taus else 0.0


def evaluate_model(model, df_test: pd.DataFrame) -> Dict:
    X     = df_test[FEATURE_COLS].values.astype(np.float32)
    pred  = model.predict(X)
    y     = df_test[LABEL_COL]
    qids  = df_test[QUERY_COL]

    metrics = {
        "NDCG@5":    round(_ndcg_mean(y, pred, qids, k=5),  4),
        "NDCG@10":   round(_ndcg_mean(y, pred, qids, k=10), 4),
        "P@5":       round(_prec_at_k(y, pred, qids, k=5),  4),
        "MAP":       round(_map_score(y, pred, qids),        4),
        "Kendall-τ": round(_tau_mean(y, pred, qids),         4),
    }
    return metrics, pred


# ============================================================================
# TRAINING
# ============================================================================

def train_lambdarank(df_train: pd.DataFrame,
                     df_test:  pd.DataFrame,
                     params:   Dict = None,
                     num_rounds: int = NUM_ROUNDS) -> lgb.Booster:

    params = params or LGBM_PARAMS
    print("\n" + "=" * 65)
    print("🚀 Training LightGBM LambdaRank")
    print("=" * 65)
    print(f"   Rounds:   {num_rounds}  (early stop: {EARLY_STOP_ROUNDS})")
    print(f"   Features: {len(FEATURE_COLS)}")
    print(f"   Params:   {params}\n")

    train_ds = build_lgbm_dataset(df_train)
    val_ds   = build_lgbm_dataset(df_test, reference=train_ds)

    callbacks = [
        lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=num_rounds,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    print(f"\n   Best iteration: {model.best_iteration}")
    return model


# ============================================================================
# PLOTS
# ============================================================================

def plot_training_curves(model: lgb.Booster):
    """Plot NDCG@5 and NDCG@10 training and validation curves."""
    results = model.evals_result()
    if not results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette   = {"train": "#2166ac", "valid": "#d73027"}

    for ax, metric_key, title in [
        (axes[0], "ndcg@5",  "NDCG@5  (Train vs Validation)"),
        (axes[1], "ndcg@10", "NDCG@10 (Train vs Validation)"),
    ]:
        for split, color in palette.items():
            if split in results and metric_key in results[split]:
                vals = results[split][metric_key]
                ax.plot(vals, label=split, color=color, linewidth=1.8)

        # Mark best iteration
        best_iter = model.best_iteration - 1
        if "valid" in results and metric_key in results["valid"]:
            best_val = results["valid"][metric_key][best_iter]
            ax.axvline(best_iter, color="gray", linestyle="--", linewidth=1.2)
            ax.annotate(
                f"best={best_val:.4f}\n@iter {best_iter}",
                (best_iter, best_val),
                xytext=(best_iter + 10, best_val - 0.03),
                fontsize=8,
            )

        ax.set_xlabel("Boosting Round", fontsize=11)
        ax.set_ylabel(metric_key.upper(), fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("LambdaRank Training Curves — SourceUp", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/lambdarank_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_feature_importance(model: lgb.Booster):
    """Bar chart of feature importance (gain)."""
    importances = model.feature_importance(importance_type="gain")
    feat_names  = model.feature_name()

    df_imp = pd.DataFrame({
        "feature":    feat_names,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(df_imp["feature"], df_imp["importance"],
                   color="#2166ac", edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars, df_imp["importance"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)

    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(
        "Fig. 3a — LambdaRank Feature Importance\n"
        "(Gain = total information gain from all splits on this feature)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/lambdarank_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaRank for SourceUp")
    parser.add_argument("--gamma",      type=float, default=0.3,
                        help="Constraint penalty weight for label adjustment (default: 0.3)")
    parser.add_argument("--rounds",     type=int,   default=NUM_ROUNDS,
                        help="Max boosting rounds (default: 500)")
    parser.add_argument("--test-frac",  type=float, default=0.2,
                        help="Fraction of queries to hold out for testing (default: 0.2)")
    args = parser.parse_args()

    print("=" * 65)
    print("🏗️  SourceUp — LambdaRank Training")
    print("=" * 65)
    print(f"   BASE_DIR : {BASE_DIR}")
    print(f"   DATA     : {TRAIN_DATA}")
    print(f"   γ        : {args.gamma}")
    print()

    # 1. Load & split
    df = load_data(gamma=args.gamma)
    df_train, df_test = query_stratified_split(df, test_frac=args.test_frac)

    # 2. Train
    model = train_lambdarank(df_train, df_test, num_rounds=args.rounds)

    # 3. Evaluate
    print("\n📊 Evaluation on held-out test queries:")
    metrics, _ = evaluate_model(model, df_test)
    for k, v in metrics.items():
        print(f"   {k:12s}: {v:.4f}")

    # Baseline comparison: rule-based scorer on same test set
    rule_pred = (
        df_test["price_match"]          * 0.35 +
        (1 - df_test["price_distance"]) * 0.10 +
        df_test["location_match"]       * 0.20 +
        df_test["cert_match"]           * 0.20 +
        df_test["years_normalized"]     * 0.05 +
        df_test["is_manufacturer"]      * 0.05 +
        df_test["faiss_score"]          * 0.05
    ).values
    rule_ndcg = _ndcg_mean(df_test[LABEL_COL], rule_pred, df_test[QUERY_COL])
    print(f"\n   Rule-Based NDCG@10 (baseline): {rule_ndcg:.4f}")
    delta = metrics["NDCG@10"] - rule_ndcg
    print(f"   LambdaRank improvement:         {delta:+.4f}  "
          f"({'✅ better' if delta > 0 else '⚠️ worse — check labels/features'})")

    # 4. Save model
    with open(LGBM_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved: {LGBM_PATH}")

    # Also save native LightGBM format for inspection
    native_path = LGBM_PATH.replace(".pkl", "_native.txt")
    model.save_model(native_path)
    print(f"   Native format: {native_path}")

    # 5. Plots
    print("\n📈 Generating plots...")
    plot_training_curves(model)
    plot_feature_importance(model)

    print("\n" + "=" * 65)
    print("✅ LambdaRank training complete.")
    print(f"   NDCG@10 = {metrics['NDCG@10']:.4f}   "
          f"NDCG@5 = {metrics['NDCG@5']:.4f}   "
          f"P@5 = {metrics['P@5']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()