"""
Label Noise Analysis — SourceUp Supplier Ranking
-------------------------------------------------
Evaluates model robustness to label noise.

FIX: Noise is injected ONLY into the training set labels.
     The test set must remain clean so NDCG degradation is real
     and meaningful — not an artifact of corrupted evaluation.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))
from config import cfg

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit

TRAIN_DATA  = str(cfg.TRAINING_DATA)
OUT_DIR     = str(cfg.EVAL_DIR)
PLOTS_DIR   = str(cfg.EVAL_PLOTS_DIR)
LABEL_COL   = "relevance"
QUERY_COL   = "query_id"

FEATURE_COLS = [
    "price_match", "price_ratio",
    "location_match", "cert_match",
    "faiss_score",
    # NOTE: years_normalized, is_manufacturer, is_trading_company removed —
    # confirmed zero SHAP importance across two independent training runs
    # (near-constant values in current data). Re-add here if richer supplier
    # tenure/business-type data becomes available.
    # NOTE: price_distance removed — for price/max_price <= 2 (the vast
    # majority of rows) it equals abs(price_ratio - 1) exactly, a pure
    # deterministic transform of price_ratio. Keeping both caused the model
    # to split arbitrarily between two copies of the same signal, which is
    # why SHAP rank order for price features flipped between training runs.
]

LGBM_PARAMS = {
    "objective":        "lambdarank",
    "metric":           "ndcg",
    "ndcg_eval_at":     [5, 10],
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbosity":        -1,
    "seed":             42,
}


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def inject_noise(labels: pd.Series, noise_rate: float, max_label: int = 5,
                 random_state: int = 42) -> pd.Series:
    """
    Flip a fraction of training labels to a random different value.

    Args:
        labels:       clean integer relevance labels
        noise_rate:   fraction in [0, 1] of labels to corrupt
        max_label:    maximum label value (inclusive). Default 5 matches the
                       weak-label scale produced by feature_builder.py /
                       weak_label_generator.py (relevance in {0..5}).
        random_state: for reproducibility

    Returns:
        Noisy copy of labels (original series unchanged).
    """
    rng = np.random.default_rng(random_state)
    noisy = labels.copy()
    n = len(noisy)
    n_corrupt = int(n * noise_rate)
    if n_corrupt == 0:
        return noisy

    corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)
    for i in corrupt_idx:
        original = noisy.iloc[i]
        # Choose any label that is different from the original
        choices = [v for v in range(max_label + 1) if v != original]
        noisy.iloc[i] = int(rng.choice(choices))

    return noisy


# ---------------------------------------------------------------------------
# Train / evaluate helpers
# ---------------------------------------------------------------------------

def split_by_query(df: pd.DataFrame, test_frac: float = 0.2,
                   seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df[QUERY_COL]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def ndcg_at_k(y_true, y_pred, query_ids, k: int = 10) -> float:
    scores = []
    for qid in query_ids.unique():
        mask = query_ids == qid
        if mask.sum() < 2:
            continue
        t = y_true[mask].values.reshape(1, -1)
        p = y_pred[mask].reshape(1, -1)
        scores.append(ndcg_score(t, p, k=k))
    return float(np.mean(scores)) if scores else 0.0


def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """Train LambdaRank on (potentially noisy) train_df; evaluate on clean test_df."""
    train_groups = train_df.groupby(QUERY_COL, sort=False).size().values
    test_groups  = test_df.groupby(QUERY_COL, sort=False).size().values

    model = lgb.LGBMRanker(**LGBM_PARAMS, n_estimators=100)
    model.fit(
        train_df[FEATURE_COLS].values.astype(np.float32),
        train_df[LABEL_COL].values,
        group=train_groups,
        eval_set=[(test_df[FEATURE_COLS].values.astype(np.float32),
                   test_df[LABEL_COL].values)],
        eval_group=[test_groups],
        callbacks=[lgb.early_stopping(10, verbose=False)],
    )

    pred = model.predict(test_df[FEATURE_COLS].values.astype(np.float32))
    return ndcg_at_k(test_df[LABEL_COL], pred, test_df[QUERY_COL], k=10)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_label_noise_analysis(
    noise_rates: List[float] = None,
    n_trials: int = 3,
) -> Dict:
    """
    Run label noise robustness experiment.

    CRITICAL FIX:
        Noise is applied ONLY to train_df["relevance"].
        test_df["relevance"] stays clean throughout — this is the only way
        the NDCG degradation curve is scientifically valid.

    Returns dict with noise_rates and corresponding mean NDCG@10 values.
    """
    if noise_rates is None:
        noise_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA}")

    df = pd.read_csv(TRAIN_DATA)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df[LABEL_COL] = df[LABEL_COL].round().clip(0, 5).astype(int)

    # Clean train/test split (no noise yet)
    train_df, test_df = split_by_query(df)

    print("=" * 60)
    print("🔬 Label Noise Robustness Analysis")
    print(f"   Train: {len(train_df)} rows | Test (clean): {len(test_df)} rows")
    print("=" * 60)

    results = {"noise_rate": [], "ndcg_mean": [], "ndcg_std": []}

    for rate in noise_rates:
        trial_scores = []
        for trial in range(n_trials):
            # ✅ CORRECT: noise on TRAIN labels only
            noisy_train = train_df.copy()
            noisy_train[LABEL_COL] = inject_noise(
                train_df[LABEL_COL], noise_rate=rate, random_state=trial
            )
            # ❌ NEVER: test_df[LABEL_COL] = inject_noise(...)

            score = train_and_eval(noisy_train, test_df)
            trial_scores.append(score)

        mean_score = float(np.mean(trial_scores))
        std_score  = float(np.std(trial_scores))
        results["noise_rate"].append(rate)
        results["ndcg_mean"].append(mean_score)
        results["ndcg_std"].append(std_score)

        print(f"   Noise {rate*100:4.0f}%  →  NDCG@10 = {mean_score:.4f} ± {std_score:.4f}")

    # Plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        [r * 100 for r in results["noise_rate"]],
        results["ndcg_mean"],
        yerr=results["ndcg_std"],
        marker="o", capsize=4, linewidth=2,
    )
    ax.set_xlabel("Label Noise Rate (%)")
    ax.set_ylabel("NDCG@10 (clean test set)")
    ax.set_title("LambdaRank Robustness to Training Label Noise")
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(PLOTS_DIR, "label_noise_robustness.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Plot saved: {plot_path}")

    # Save CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "label_noise_results.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"✅ Results saved: {csv_path}")

    return results


if __name__ == "__main__":
    run_label_noise_analysis()