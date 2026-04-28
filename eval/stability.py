"""
Rank Stability Analysis — SourceUp Supplier Ranking
----------------------------------------------------
Measures whether the ranker produces consistent rankings under
minor input perturbations — a key robustness requirement for
IEEE publication and production deployment.

Two experiments:

  1. Score Perturbation Stability
     For each test query, inject Gaussian noise (σ ∈ {0.01, 0.03, 0.05})
     into each numeric feature and re-rank. Measure Kendall's Tau between
     the original ranking and the perturbed ranking.
     IEEE criterion: τ ≥ 0.85 at σ=0.03 indicates a stable ranker.

  2. Query Paraphrase Stability
     Simulate semantic query variation by adding small Gaussian noise to
     the faiss_score and faiss_rank columns (SBERT retrieval noise).
     Measures sensitivity of ranking to embedding-level perturbations.

Outputs:
  data/eval/stability_results.csv
  data/eval/plots/stability_tau_boxplot.png
  data/eval/plots/stability_noise_curve.png
  data/eval/plots/stability_heatmap.png

IEEE reference:
    Sculley (2009). Large Scale Learning to Rank. NIPS Workshop on
    Advances in Ranking. (Rank stability as evaluation criterion.)
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — all resolved via config.cfg (no hardcoded paths)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import cfg

TRAIN_DATA = str(cfg.TRAINING_DATA)
LGBM_PATH  = str(cfg.LGBM_MODEL)
OUT_DIR    = str(cfg.EVAL_DIR)
PLOTS_DIR  = str(cfg.EVAL_PLOTS_DIR)

cfg.ensure_dirs()

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "price_match", "price_ratio", "price_distance",
    "location_match", "cert_match", "years_normalized",
    "is_manufacturer", "is_trading_company",
    "faiss_score", "faiss_rank",
]

# Features that receive perturbation (continuous only — not binary flags)
CONTINUOUS_FEATURES = [
    "price_ratio", "price_distance",
    "years_normalized",
    "faiss_score", "faiss_rank",
]

# Features representing SBERT/retrieval noise only
RETRIEVAL_FEATURES = ["faiss_score", "faiss_rank"]

# Noise levels to sweep (as fraction of feature std)
NOISE_SIGMAS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
N_REPEATS    = 10   # trials per noise level (averaged for stability)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


# ============================================================================
# DATA & MODEL
# ============================================================================

def load_model():
    if not os.path.exists(LGBM_PATH):
        raise FileNotFoundError(
            f"LightGBM model not found: {LGBM_PATH}\n"
            "Run: python train_lambdarank.py"
        )
    with open(LGBM_PATH, "rb") as f:
        return pickle.load(f)


def load_test_data(test_frac: float = 0.2, seed: int = 42) -> pd.DataFrame:
    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA}")
    df      = pd.read_csv(TRAIN_DATA)
    queries = df["query_id"].unique()
    rng     = np.random.default_rng(seed)
    rng.shuffle(queries)
    split   = int(len(queries) * (1 - test_frac))
    test_qs = set(queries[split:])
    df_test = df[df["query_id"].isin(test_qs)].reset_index(drop=True)
    df_test["relevance"] = df_test["relevance"].round().clip(0, 5).astype(int)
    print(f"✅ Test set: {len(df_test)} rows, {df_test['query_id'].nunique()} queries")
    return df_test


def rule_based_score(df: pd.DataFrame) -> np.ndarray:
    """Rule-based fallback scorer (same as ranker.py)."""
    return (
        df["price_match"]          * 0.35 +
        (1 - df["price_distance"]) * 0.10 +
        df["location_match"]       * 0.20 +
        df["cert_match"]           * 0.20 +
        df["years_normalized"]     * 0.05 +
        df["is_manufacturer"]      * 0.05 +
        df["faiss_score"]          * 0.05
    ).values


def predict(model, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].values.astype(np.float32)
    return model.predict(X)


# ============================================================================
# PERTURBATION UTILITIES
# ============================================================================

def perturb_features(df: pd.DataFrame,
                      feature_names: List[str],
                      sigma: float,
                      seed: int = None) -> pd.DataFrame:
    """
    Add Gaussian noise N(0, σ) to specified continuous features.
    Noise is scaled by each feature's standard deviation.
    Result is clipped to [0, 1] for normalised features.
    """
    df_p = df.copy()
    rng  = np.random.default_rng(seed)

    for col in feature_names:
        if col not in df_p.columns:
            continue
        std  = df_p[col].std() if df_p[col].std() > 0 else 1.0
        noise = rng.normal(0, sigma * std, size=len(df_p))
        df_p[col] = (df_p[col] + noise).clip(0.0, 1.0)

    return df_p


def ranking_kendall_tau(pred_orig: np.ndarray,
                         pred_pert: np.ndarray,
                         query_ids: pd.Series) -> List[float]:
    """
    Compute per-query Kendall's Tau between original and perturbed rankings.
    Returns a list of tau values (one per query).
    """
    taus = []
    for qid in query_ids.unique():
        m = (query_ids == qid).values
        if m.sum() < 3:
            continue
        tau, _ = kendalltau(pred_orig[m], pred_pert[m])
        if not np.isnan(tau):
            taus.append(float(tau))
    return taus


def ndcg_delta(y_true: pd.Series,
               pred_orig: np.ndarray,
               pred_pert: np.ndarray,
               query_ids: pd.Series,
               k: int = 10) -> List[float]:
    """
    Compute per-query NDCG change between original and perturbed predictions.
    Returns list of (ndcg_original - ndcg_perturbed) per query.
    """
    deltas = []
    for qid in query_ids.unique():
        m = (query_ids == qid).values
        if m.sum() < 2:
            continue
        t = y_true[m].values.reshape(1, -1)
        n_orig = ndcg_score(t, pred_orig[m].reshape(1, -1), k=k)
        n_pert = ndcg_score(t, pred_pert[m].reshape(1, -1), k=k)
        deltas.append(n_orig - n_pert)
    return deltas


# ============================================================================
# EXPERIMENT 1: SCORE PERTURBATION STABILITY
# ============================================================================

def run_score_perturbation_stability(model, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Sweep noise σ over continuous features.
    At each level, run N_REPEATS trials and average Kendall's Tau.
    """
    print("\n── Experiment 1: Score Perturbation Stability ─────────────")
    pred_orig = predict(model, df_test)
    y         = df_test["relevance"]
    qids      = df_test["query_id"]

    rows = []
    for sigma in NOISE_SIGMAS:
        trial_taus   = []
        trial_ndcg_d = []

        for trial in range(N_REPEATS):
            df_p     = perturb_features(df_test, CONTINUOUS_FEATURES, sigma,
                                         seed=trial * 100 + int(sigma * 1000))
            pred_p   = predict(model, df_p)
            taus     = ranking_kendall_tau(pred_orig, pred_p, qids)
            deltas   = ndcg_delta(y, pred_orig, pred_p, qids)
            trial_taus.extend(taus)
            trial_ndcg_d.extend(deltas)

        rows.append({
            "sigma":            sigma,
            "mean_tau":         round(float(np.mean(trial_taus)),  4),
            "std_tau":          round(float(np.std(trial_taus)),   4),
            "median_tau":       round(float(np.median(trial_taus)),4),
            "pct_tau_above_85": round(float(np.mean(np.array(trial_taus) >= 0.85)), 4),
            "mean_ndcg_delta":  round(float(np.mean(trial_ndcg_d)), 4),
            "n_queries":        len(trial_taus) // N_REPEATS,
        })
        print(f"  σ={sigma:.3f}  τ̄={rows[-1]['mean_tau']:.4f}  "
              f"(±{rows[-1]['std_tau']:.4f})  "
              f"τ≥0.85: {rows[-1]['pct_tau_above_85']*100:.0f}%  "
              f"ΔNDCG: {rows[-1]['mean_ndcg_delta']:+.4f}")

    df_out = pd.DataFrame(rows)
    path   = f"{OUT_DIR}/stability_score_perturbation.csv"
    df_out.to_csv(path, index=False)
    print(f"  ✅ Saved: {path}")
    return df_out


# ============================================================================
# EXPERIMENT 2: RETRIEVAL NOISE STABILITY (SBERT/FAISS)
# ============================================================================

def run_retrieval_noise_stability(model, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Perturb only the SBERT/FAISS retrieval features.
    Isolates sensitivity to embedding-level noise.
    """
    print("\n── Experiment 2: Retrieval (SBERT/FAISS) Noise Stability ──")
    pred_orig = predict(model, df_test)
    qids      = df_test["query_id"]

    rows = []
    for sigma in NOISE_SIGMAS:
        trial_taus = []
        for trial in range(N_REPEATS):
            df_p   = perturb_features(df_test, RETRIEVAL_FEATURES, sigma,
                                       seed=trial * 200 + int(sigma * 1000))
            pred_p = predict(model, df_p)
            taus   = ranking_kendall_tau(pred_orig, pred_p, qids)
            trial_taus.extend(taus)

        rows.append({
            "sigma":            sigma,
            "mean_tau":         round(float(np.mean(trial_taus)),  4),
            "std_tau":          round(float(np.std(trial_taus)),   4),
            "pct_tau_above_85": round(float(np.mean(np.array(trial_taus) >= 0.85)), 4),
        })
        print(f"  σ={sigma:.3f}  τ̄={rows[-1]['mean_tau']:.4f}  "
              f"τ≥0.85: {rows[-1]['pct_tau_above_85']*100:.0f}%")

    df_out = pd.DataFrame(rows)
    path   = f"{OUT_DIR}/stability_retrieval_noise.csv"
    df_out.to_csv(path, index=False)
    print(f"  ✅ Saved: {path}")
    return df_out


# ============================================================================
# EXPERIMENT 3: PER-QUERY TAU DISTRIBUTION (σ = 0.03)
# ============================================================================

def run_per_query_tau(model, df_test: pd.DataFrame,
                       sigma: float = 0.03) -> Dict:
    """
    Run N_REPEATS perturbations at σ=0.03 and collect per-query τ distributions.
    Used for the box plot.
    """
    print(f"\n── Experiment 3: Per-Query τ Distribution (σ={sigma}) ───")
    pred_orig = predict(model, df_test)
    qids      = df_test["query_id"]

    # Collect τ per query across all trials
    query_tau_map = {qid: [] for qid in qids.unique()}

    for trial in range(N_REPEATS * 3):  # more trials for smoother distribution
        df_p   = perturb_features(df_test, CONTINUOUS_FEATURES, sigma, seed=trial * 7)
        pred_p = predict(model, df_p)

        for qid in qids.unique():
            m = (qids == qid).values
            if m.sum() < 3:
                continue
            tau, _ = kendalltau(pred_orig[m], pred_p[m])
            if not np.isnan(tau):
                query_tau_map[qid].append(float(tau))

    # Global
    all_taus = [t for taus in query_tau_map.values() for t in taus]
    print(f"  Global τ̄ = {np.mean(all_taus):.4f}  "
          f"median = {np.median(all_taus):.4f}  "
          f"σ = {np.std(all_taus):.4f}")
    return query_tau_map


# ============================================================================
# PLOTS
# ============================================================================

def plot_stability_noise_curve(df_score: pd.DataFrame, df_retrieval: pd.DataFrame):
    """
    Fig: Mean Kendall's τ vs noise level σ for both perturbation types.
    The primary stability result for the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        df_score["sigma"], df_score["mean_tau"],
        yerr=df_score["std_tau"],
        fmt="o-", color="#2166ac", linewidth=2, markersize=7,
        capsize=4, label="Score Perturbation (all features)"
    )
    ax.errorbar(
        df_retrieval["sigma"], df_retrieval["mean_tau"],
        yerr=df_retrieval["std_tau"],
        fmt="s--", color="#d73027", linewidth=2, markersize=7,
        capsize=4, label="Retrieval Noise (SBERT/FAISS only)"
    )

    # IEEE stability threshold
    ax.axhline(0.85, color="gray", linestyle=":", linewidth=1.5,
               label="τ = 0.85 (stability threshold)")
    ax.fill_between(df_score["sigma"], 0.85, 1.0, alpha=0.06, color="#2166ac",
                    label="Stable region")

    ax.set_xlabel("Noise Level σ (fraction of feature std)", fontsize=11)
    ax.set_ylabel("Mean Kendall's Tau (τ)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Rank Stability Under Input Perturbations\n"
        "(τ closer to 1.0 = more stable ranking; "
        "τ ≥ 0.85 at σ=0.03 = IEEE stability criterion)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/stability_noise_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_stability_tau_boxplot(query_tau_map: Dict):
    """
    Box plot of per-query τ distribution at σ = 0.03.
    Shows consistency across different query types.
    """
    # Sample up to 15 queries for readable plot
    qids    = [q for q, taus in query_tau_map.items() if len(taus) >= 3]
    sampled = qids[:15] if len(qids) > 15 else qids

    data    = [query_tau_map[q] for q in sampled]
    labels  = [f"Q{i+1}" for i in range(len(sampled))]

    fig, ax = plt.subplots(figsize=(max(10, len(sampled) * 0.8), 5))
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.4},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#2166ac")
        patch.set_alpha(0.6)

    ax.axhline(0.85, color="#d73027", linestyle="--", linewidth=1.5,
               label="τ = 0.85 threshold")
    ax.set_xlabel("Query (anonymised)", fontsize=11)
    ax.set_ylabel("Kendall's Tau (σ = 0.03)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Per-Query Rank Stability Distribution (σ = 0.03)\n"
        "(Higher median τ and tighter box = more stable per query)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/stability_tau_boxplot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_stability_heatmap(model, df_test: pd.DataFrame):
    """
    Heatmap: τ per (feature_group, sigma) combination.
    Shows which features are the largest sources of instability.
    """
    print("  Plotting: Stability heatmap by feature group...")
    feature_groups = {
        "Price":         ["price_ratio", "price_distance"],
        "Location/Cert": ["location_match", "cert_match"],
        "Supplier Age":  ["years_normalized"],
        "Retrieval":     ["faiss_score", "faiss_rank"],
        "All Features":  CONTINUOUS_FEATURES,
    }

    pred_orig = predict(model, df_test)
    qids      = df_test["query_id"]
    heatmap   = {}

    for group_name, feats in feature_groups.items():
        row = {}
        for sigma in NOISE_SIGMAS:
            trial_taus = []
            for trial in range(N_REPEATS):
                df_p   = perturb_features(df_test, feats, sigma, seed=trial * 31)
                pred_p = predict(model, df_p)
                taus   = ranking_kendall_tau(pred_orig, pred_p, qids)
                trial_taus.extend(taus)
            row[f"σ={sigma:.2f}"] = round(float(np.mean(trial_taus)), 4)
        heatmap[group_name] = row

    df_hmap = pd.DataFrame(heatmap).T

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        df_hmap, ax=ax, annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=0.7, vmax=1.0,
        linewidths=0.5, cbar_kws={"label": "Mean Kendall's τ"},
        annot_kws={"fontsize": 9}
    )
    ax.set_xlabel("Noise Level (σ)", fontsize=11)
    ax.set_ylabel("Feature Group", fontsize=11)
    ax.set_title(
        "Stability Heatmap — Mean Kendall's τ by Feature Group and Noise Level\n"
        "(Green = stable, Red = sensitive to perturbation)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = f"{PLOTS_DIR}/stability_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")
    return df_hmap


# ============================================================================
# MAIN
# ============================================================================

def run_stability():
    print("=" * 65)
    print("📐 SourceUp — Rank Stability Analysis")
    print("=" * 65)

    model   = load_model()
    df_test = load_test_data(test_frac=0.2)

    # Run experiments
    df_score      = run_score_perturbation_stability(model, df_test)
    df_retrieval  = run_retrieval_noise_stability(model, df_test)
    query_tau_map = run_per_query_tau(model, df_test, sigma=0.03)

    # Combined summary CSV
    df_score["type"]     = "All Features"
    df_retrieval["type"] = "Retrieval Only"
    df_combined = pd.concat([df_score, df_retrieval], ignore_index=True)
    df_combined.to_csv(f"{OUT_DIR}/stability_results.csv", index=False)
    print(f"\n✅ Combined results saved: {OUT_DIR}/stability_results.csv")

    # Verdict
    tau_at_3pct = df_score.loc[df_score["sigma"] == 0.03, "mean_tau"]
    if len(tau_at_3pct):
        tau_val = tau_at_3pct.values[0]
        verdict = "✅ STABLE (τ ≥ 0.85)" if tau_val >= 0.85 else "⚠️ MARGINAL"
        print(f"\n  Stability verdict at σ=0.03: τ={tau_val:.4f}  {verdict}")

    # Plots
    print("\n📊 Generating plots...")
    plot_stability_noise_curve(df_score, df_retrieval)
    plot_stability_tau_boxplot(query_tau_map)
    plot_stability_heatmap(model, df_test)

    print(f"\n✅ All stability outputs saved in: {OUT_DIR}")
    return df_score, df_retrieval, query_tau_map


if __name__ == "__main__":
    run_stability()