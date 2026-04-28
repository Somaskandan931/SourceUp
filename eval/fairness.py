"""
Fairness Analysis — SourceUp Supplier Ranking
----------------------------------------------
Measures whether suppliers from large metropolitan cities (Mumbai, Delhi,
Chennai, Bengaluru) systematically receive disproportionate exposure
compared to Tier-2/3 city suppliers with equivalent composite scores.

This is the fairness contribution described in Section 3.2B of the
SourceUp blueprint. Without this analysis, the system may amplify
existing geographic inequality in the Indian B2B supply chain.

Experiments:

  1. Exposure Disparity Analysis
     - Compute average exposure (rank position) for Metro vs Tier-2 suppliers
     - At matched score levels (suppliers binned by composite score)
     - Metric: Exposure Ratio = avg_rank(Metro) / avg_rank(Tier2)
       Ideal: ratio ≈ 1.0 (no geographic bias)

  2. Score Distribution Comparison
     - Box plots of composite scores by city tier
     - Tests whether score differences are intrinsic or bias-introduced

  3. Disparate Impact Ratio (DIR)
     - DIR = P(top-k | Tier2) / P(top-k | Metro)
     - IEEE fairness criterion: DIR ≥ 0.8 (EEOC 80% rule adapted)

  4. Counterfactual Fairness Test
     - Take a Tier-2 supplier and a Metro supplier with identical feature
       vectors except location. Measure score difference.
     - Ideal: Δscore ≈ 0 (location not driving score when other
       features are equal)

Outputs:
  data/eval/fairness_results.csv
  data/eval/plots/fairness_exposure_bar.png
  data/eval/plots/fairness_score_distribution.png
  data/eval/plots/fairness_dir_curve.png
  data/eval/plots/fairness_counterfactual.png

IEEE reference:
    Biega et al. (2018). Equity of Attention: Amortizing Individual
    Fairness in Rankings. SIGIR 2018.
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
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.metrics import ndcg_score
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — all resolved via config.cfg (no hardcoded paths)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import cfg

TRAIN_DATA = str(cfg.TRAINING_DATA)
CLEAN_DATA = str(cfg.CLEAN_DATA)
LGBM_PATH  = str(cfg.LGBM_MODEL)
OUT_DIR    = str(cfg.EVAL_DIR)
PLOTS_DIR  = str(cfg.EVAL_PLOTS_DIR)

cfg.ensure_dirs()

# ---------------------------------------------------------------------------
# Geographic classification
# ---------------------------------------------------------------------------
METRO_CITIES = {
    "mumbai", "delhi", "new delhi", "chennai", "bangalore", "bengaluru",
    "hyderabad", "kolkata", "pune", "ahmedabad"
}

TIER2_CITIES = {
    "noida", "thane", "surat", "vadodara", "nagpur", "jaipur", "lucknow",
    "kanpur", "visakhapatnam", "indore", "bhopal", "patna", "coimbatore",
    "jalandhar", "agra", "ludhiana", "nashik", "meerut", "kochi",
    "bahadurgarh", "rajkot", "amritsar", "ranchi", "chandigarh"
}

FEATURE_COLS = [
    "price_match", "price_ratio", "price_distance",
    "location_match", "cert_match", "years_normalized",
    "is_manufacturer", "is_trading_company",
    "faiss_score", "faiss_rank",
]

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


def load_data(test_frac=0.2, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training data and split, preserving location column."""
    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA}")

    df = pd.read_csv(TRAIN_DATA)
    df["relevance"] = df["relevance"].round().clip(0, 5).astype(int)

    # Attempt to merge location from clean data if not present
    if "location" not in df.columns and os.path.exists(CLEAN_DATA):
        df_clean = pd.read_csv(CLEAN_DATA)
        if "location" in df_clean.columns:
            # Join on index (training data is aligned with clean data)
            df["location"] = df_clean["location"].values[:len(df)]
            print("   ✅ Location column merged from suppliers_clean.csv")
        else:
            # Synthesize from location_match feature for demo purposes
            print("   ⚠️  Location column not found — synthesizing from location_match")
            df["location"] = _synthesize_location(df)
    elif "location" not in df.columns:
        print("   ⚠️  No location data — synthesizing from location_match")
        df["location"] = _synthesize_location(df)

    # Classify city tier
    df["city_tier"] = df["location"].apply(classify_tier)

    queries  = df["query_id"].unique()
    rng      = np.random.default_rng(seed)
    rng.shuffle(queries)
    split    = int(len(queries) * (1 - test_frac))
    test_qs  = set(queries[split:])
    df_test  = df[df["query_id"].isin(test_qs)].reset_index(drop=True)

    print(f"✅ Test set: {len(df_test)} rows, {df_test['query_id'].nunique()} queries")
    tier_counts = df_test["city_tier"].value_counts()
    print(f"   City tiers:\n{tier_counts.to_string()}")
    return df_test


def _synthesize_location(df: pd.DataFrame) -> pd.Series:
    """
    Synthesize location strings from location_match feature.
    High location_match → Metro; Low → Tier-2; Neutral → Other.
    Used only when real location data is unavailable.
    """
    cities_metro = list(METRO_CITIES)
    cities_tier2 = list(TIER2_CITIES)
    rng = np.random.default_rng(77)

    locs = []
    for v in df["location_match"].values:
        if v >= 0.8:
            locs.append(rng.choice(cities_metro))
        elif v <= 0.2:
            locs.append(rng.choice(cities_tier2))
        else:
            locs.append("India")  # neutral / unknown
    return pd.Series(locs, index=df.index)


def classify_tier(location: str) -> str:
    """Classify a location string into Metro, Tier-2, or Other."""
    if not isinstance(location, str):
        return "Other"
    loc_l = location.lower()
    if any(city in loc_l for city in METRO_CITIES):
        return "Metro"
    if any(city in loc_l for city in TIER2_CITIES):
        return "Tier-2"
    return "Other"


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[FEATURE_COLS].values.astype(np.float32))


# ============================================================================
# EXPERIMENT 1: EXPOSURE DISPARITY
# ============================================================================

def compute_exposure(pred: np.ndarray, query_ids: pd.Series) -> np.ndarray:
    """
    Compute normalised exposure for each supplier = 1 / log2(rank + 1).
    Rank 1 = highest predicted score.
    """
    exposure = np.zeros(len(pred))
    for qid in query_ids.unique():
        m    = (query_ids == qid).values
        pv   = pred[m]
        rank = np.argsort(np.argsort(-pv)) + 1  # 1-indexed ranks
        exposure[m] = 1.0 / np.log2(rank + 1)
    return exposure


def run_exposure_disparity(model, df_test: pd.DataFrame) -> Dict:
    """
    Compare average exposure between Metro and Tier-2 suppliers.
    """
    print("\n── Experiment 1: Exposure Disparity ────────────────────────")
    pred     = predict(model, df_test)
    exposure = compute_exposure(pred, df_test["query_id"])
    df_test  = df_test.copy()
    df_test["pred_score"] = pred
    df_test["exposure"]   = exposure

    results = {}
    for tier in ["Metro", "Tier-2", "Other"]:
        mask  = df_test["city_tier"] == tier
        n     = mask.sum()
        if n == 0:
            continue
        avg_exp  = df_test.loc[mask, "exposure"].mean()
        avg_rank = (1.0 / df_test.loc[mask, "exposure"]).mean()  # approx
        avg_score= df_test.loc[mask, "pred_score"].mean()
        results[tier] = {
            "n":          n,
            "avg_exposure": round(avg_exp, 5),
            "avg_score":    round(avg_score, 5),
        }
        print(f"  {tier:8s}  n={n:4d}  avg_exposure={avg_exp:.5f}  "
              f"avg_score={avg_score:.4f}")

    # Exposure ratio (Tier-2 / Metro)
    if "Metro" in results and "Tier-2" in results:
        ratio = results["Tier-2"]["avg_exposure"] / max(results["Metro"]["avg_exposure"], 1e-9)
        results["exposure_ratio_tier2_vs_metro"] = round(ratio, 4)
        bias_dir = "⚠️  Metro-favoured bias" if ratio < 0.8 else "✅ Fair"
        print(f"\n  Exposure ratio (Tier-2 / Metro) = {ratio:.4f}  {bias_dir}")
        print("  [IEEE fairness: ratio ≥ 0.8 = acceptable]")

    return results, df_test


# ============================================================================
# EXPERIMENT 2: SCORE DISTRIBUTION
# ============================================================================

def run_score_distribution_test(model, df_test: pd.DataFrame) -> Dict:
    """
    Mann-Whitney U test: are Metro and Tier-2 score distributions different?
    If p < 0.05, geographic location is influencing scores.
    """
    print("\n── Experiment 2: Score Distribution Test ───────────────────")
    pred = predict(model, df_test)
    df_test = df_test.copy()
    df_test["pred_score"] = pred

    metro_scores = df_test.loc[df_test["city_tier"] == "Metro",  "pred_score"].values
    tier2_scores = df_test.loc[df_test["city_tier"] == "Tier-2", "pred_score"].values

    if len(metro_scores) < 5 or len(tier2_scores) < 5:
        print("  ⚠️  Insufficient samples for statistical test")
        return {"status": "insufficient_samples"}

    stat_mw, p_mw   = mannwhitneyu(metro_scores, tier2_scores, alternative="two-sided")
    stat_ks, p_ks   = ks_2samp(metro_scores, tier2_scores)
    mean_diff       = metro_scores.mean() - tier2_scores.mean()

    print(f"  Metro  — mean={metro_scores.mean():.4f}  std={metro_scores.std():.4f}  n={len(metro_scores)}")
    print(f"  Tier-2 — mean={tier2_scores.mean():.4f}  std={tier2_scores.std():.4f}  n={len(tier2_scores)}")
    print(f"  Mean difference (Metro - Tier-2) = {mean_diff:+.4f}")
    print(f"  Mann-Whitney U: stat={stat_mw:.1f}  p={p_mw:.4f}")
    print(f"  KS test:        stat={stat_ks:.4f}  p={p_ks:.4f}")

    if p_mw < 0.05:
        print("  ⚠️  Statistically significant score difference (p < 0.05)")
    else:
        print("  ✅ No statistically significant score difference")

    return {
        "mean_metro":     round(float(metro_scores.mean()), 4),
        "mean_tier2":     round(float(tier2_scores.mean()), 4),
        "mean_diff":      round(float(mean_diff), 4),
        "mw_stat":        round(float(stat_mw), 2),
        "mw_p":           round(float(p_mw), 4),
        "ks_stat":        round(float(stat_ks), 4),
        "ks_p":           round(float(p_ks), 4),
        "significant":    p_mw < 0.05,
        "df_with_pred":   df_test,
    }


# ============================================================================
# EXPERIMENT 3: DISPARATE IMPACT RATIO (DIR)
# ============================================================================

def run_disparate_impact(model, df_test: pd.DataFrame,
                          k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    DIR = P(top-k | Tier-2) / P(top-k | Metro)
    Measures what fraction of each group appears in top-k results.
    IEEE/EEOC criterion: DIR ≥ 0.8.
    """
    print("\n── Experiment 3: Disparate Impact Ratio ───────────────────")
    pred   = predict(model, df_test)
    df_t   = df_test.copy()
    df_t["pred_score"] = pred

    rows = []
    for k in k_values:
        metro_topk  = 0
        metro_total = 0
        tier2_topk  = 0
        tier2_total = 0

        for qid in df_t["query_id"].unique():
            m    = (df_t["query_id"] == qid).values
            sub  = df_t[m].reset_index(drop=True)
            top_k_mask = np.argsort(-sub["pred_score"].values)[:k]

            metro_mask = (sub["city_tier"] == "Metro").values
            tier2_mask = (sub["city_tier"] == "Tier-2").values

            metro_total += metro_mask.sum()
            tier2_total += tier2_mask.sum()
            metro_topk  += metro_mask[top_k_mask].sum()
            tier2_topk  += tier2_mask[top_k_mask].sum()

        p_metro = metro_topk / max(metro_total, 1)
        p_tier2 = tier2_topk / max(tier2_total, 1)
        dir_val = p_tier2 / max(p_metro, 1e-9)

        rows.append({
            "k":            k,
            "P(top-k|Metro)":   round(p_metro, 4),
            "P(top-k|Tier-2)":  round(p_tier2, 4),
            "DIR":              round(dir_val, 4),
            "Fair (≥0.8)":      "✅" if dir_val >= 0.8 else "⚠️",
        })
        print(f"  k={k:2d}  P_Metro={p_metro:.4f}  P_Tier2={p_tier2:.4f}  "
              f"DIR={dir_val:.4f}  {'✅ Fair' if dir_val >= 0.8 else '⚠️  Biased'}")

    df_out = pd.DataFrame(rows)
    return df_out


# ============================================================================
# EXPERIMENT 4: COUNTERFACTUAL FAIRNESS
# ============================================================================

def run_counterfactual_fairness(model, df_test: pd.DataFrame,
                                  n_pairs: int = 100) -> Dict:
    """
    For each Metro supplier, create a counterfactual Tier-2 twin by
    swapping location_match (1.0 → 0.5) while keeping all other features
    identical. Measure the resulting score difference.

    Ideal: Δscore ≈ 0 (location doesn't drive the score independently).
    """
    print("\n── Experiment 4: Counterfactual Fairness ───────────────────")
    metro_df = df_test[df_test["city_tier"] == "Metro"].reset_index(drop=True)

    if len(metro_df) == 0:
        print("  ⚠️  No Metro suppliers in test set — skipping")
        return {}

    sample_size = min(n_pairs, len(metro_df))
    sample      = metro_df.sample(sample_size, random_state=42)

    # Original scores for Metro suppliers
    orig_pred = model.predict(sample[FEATURE_COLS].values.astype(np.float32))

    # Counterfactual: set location_match to 0.5 (neutral)
    df_cf = sample.copy()
    df_cf["location_match"] = 0.5
    cf_pred = model.predict(df_cf[FEATURE_COLS].values.astype(np.float32))

    delta = orig_pred - cf_pred   # positive = Metro location boosts score

    print(f"  Counterfactual pairs: {sample_size}")
    print(f"  Mean Δscore (Metro location advantage): {delta.mean():+.5f}")
    print(f"  Std Δscore:                             {delta.std():.5f}")
    print(f"  Max Δscore:                             {delta.max():+.5f}")

    interpretation = (
        "✅ Location has minimal impact on score" if abs(delta.mean()) < 0.02
        else "⚠️  Location significantly inflates Metro scores"
    )
    print(f"  {interpretation}")

    return {
        "mean_delta":    round(float(delta.mean()), 6),
        "std_delta":     round(float(delta.std()), 6),
        "max_delta":     round(float(delta.max()), 6),
        "n_pairs":       sample_size,
        "fair":          abs(delta.mean()) < 0.02,
        "deltas":        delta,
        "orig_pred":     orig_pred,
        "cf_pred":       cf_pred,
    }


# ============================================================================
# PLOTS
# ============================================================================

def plot_exposure_bar(results_exp: Dict):
    """Bar chart: average exposure by city tier."""
    tiers  = [t for t in ["Metro", "Tier-2", "Other"] if t in results_exp]
    exps   = [results_exp[t]["avg_exposure"] for t in tiers]
    scores = [results_exp[t]["avg_score"] for t in tiers]
    counts = [results_exp[t]["n"] for t in tiers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2166ac", "#d73027", "#aaaaaa"][:len(tiers)]

    # Exposure
    bars = ax1.bar(tiers, exps, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    for bar, val, n in zip(bars, exps, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.0005,
                 f"{val:.4f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax1.axhline(exps[0] * 0.8, color="gray", linestyle="--", linewidth=1.2,
                label="80% of Metro exposure (fairness threshold)")
    ax1.set_ylabel("Average Exposure (1/log₂(rank+1))", fontsize=10)
    ax1.set_title("(a) Average Exposure by City Tier", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Scores
    bars2 = ax2.bar(tiers, scores, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars2, scores):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Average Predicted Score", fontsize=10)
    ax2.set_title("(b) Average Score by City Tier", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Geographic Fairness — Exposure and Score Disparity by City Tier\n"
        "(Tier-2 / Metro exposure ratio ≥ 0.8 = IEEE fairness criterion)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = f"{PLOTS_DIR}/fairness_exposure_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_score_distributions(df_test_pred: pd.DataFrame):
    """Violin + box plot of score distributions by tier."""
    fig, ax = plt.subplots(figsize=(10, 5))
    order   = ["Metro", "Tier-2", "Other"]
    palette = {"Metro": "#2166ac", "Tier-2": "#d73027", "Other": "#aaaaaa"}

    available = [t for t in order if t in df_test_pred["city_tier"].values]
    df_plot   = df_test_pred[df_test_pred["city_tier"].isin(available)]

    sns.violinplot(
        data=df_plot, x="city_tier", y="pred_score",
        order=available, palette=palette,
        inner="box", linewidth=1.2, ax=ax
    )
    ax.set_xlabel("City Tier", fontsize=11)
    ax.set_ylabel("Predicted Ranking Score", fontsize=11)
    ax.set_title(
        "Score Distribution by Geographic Tier\n"
        "(Overlapping distributions = no systematic geographic bias)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/fairness_score_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_dir_curve(df_dir: pd.DataFrame):
    """Plot Disparate Impact Ratio vs k."""
    if df_dir.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(df_dir["k"], df_dir["DIR"],
            "o-", color="#2166ac", linewidth=2.2, markersize=8, label="DIR (Tier-2 / Metro)")
    ax.axhline(0.8, color="#d73027", linestyle="--", linewidth=1.5,
               label="DIR = 0.8 (EEOC 80% Rule)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="DIR = 1.0 (perfect parity)")
    ax.fill_between(df_dir["k"], 0.8, 1.2, alpha=0.07, color="#2166ac", label="Fair zone")

    for _, row in df_dir.iterrows():
        ax.annotate(f"{row['DIR']:.3f}", (row["k"], row["DIR"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8.5)

    ax.set_xlabel("Top-k cutoff", fontsize=11)
    ax.set_ylabel("Disparate Impact Ratio (DIR)", fontsize=11)
    ax.set_ylim(0, 1.5)
    ax.set_title(
        "Disparate Impact Ratio — Geographic Fairness @ Top-k\n"
        "(DIR = Tier-2 top-k rate / Metro top-k rate; ≥ 0.8 = fair)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/fairness_dir_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_counterfactual(cf_results: Dict):
    """Distribution of counterfactual score deltas."""
    if not cf_results or "deltas" not in cf_results:
        return
    deltas = cf_results["deltas"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(deltas, bins=30, color="#2166ac", edgecolor="black",
            linewidth=0.5, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Δ = 0 (no bias)")
    ax.axvline(deltas.mean(), color="#d73027", linewidth=2,
               label=f"Mean Δ = {deltas.mean():+.5f}")

    ax.set_xlabel("Δscore (Metro location score − Neutral location score)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Counterfactual Fairness: Score Change When Removing Metro Location Signal\n"
        "(Histogram centred at 0 = location not driving scores)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/fairness_counterfactual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def run_fairness_analysis():
    print("=" * 65)
    print("⚖️  SourceUp — Geographic Fairness Analysis")
    print("=" * 65)

    model   = load_model()
    df_test = load_data(test_frac=0.2)

    # Experiment 1: Exposure disparity
    results_exp, df_with_exp = run_exposure_disparity(model, df_test)

    # Experiment 2: Score distribution
    score_res = run_score_distribution_test(model, df_test)
    df_with_pred = score_res.pop("df_with_pred", df_test.copy())

    # Experiment 3: Disparate Impact Ratio
    df_dir = run_disparate_impact(model, df_test, k_values=[5, 10, 20])

    # Experiment 4: Counterfactual
    cf_results = run_counterfactual_fairness(model, df_test, n_pairs=100)

    # Save summary CSV
    summary = {
        "exposure_ratio_tier2_vs_metro": results_exp.get("exposure_ratio_tier2_vs_metro", "N/A"),
        "score_mean_metro":              score_res.get("mean_metro", "N/A"),
        "score_mean_tier2":              score_res.get("mean_tier2", "N/A"),
        "score_diff_mw_p":               score_res.get("mw_p", "N/A"),
        "dir_at_k10":                    df_dir.loc[df_dir["k"] == 10, "DIR"].values[0]
                                         if not df_dir.empty else "N/A",
        "counterfactual_mean_delta":     cf_results.get("mean_delta", "N/A"),
        "counterfactual_fair":           cf_results.get("fair", "N/A"),
    }
    pd.DataFrame([summary]).to_csv(f"{OUT_DIR}/fairness_results.csv", index=False)
    print(f"\n✅ Summary saved: {OUT_DIR}/fairness_results.csv")

    # Verdict
    print("\n" + "=" * 65)
    print("FAIRNESS VERDICT")
    print("=" * 65)
    exp_ratio = results_exp.get("exposure_ratio_tier2_vs_metro", 0)
    dir_10    = float(df_dir.loc[df_dir["k"] == 10, "DIR"].values[0]) \
                if not df_dir.empty else 0
    cf_fair   = cf_results.get("fair", False)

    print(f"  Exposure ratio (Tier-2 / Metro):     {exp_ratio:.4f}  "
          f"{'✅' if exp_ratio >= 0.8 else '⚠️'}")
    print(f"  DIR @ k=10:                          {dir_10:.4f}  "
          f"{'✅' if dir_10 >= 0.8 else '⚠️'}")
    print(f"  Counterfactual location independence: "
          f"{'✅ Yes' if cf_fair else '⚠️ No'}")

    # Plots
    print("\n📊 Generating plots...")
    plot_exposure_bar(results_exp)
    plot_score_distributions(df_with_pred)
    plot_dir_curve(df_dir)
    plot_counterfactual(cf_results)

    print(f"\n✅ All fairness outputs saved in: {OUT_DIR}")
    return results_exp, df_dir, cf_results


if __name__ == "__main__":
    run_fairness_analysis()