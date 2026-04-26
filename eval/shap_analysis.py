"""
SHAP Feature Attribution Analysis — SourceUp Supplier Ranking
--------------------------------------------------------------
Uses the SHAP library to generate scientifically validated feature
attributions from the trained LightGBM LambdaRank model.

This converts decision traces from "heuristic reasons" into
model-grounded explanations suitable for IEEE publication.

Outputs:
  1. Global feature importance (summary plot, bar plot)
  2. Per-query SHAP beeswarm showing each supplier's score composition
  3. Dependence plots for the top 3 most important features
  4. Force plot for top-ranked supplier per example query
  5. SHAP value CSV for full reproducibility

Files saved:
  data/eval/shap_values.csv
  data/eval/plots/shap_summary_beeswarm.png
  data/eval/plots/shap_summary_bar.png
  data/eval/plots/shap_dependence_<feature>.png
  data/eval/plots/shap_force_top_supplier.png
  data/eval/plots/shap_heatmap.png

IEEE reference:
    Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions.
    NeurIPS 2017.
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

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.getenv("SOURCEUP_DIR", "C:/Users/somas/PycharmProjects/SourceUp")
TRAIN_DATA = f"{BASE_DIR}/data/training/ranking_data.csv"
LGBM_PATH  = f"{BASE_DIR}/backend/app/models/embeddings/ranker_lightgbm.pkl"
OUT_DIR    = f"{BASE_DIR}/data/eval"
PLOTS_DIR  = f"{OUT_DIR}/plots"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# SHAP import
# ---------------------------------------------------------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("❌  shap not installed.  Run: pip install shap")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "price_match",
    "price_ratio",
    "price_distance",
    "location_match",
    "cert_match",
    "years_normalized",
    "is_manufacturer",
    "is_trading_company",
    "faiss_score",
    "faiss_rank",
]

FEATURE_LABELS = {
    "price_match":        "Price Within Budget",
    "price_ratio":        "Price / Budget Ratio",
    "price_distance":     "Price Distance",
    "location_match":     "Location Match",
    "cert_match":         "Certification Match",
    "years_normalized":   "Years on Platform",
    "is_manufacturer":    "Is Manufacturer",
    "is_trading_company": "Is Trading Company",
    "faiss_score":        "Semantic Similarity (SBERT)",
    "faiss_rank":         "FAISS Retrieval Rank",
}

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
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_DATA}\n"
            "Run: python pipeline/feature_builder.py"
        )
    df     = pd.read_csv(TRAIN_DATA)
    queries = df["query_id"].unique()
    rng     = np.random.default_rng(seed)
    rng.shuffle(queries)
    split   = int(len(queries) * (1 - test_frac))
    test_qs = set(queries[split:])
    df_test = df[df["query_id"].isin(test_qs)].reset_index(drop=True)
    print(f"✅ Test set: {len(df_test)} rows, {df_test['query_id'].nunique()} queries")
    return df_test


# ============================================================================
# SHAP COMPUTATION
# ============================================================================

def compute_shap_values(model, df_test: pd.DataFrame):
    """
    Compute SHAP values using TreeExplainer (exact, fast for tree models).
    Returns shap_values array of shape (n_samples, n_features).
    """
    print("\n🔬 Computing SHAP values (TreeExplainer)...")
    X = df_test[FEATURE_COLS].values.astype(np.float32)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # LightGBM ranking models return a list for some versions
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    print(f"   SHAP values shape: {shap_values.shape}")
    return shap_values, explainer, X


def save_shap_csv(shap_values: np.ndarray, df_test: pd.DataFrame):
    """Save raw SHAP values for full reproducibility (auditable)."""
    df_shap = pd.DataFrame(shap_values, columns=FEATURE_COLS)
    df_shap.insert(0, "query_id", df_test["query_id"].values)

    if "relevance" in df_test.columns:
        df_shap.insert(1, "relevance", df_test["relevance"].values)

    path = f"{OUT_DIR}/shap_values.csv"
    df_shap.to_csv(path, index=False)
    print(f"  ✅ SHAP values saved: {path}")
    return df_shap


# ============================================================================
# PLOTS
# ============================================================================

def plot_shap_summary_beeswarm(shap_values: np.ndarray, X: np.ndarray):
    """
    Fig. 3a: Beeswarm summary plot.
    Shows the distribution of SHAP values for each feature,
    coloured by feature value magnitude.
    """
    print("  Plotting: SHAP beeswarm summary...")
    fig, ax = plt.subplots(figsize=(11, 7))

    shap.summary_plot(
        shap_values, X,
        feature_names=[FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
        show=False,
        max_display=10,
        plot_type="dot",
    )

    plt.title(
        "Fig. 3a — SHAP Beeswarm: Feature Contributions to Supplier Score\n"
        "(Red = high feature value, Blue = low; width = impact frequency)",
        fontsize=11, fontweight="bold", pad=14
    )
    plt.tight_layout()
    path = f"{PLOTS_DIR}/shap_summary_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_shap_summary_bar(shap_values: np.ndarray, X: np.ndarray):
    """
    Fig. 3b: Mean absolute SHAP bar chart.
    This is the publishable global feature importance figure.
    """
    print("  Plotting: SHAP mean-absolute bar chart...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_labels = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]

    df_imp = pd.DataFrame({
        "feature":    feat_labels,
        "mean_shap":  mean_abs,
    }).sort_values("mean_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.Blues(np.linspace(0.4, 0.9, len(df_imp)))
    bars    = ax.barh(df_imp["feature"], df_imp["mean_shap"],
                      color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, df_imp["mean_shap"]):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5)

    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Score)", fontsize=11)
    ax.set_title(
        "Fig. 3b — Global Feature Importance (SHAP)\n"
        "Higher = feature has more impact on supplier ranking",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/shap_summary_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_shap_dependence(shap_values: np.ndarray, X: np.ndarray, top_n: int = 3):
    """
    Dependence plots for the top-N most important features.
    Shows how SHAP value changes with feature value,
    coloured by the most interacting other feature.
    """
    print(f"  Plotting: Dependence plots for top {top_n} features...")
    mean_abs    = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    for rank, feat_idx in enumerate(top_indices):
        feat_name  = FEATURE_COLS[feat_idx]
        feat_label = FEATURE_LABELS.get(feat_name, feat_name)

        # Find most correlated interaction feature
        correlations  = [abs(np.corrcoef(shap_values[:, feat_idx], X[:, j])[0, 1])
                         for j in range(X.shape[1]) if j != feat_idx]
        interact_idx  = int(np.argmax(correlations))
        if interact_idx >= feat_idx:
            interact_idx += 1  # skip self
        interact_label = FEATURE_LABELS.get(FEATURE_COLS[interact_idx], FEATURE_COLS[interact_idx])

        fig, ax = plt.subplots(figsize=(9, 5))
        sc = ax.scatter(
            X[:, feat_idx],
            shap_values[:, feat_idx],
            c=X[:, interact_idx],
            cmap="RdBu",
            alpha=0.6, s=20, edgecolors="none"
        )
        plt.colorbar(sc, ax=ax, label=f"Colour: {interact_label}")

        ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_xlabel(feat_label, fontsize=11)
        ax.set_ylabel(f"SHAP Value for {feat_label}", fontsize=11)
        ax.set_title(
            f"SHAP Dependence Plot — {feat_label}\n"
            f"(Colour = {interact_label} interaction)",
            fontsize=11, fontweight="bold"
        )
        ax.grid(alpha=0.25)
        plt.tight_layout()
        safe_name = feat_name.replace("/", "_")
        path = f"{PLOTS_DIR}/shap_dependence_{safe_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved: {path}  (rank #{rank+1}: {feat_label})")


def plot_shap_heatmap(shap_values: np.ndarray, df_test: pd.DataFrame,
                      n_suppliers: int = 50):
    """
    Heatmap of SHAP values across a sample of suppliers.
    Shows at a glance which features drive each individual decision.
    """
    print(f"  Plotting: SHAP heatmap (sample of {n_suppliers} suppliers)...")
    n    = min(n_suppliers, len(shap_values))
    idxs = np.random.default_rng(0).choice(len(shap_values), n, replace=False)
    idxs = np.sort(idxs)

    df_hmap = pd.DataFrame(
        shap_values[idxs],
        columns=[FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]
    )

    # Sort columns by mean absolute SHAP
    col_order = df_hmap.abs().mean().sort_values(ascending=False).index
    df_hmap   = df_hmap[col_order]

    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(
        df_hmap.T,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=True,
        linewidths=0,
        cbar_kws={"label": "SHAP Value (contribution to score)"},
    )
    ax.set_xlabel(f"Individual Suppliers (n={n})", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(
        "SHAP Heatmap — Per-Supplier Feature Contributions\n"
        "(Red = increases score, Blue = decreases score)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = f"{PLOTS_DIR}/shap_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_shap_force_example(shap_values: np.ndarray,
                             explainer,
                             X: np.ndarray,
                             df_test: pd.DataFrame):
    """
    Force plot for the top-ranked supplier in a sample query.
    Provides the per-decision explanation example for the paper.
    """
    print("  Plotting: Force plot for top-ranked supplier...")
    # Pick first query with at least 5 suppliers
    for qid in df_test["query_id"].unique():
        m = (df_test["query_id"] == qid).values
        if m.sum() >= 5:
            break

    # Find top supplier in this query (by relevance label)
    sub_idx = np.where(m)[0]
    if "relevance" in df_test.columns:
        best_rel_idx = sub_idx[df_test["relevance"].values[sub_idx].argmax()]
    else:
        best_rel_idx = sub_idx[0]

    feat_labels = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]

    # Bar-based force plot (matplotlib-native, no JS)
    sv   = shap_values[best_rel_idx]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(base[0])

    pos_feats = [(fl, v) for fl, v in zip(feat_labels, sv) if v > 0]
    neg_feats = [(fl, v) for fl, v in zip(feat_labels, sv) if v < 0]
    pos_feats.sort(key=lambda x: x[1], reverse=True)
    neg_feats.sort(key=lambda x: x[1])

    all_feats = pos_feats + neg_feats
    labels    = [x[0] for x in all_feats]
    values    = [x[1] for x in all_feats]
    colors    = ["#2166ac" if v > 0 else "#d73027" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars    = ax.barh(labels, values, color=colors,
                      edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + (0.002 if val > 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center",
            ha="left" if val > 0 else "right",
            fontsize=8.5
        )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Value (contribution to ranking score)", fontsize=11)
    ax.set_title(
        f"SHAP Force Plot — Top-Ranked Supplier (Query: {qid})\n"
        f"Base value: {base:.4f}  |  Blue = pushes score up, Red = pulls down",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/shap_force_top_supplier.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_shap_summary(shap_values: np.ndarray):
    """Print per-feature SHAP statistics table for paper reporting."""
    mean_abs  = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    std_shap  = shap_values.std(axis=0)

    df_sum = pd.DataFrame({
        "Feature":         [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
        "Mean SHAP":       mean_shap.round(5),
        "Mean |SHAP|":     mean_abs.round(5),
        "Std SHAP":        std_shap.round(5),
        "Rank (Impact)":   len(FEATURE_COLS) - np.argsort(np.argsort(mean_abs)),
    }).sort_values("Rank (Impact)")

    print("\n" + "=" * 65)
    print("SHAP GLOBAL FEATURE ATTRIBUTION SUMMARY")
    print("(suitable for Table III in the paper)")
    print("=" * 65)
    print(df_sum.to_string(index=False))

    path = f"{OUT_DIR}/shap_summary_statistics.csv"
    df_sum.to_csv(path, index=False)
    print(f"\n✅ Summary stats saved: {path}")
    return df_sum


# ============================================================================
# MAIN
# ============================================================================

def run_shap_analysis():
    print("=" * 65)
    print("🔍 SourceUp — SHAP Feature Attribution Analysis")
    print("=" * 65)

    # Load
    model   = load_model()
    df_test = load_test_data(test_frac=0.2)

    # Compute SHAP values
    shap_values, explainer, X = compute_shap_values(model, df_test)

    # Save raw values
    save_shap_csv(shap_values, df_test)

    # Plots
    print("\n📊 Generating plots...")
    plot_shap_summary_bar(shap_values, X)
    plot_shap_summary_beeswarm(shap_values, X)
    plot_shap_dependence(shap_values, X, top_n=3)
    plot_shap_heatmap(shap_values, df_test, n_suppliers=60)
    plot_shap_force_example(shap_values, explainer, X, df_test)

    # Summary table
    print_shap_summary(shap_values)

    print(f"\n✅ All SHAP outputs saved in: {OUT_DIR}")
    return shap_values, df_test


if __name__ == "__main__":
    run_shap_analysis()