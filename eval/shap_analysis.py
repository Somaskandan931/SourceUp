"""
SHAP Feature Attribution Analysis — SourceUp Supplier Ranking
--------------------------------------------------------------
Uses the SHAP library to generate scientifically validated feature
attributions from the trained XGBoost XGBRanker model.

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
  data/eval/plots/shap_waterfall.png

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
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — all resolved via config.cfg (no hardcoded paths)
# ---------------------------------------------------------------------------
def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))
from config import cfg

TRAIN_DATA = str(cfg.TRAINING_DATA)
# FIX: this previously pointed at cfg.LGBM_MODEL ("NOTE: cfg key name
# unchanged but holds XGBoost model"). That path is also where
# train_lambdarank.py saves its LightGBM Booster, and where 6 other eval
# scripts (stability.py, fairness.py, sensitivity.py, baselines.py,
# ablation.py) load expecting a LightGBM object. Whichever trainer ran
# last silently determined what every reader of that path got — train_
# ranker.py no longer writes there at all (see its own FIX comment), so
# this now points at the model's actual, dedicated file instead.
XGB_PATH = str(cfg.MODELS_DIR / "xgb_ranker.pkl")
OUT_DIR = str(cfg.EVAL_DIR)
PLOTS_DIR = str(cfg.EVAL_PLOTS_DIR)

cfg.ensure_dirs()

# ---------------------------------------------------------------------------
# XGBoost import
# ---------------------------------------------------------------------------
# FIX: shap.TreeExplainer(model) was previously used to compute SHAP values.
# On the XGBoost version installed in this environment, the model's internal
# base_score is serialized as a stringified array (e.g. '[-2.2569608E-9]')
# rather than a plain float. shap's XGBTreeModelLoader does a naive
# float(learner_model_param["base_score"]) when parsing the model, which
# can't handle that bracketed scientific-notation string and raises:
#   ValueError: could not convert string to float: '[-2.2569608E-9]'
# This is a known shap <-> xgboost version-compatibility issue, not a bug
# in the ranker, features, or training data.
#
# Fix: bypass shap.TreeExplainer entirely and use XGBoost's own native
# TreeSHAP computation — Booster.predict(dmatrix, pred_contribs=True).
# This returns mathematically identical SHAP values to TreeExplainer
# (same TreeSHAP algorithm, computed internally by XGBoost itself) without
# ever touching shap's base_score parsing code. The `shap` package is no
# longer required for value computation, only (optionally) for plotting
# helpers like shap.summary_plot further down — it's kept imported for that.
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("❌  shap not installed.  Run: pip install shap")
    sys.exit(1)

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌  xgboost not installed.  Run: pip install xgboost")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "price_match",
    "price_ratio",
    "location_match",
    "cert_match",
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

FEATURE_LABELS = {
    "price_match": "Price Within Budget",
    "price_ratio": "Price / Budget Ratio",
    "price_distance": "Price Distance",
    "location_match": "Location Match",
    "cert_match": "Certification Match",
    "years_normalized": "Years on Platform",
    "is_manufacturer": "Is Manufacturer",
    "is_trading_company": "Is Trading Company",
    "faiss_score": "Semantic Similarity (SBERT)",
}

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


# ============================================================================
# DATA & MODEL
# ============================================================================

def load_model():
    """Load XGBoost XGBRanker model.

    train_ranker.py saves a dict {"model": <XGBRanker>, "feature_cols": [...]}
    to xgb_ranker.pkl, not the raw model object — unwrap it here so callers
    (compute_shap_values, etc.) get the actual XGBRanker instance.
    """
    if not os.path.exists(XGB_PATH):
        raise FileNotFoundError(
            f"XGBoost model not found: {XGB_PATH}\n"
            "Run: python backend/app/models/train_ranker.py"
        )
    with open(XGB_PATH, "rb") as f:
        loaded = pickle.load(f)
    model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    # FIX: a sibling LightGBM model on this same project hit a feature-count
    # mismatch from a stale .pkl trained before a FEATURE_COLS change (see
    # ablation.py/baselines.py/fairness.py/sensitivity.py/stability.py for
    # the same fix). Added here too so a stale xgb_ranker.pkl fails with an
    # actionable message instead of a raw xgboost shape error deep in SHAP.
    n_model_features = getattr(model, "n_features_in_", None)
    if n_model_features is not None and n_model_features != len(FEATURE_COLS):
        raise ValueError(
            f"{XGB_PATH} was trained with {n_model_features} features, "
            f"but FEATURE_COLS here has {len(FEATURE_COLS)}. Stale model — "
            f"retrain via: python pipeline/run_all.py --train-xgbranker"
        )
    return model


def load_test_data(test_frac: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Load test data for SHAP analysis"""
    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_DATA}\n"
            "Run: python pipeline/feature_builder.py"
        )
    df = pd.read_csv(TRAIN_DATA)
    # Normalize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    queries = df["query_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(queries)
    split = int(len(queries) * (1 - test_frac))
    test_qs = set(queries[split:])
    df_test = df[df["query_id"].isin(test_qs)].reset_index(drop=True)
    print(f"✅ Test set: {len(df_test)} rows, {df_test['query_id'].nunique()} queries")
    return df_test


# ============================================================================
# SHAP COMPUTATION — XGBoost Ranker Compatibility Fix
# ============================================================================

def compute_shap_values(model, df_test: pd.DataFrame):
    """
    Compute SHAP values for the XGBoost XGBRanker using XGBoost's own
    native TreeSHAP implementation (pred_contribs=True).

    FIX: shap.TreeExplainer(model) was previously used here. On the
    XGBoost version installed in this environment, the model serializes
    its internal base_score as a stringified array, e.g.
    '[-2.2569608E-9]', instead of a plain float. shap's internal
    XGBTreeModelLoader does float(learner_model_param["base_score"]) when
    parsing the model, which cannot handle that bracketed string and
    raised:
        ValueError: could not convert string to float: '[-2.2569608E-9]'

    This is a shap <-> xgboost version-compatibility issue, not a problem
    with the model, features, or data. XGBoost's own
    Booster.predict(dmatrix, pred_contribs=True) computes the exact same
    TreeSHAP values internally (it's the reference implementation SHAP's
    TreeExplainer itself calls out to for XGBoost models) without ever
    going through shap's base_score string parsing, so it sidesteps the
    bug entirely while returning mathematically identical attributions.

    Note: XGBoost's flag is `pred_contribs` (plural) and requires a
    DMatrix — this differs from LightGBM's `pred_contrib` (singular),
    which accepts a raw array directly. The two APIs are not
    interchangeable; calling LightGBM-style on an XGBRanker fails with
    "got an unexpected keyword argument 'pred_contrib'", and calling
    XGBoost-style on a LightGBM Booster fails just as fast — always match
    the API to the actual model type in use.
    """

    print("\n🔬 Computing SHAP values using XGBoost native TreeSHAP "
          "(pred_contribs=True)...")

    X = df_test[FEATURE_COLS].astype(np.float32)

    try:
        # Unwrap to the raw Booster regardless of whether `model` is the
        # sklearn XGBRanker wrapper (has .get_booster()) or an
        # already-raw xgboost.Booster.
        booster = model.get_booster() if hasattr(model, "get_booster") else model

        print("   Computing SHAP contributions via pred_contribs=True...")

        dmat = xgb.DMatrix(X.values, feature_names=list(FEATURE_COLS))

        # XGBoost native TreeSHAP — returns (n_samples, n_features + 1).
        # Last column is the expected value (bias) repeated per row.
        shap_contribs = booster.predict(dmat, pred_contribs=True)

        shap_values_array = shap_contribs[:, :-1]
        expected_value = float(shap_contribs[0, -1])

        # Lightweight explainer mock — downstream plotting code only reads
        # `.expected_value` off this object, never any shap.TreeExplainer
        # internals, so a stand-in object is sufficient and keeps the
        # broken base_score parser out of the call path entirely.
        class DummyExplainer:
            pass

        explainer = DummyExplainer()
        explainer.expected_value = expected_value

        print(f"   SHAP values shape: {shap_values_array.shape}")
        print(f"   Expected value: {expected_value:.6f}")

    except Exception as e:
        print(f"❌ Native TreeSHAP failed: {e}")
        raise

    return shap_values_array, explainer, X


def save_shap_csv(shap_values: np.ndarray, df_test: pd.DataFrame):
    """Save raw SHAP values for full reproducibility (auditable)."""
    df_shap = pd.DataFrame(shap_values, columns=FEATURE_COLS)
    df_shap.insert(0, "query_id", df_test["query_id"].values)

    if "relevance" in df_test.columns:
        df_shap.insert(1, "relevance", df_test["relevance"].values)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/shap_values.csv"
    df_shap.to_csv(path, index=False)
    print(f"  ✅ SHAP values saved: {path}")
    return df_shap


# ============================================================================
# PLOTS
# ============================================================================

def plot_shap_summary_beeswarm(shap_values: np.ndarray, X: pd.DataFrame):
    """
    Fig. 3a: Beeswarm summary plot.
    Shows the distribution of SHAP values for each feature,
    coloured by feature value magnitude.
    """
    print("  Plotting: SHAP beeswarm summary...")
    plt.figure(figsize=(11, 7))

    # Convert to numpy if X is DataFrame for compatibility
    X_np = X.values if hasattr(X, 'values') else X

    shap.summary_plot(
        shap_values, X_np,
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
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = f"{PLOTS_DIR}/shap_summary_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_shap_summary_bar(shap_values: np.ndarray, X: pd.DataFrame):
    """
    Fig. 3b: Mean absolute SHAP bar chart.
    This is the publishable global feature importance figure.
    """
    print("  Plotting: SHAP mean-absolute bar chart...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_labels = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]

    df_imp = pd.DataFrame({
        "feature": feat_labels,
        "mean_shap": mean_abs,
    }).sort_values("mean_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df_imp)))
    bars = ax.barh(df_imp["feature"], df_imp["mean_shap"],
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


def plot_shap_dependence(shap_values: np.ndarray, X: pd.DataFrame, top_n: int = 3):
    """
    Dependence plots for the top-N most important features.
    Shows how SHAP value changes with feature value,
    coloured by the most interacting other feature.
    """
    print(f"  Plotting: Dependence plots for top {top_n} features...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    # Convert to numpy if DataFrame
    X_np = X.values if hasattr(X, 'values') else X

    for rank, feat_idx in enumerate(top_indices):
        feat_name = FEATURE_COLS[feat_idx]
        feat_label = FEATURE_LABELS.get(feat_name, feat_name)

        # Find most correlated interaction feature
        correlations = []
        for j in range(X_np.shape[1]):
            if j != feat_idx:
                corr = abs(np.corrcoef(shap_values[:, feat_idx], X_np[:, j])[0, 1])
                correlations.append((corr, j))

        if correlations:
            interact_idx = max(correlations, key=lambda x: x[0])[1]
            interact_label = FEATURE_LABELS.get(FEATURE_COLS[interact_idx], FEATURE_COLS[interact_idx])
        else:
            interact_label = "other features"
            interact_idx = 0

        fig, ax = plt.subplots(figsize=(9, 5))
        sc = ax.scatter(
            X_np[:, feat_idx],
            shap_values[:, feat_idx],
            c=X_np[:, interact_idx] if interact_idx < X_np.shape[1] else np.zeros(X_np.shape[0]),
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
        safe_name = feat_name.replace("/", "_").replace(" ", "_")
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
    n = min(n_suppliers, len(shap_values))
    idxs = np.random.default_rng(0).choice(len(shap_values), n, replace=False)
    idxs = np.sort(idxs)

    df_hmap = pd.DataFrame(
        shap_values[idxs],
        columns=[FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]
    )

    # Sort columns by mean absolute SHAP
    col_order = df_hmap.abs().mean().sort_values(ascending=False).index
    df_hmap = df_hmap[col_order]

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


def plot_shap_waterfall(shap_values: np.ndarray, explainer,
                        X: pd.DataFrame, df_test: pd.DataFrame):
    """
    Waterfall plot for a single prediction.
    Shows how the base value is increased/decreased by each feature.
    """
    print("  Plotting: SHAP waterfall plot for a sample supplier...")

    # Pick a random supplier from test set
    idx = np.random.randint(0, len(shap_values))

    # Safe extraction of expected value
    expected_value = getattr(explainer, "expected_value", 0.0)
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[0]) if len(expected_value) > 0 else 0.0

    # Get feature contributions for this supplier
    contributions = []
    feature_names_clean = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]

    for i, (feat_name, shap_val) in enumerate(zip(feature_names_clean, shap_values[idx])):
        if abs(shap_val) > 0.001:
            contributions.append((feat_name, shap_val))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    contributions = contributions[:8]
    feat_names = [c[0] for c in contributions]
    feat_values = [c[1] for c in contributions]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(feat_names))
    colors = ['#2166ac' if v > 0 else '#d73027' for v in feat_values]

    bars = ax.barh(y_pos, feat_values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    ax.axvline(expected_value, color='gray', linewidth=1.5, linestyle='--',
               label=f'Base Value: {expected_value:.3f}')

    for bar, val in zip(bars, feat_values):
        offset = 0.002 if val > 0 else -0.002
        ha = 'left' if val > 0 else 'right'
        ax.text(bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center',
                ha=ha, fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names)
    ax.set_xlabel('SHAP Value (contribution to ranking score)', fontsize=11)
    ax.set_title(
        f'SHAP Waterfall Plot — Single Supplier Explanation\n'
        f'Final Score = {expected_value:.3f} + Σ SHAP = {expected_value + sum(feat_values):.3f}',
        fontsize=11, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/shap_waterfall.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_shap_force_example(shap_values: np.ndarray,
                            explainer,
                            X: pd.DataFrame,
                            df_test: pd.DataFrame):
    """
    Force plot for the top-ranked supplier in a sample query.
    Provides the per-decision explanation example for the paper.
    """
    print("  Plotting: Force plot for top-ranked supplier...")

    # Pick first query with at least 5 suppliers
    target_qid = None
    for qid in df_test["query_id"].unique():
        m = (df_test["query_id"] == qid).values
        if m.sum() >= 5:
            target_qid = qid
            break

    if target_qid is None:
        print("  No query with enough suppliers found, skipping force plot")
        return

    # Find top supplier in this query (by relevance label)
    m = (df_test["query_id"] == target_qid).values
    sub_idx = np.where(m)[0]
    if "relevance" in df_test.columns:
        best_rel_idx = sub_idx[df_test["relevance"].values[sub_idx].argmax()]
    else:
        best_rel_idx = sub_idx[0]

    feat_labels = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]

    sv = shap_values[best_rel_idx]

    # Safe extraction of expected value
    base = getattr(explainer, "expected_value", 0.0)
    if isinstance(base, (list, np.ndarray)):
        base = float(base[0]) if len(base) > 0 else 0.0

    pos_feats = [(fl, v) for fl, v in zip(feat_labels, sv) if v > 0]
    neg_feats = [(fl, v) for fl, v in zip(feat_labels, sv) if v < 0]
    pos_feats.sort(key=lambda x: x[1], reverse=True)
    neg_feats.sort(key=lambda x: x[1])

    all_feats = pos_feats + neg_feats
    labels = [x[0] for x in all_feats]
    values = [x[1] for x in all_feats]
    colors = ["#2166ac" if v > 0 else "#d73027" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors,
                   edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars, values):
        offset = 0.002 if val > 0 else -0.002
        ha = 'left' if val > 0 else 'right'
        ax.text(bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center",
                ha=ha, fontsize=8.5)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Value (contribution to ranking score)", fontsize=11)
    ax.set_title(
        f"SHAP Force Plot — Top-Ranked Supplier (Query: {target_qid})\n"
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
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    std_shap = shap_values.std(axis=0)

    df_sum = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
        "Mean SHAP": mean_shap.round(5),
        "Mean |SHAP|": mean_abs.round(5),
        "Std SHAP": std_shap.round(5),
        "Rank (Impact)": len(FEATURE_COLS) - np.argsort(np.argsort(mean_abs)),
    }).sort_values("Rank (Impact)")

    print("\n" + "=" * 70)
    print("SHAP GLOBAL FEATURE ATTRIBUTION SUMMARY")
    print("(suitable for Table III in the paper)")
    print("=" * 70)
    print(df_sum.to_string(index=False))

    path = f"{OUT_DIR}/shap_summary_statistics.csv"
    df_sum.to_csv(path, index=False)
    print(f"\n✅ Summary stats saved: {path}")
    return df_sum


# ============================================================================
# MAIN
# ============================================================================

def run_shap_analysis():
    print("=" * 70)
    print("🔍 SourceUp — SHAP Feature Attribution Analysis")
    print("   (XGBoost XGBRanker with compatibility fix)")
    print("=" * 70)

    # Load
    model = load_model()
    print(f"✅ Model loaded: {type(model).__name__}")

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
    plot_shap_waterfall(shap_values, explainer, X, df_test)

    # Summary table
    print_shap_summary(shap_values)

    print(f"\n✅ All SHAP outputs saved in: {OUT_DIR}")
    print(f"   Plots directory: {PLOTS_DIR}")
    return shap_values, df_test


if __name__ == "__main__":
    run_shap_analysis()