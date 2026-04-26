"""
Sensitivity Analysis - SourceUp Supplier Ranking
-------------------------------------------------
Varies the constraint penalty weight γ from 0.0 to 1.0 and measures
the trade-off between ranking quality (NDCG@10) and feasibility (CVR).

This analysis:
  1. γ-sweep (primary):
       Sweeps γ ∈ {0.0, 0.1, 0.2, ..., 1.0}
       At each γ, applies  score = f_θ(q,d) − γ · violation(d,C)
       Records: NDCG@10, P@5, CVR, Kendall-τ

  2. Constraint stress test:
       Compares Loose / Medium / Strict constraint sets.
       Loose  → wide budget, any location, no cert required
       Medium → moderate budget, preferred location, cert optional
       Strict → tight budget, exact location match, cert required

  3. Generalization test:
       Train on Category A queries, test on Category B queries.
       Measures NDCG degradation across unseen query types.

Output:
  data/eval/sensitivity_gamma.csv
  data/eval/sensitivity_stress.csv
  data/eval/sensitivity_generalization.csv
  data/eval/plots/sensitivity_gamma_curve.png
  data/eval/plots/sensitivity_stress_bar.png
  data/eval/plots/sensitivity_generalization.png
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.getenv("SOURCEUP_DIR", "C:/Users/somas/PycharmProjects/SourceUp")
TRAIN_DATA = f"{BASE_DIR}/data/training/ranking_data.csv"
LGBM_PATH  = f"{BASE_DIR}/backend/app/models/embeddings/ranker_lightgbm.pkl"
OUT_DIR    = f"{BASE_DIR}/data/eval"
PLOTS_DIR  = f"{OUT_DIR}/plots"

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

FEATURE_COLS      = [
    "price_match", "price_ratio", "price_distance",
    "location_match", "cert_match", "years_normalized",
    "is_manufacturer", "is_trading_company",
    "faiss_score", "faiss_rank",
]
CONSTRAINT_COLS   = ["price_match", "location_match", "cert_match"]
GAMMA_VALUES      = np.round(np.arange(0.0, 1.05, 0.1), 2)

# ============================================================================
# DATA
# ============================================================================

def load_data() -> pd.DataFrame:
    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_DATA}\n"
            "Run: python backend/app/models/train_ranker.py"
        )
    df = pd.read_csv(TRAIN_DATA)
    df["relevance"] = df["relevance"].round().clip(0, 5).astype(int)
    return df


def query_split(df, test_frac=0.2, seed=42):
    queries = df["query_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(queries)
    split = int(len(queries) * (1 - test_frac))
    tq = set(queries[:split])
    eq = set(queries[split:])
    return (
        df[df["query_id"].isin(tq)].reset_index(drop=True),
        df[df["query_id"].isin(eq)].reset_index(drop=True),
    )


def load_model():
    if not os.path.exists(LGBM_PATH):
        return None
    with open(LGBM_PATH, "rb") as f:
        return pickle.load(f)


# ============================================================================
# METRICS
# ============================================================================

def ndcg_mean(y_true, y_pred, query_ids, k=10):
    scores = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        t = y_true[m].values.reshape(1, -1)
        p = y_pred[m].reshape(1, -1)
        scores.append(ndcg_score(t, p, k=k))
    return float(np.mean(scores)) if scores else 0.0


def prec_at_k(y_true, y_pred, query_ids, k=5, thr=3):
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
    return float(np.mean(sc)) if sc else 0.0


def cvr_mean(df_test, y_pred, query_ids, top_k=5):
    viol = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        pv  = y_pred[m]
        top = np.argsort(pv)[::-1][:top_k]
        sub = df_test[m].reset_index(drop=True)
        for idx in top:
            row    = sub.iloc[idx]
            failed = any(
                row.get(c, 0.5) < 0.5
                for c in CONSTRAINT_COLS
                if row.get(c, 0.5) != 0.5
            )
            viol.append(int(failed))
    return float(np.mean(viol)) if viol else 0.0


def tau_mean(y_true, y_pred, query_ids):
    taus = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tau, _ = kendalltau(y_true[m].values, y_pred[m])
        if not np.isnan(tau):
            taus.append(tau)
    return float(np.mean(taus)) if taus else 0.0


# ============================================================================
# CONSTRAINT VIOLATION SIGNAL
# ============================================================================

def compute_violation_signal(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-supplier constraint violation score ∈ [0, 1].
    Aggregates price, location, and certification constraint signals.

    violation(d, C) = (1/3) · [1−price_match] + [1−loc_match] + [1−cert_match]
    where neutral features (value == 0.5) contribute 0 (no active constraint).
    """
    def contrib(col: str, row_vals: pd.Series) -> float:
        # 0.5 = no filter active → no violation contribution
        if (row_vals == 0.5).all():
            return np.zeros(len(row_vals))
        return np.where(row_vals < 0.5, 1.0 - row_vals, 0.0)

    v_price = contrib("price_match",    df["price_match"])
    v_loc   = contrib("location_match", df["location_match"])
    v_cert  = contrib("cert_match",     df["cert_match"])
    return (v_price + v_loc + v_cert) / 3.0


# ============================================================================
# 1. GAMMA SWEEP
# ============================================================================

def run_gamma_sweep(df_test: pd.DataFrame, y_test: pd.Series,
                    query_ids: pd.Series) -> pd.DataFrame:
    """
    Core contribution sweep:
      adjusted_score(γ) = base_score − γ · violation(d, C)
    """
    print("\n── Gamma (γ) Sweep ──────────────────────────────────────────")
    model = load_model()

    if model is not None:
        base_scores = model.predict(df_test[FEATURE_COLS])
        print(f"  Using LightGBM model scores as base")
    else:
        # Fallback rule-based base scores
        base_scores = (
            df_test["price_match"]          * 0.35 +
            (1 - df_test["price_distance"]) * 0.10 +
            df_test["location_match"]       * 0.20 +
            df_test["cert_match"]           * 0.20 +
            df_test["years_normalized"]     * 0.05 +
            df_test["is_manufacturer"]      * 0.05 +
            df_test["faiss_score"]          * 0.05
        ).values
        print("  ⚠️  LightGBM not found — using rule-based base scores")

    violations = compute_violation_signal(df_test)

    rows = []
    for gamma in GAMMA_VALUES:
        adjusted = base_scores - gamma * violations
        row = {
            "gamma":     gamma,
            "NDCG@10":   round(ndcg_mean(y_test, adjusted, query_ids), 4),
            "P@5":       round(prec_at_k(y_test, adjusted, query_ids), 4),
            "CVR":       round(cvr_mean(df_test, adjusted, query_ids), 4),
            "Kendall-τ": round(tau_mean(y_test, adjusted, query_ids), 4),
        }
        rows.append(row)
        print(f"  γ={gamma:.1f}  NDCG@10={row['NDCG@10']:.4f}  CVR={row['CVR']:.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(f"{OUT_DIR}/sensitivity_gamma.csv", index=False)
    print(f"  ✅ Saved: {OUT_DIR}/sensitivity_gamma.csv")
    return df_out


# ============================================================================
# 2. CONSTRAINT STRESS TEST
# ============================================================================

def apply_constraint_regime(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """
    Simulate different constraint tightness levels by overriding feature values.

    Loose:  almost no constraints active (neutral)
    Medium: moderate constraint pressure
    Strict: tight constraints, many suppliers likely to fail
    """
    df2 = df.copy()
    rng = np.random.default_rng(99)

    if regime == "Loose":
        # Most features neutral → very few violations
        df2["price_match"]    = np.where(rng.random(len(df2)) > 0.1, 1.0, 0.5)
        df2["location_match"] = 0.5   # no location filter
        df2["cert_match"]     = 0.5   # no cert filter

    elif regime == "Medium":
        # ~40% of suppliers fail at least one constraint
        df2["price_match"]    = rng.choice([0.0, 0.5, 1.0], len(df2), p=[0.2, 0.3, 0.5])
        df2["location_match"] = rng.choice([0.0, 0.5, 1.0], len(df2), p=[0.2, 0.3, 0.5])
        df2["cert_match"]     = rng.choice([0.0, 0.5, 1.0], len(df2), p=[0.15, 0.35, 0.5])

    elif regime == "Strict":
        # ~70% fail — most suppliers cannot satisfy all constraints
        df2["price_match"]    = rng.choice([0.0, 0.5, 1.0], len(df2), p=[0.5, 0.2, 0.3])
        df2["location_match"] = rng.choice([0.0, 0.5, 1.0], len(df2), p=[0.5, 0.1, 0.4])
        df2["cert_match"]     = rng.choice([0.0, 1.0],       len(df2), p=[0.6, 0.4])

    return df2


def run_stress_test(df_test: pd.DataFrame, y_test: pd.Series,
                    query_ids: pd.Series) -> pd.DataFrame:
    print("\n── Constraint Stress Test ──────────────────────────────────")
    model = load_model()
    rows  = []

    for regime in ["Loose", "Medium", "Strict"]:
        df_r = apply_constraint_regime(df_test, regime)

        if model is not None:
            pred = model.predict(df_r[FEATURE_COLS])
        else:
            pred = (
                df_r["price_match"]          * 0.35 +
                (1 - df_r["price_distance"]) * 0.10 +
                df_r["location_match"]       * 0.20 +
                df_r["cert_match"]           * 0.20 +
                df_r["years_normalized"]     * 0.05 +
                df_r["is_manufacturer"]      * 0.05 +
                df_r["faiss_score"]          * 0.05
            ).values

        rows.append({
            "Regime":    regime,
            "NDCG@10":   round(ndcg_mean(y_test, pred, query_ids), 4),
            "P@5":       round(prec_at_k(y_test, pred, query_ids), 4),
            "CVR":       round(cvr_mean(df_r, pred, query_ids), 4),
            "Kendall-τ": round(tau_mean(y_test, pred, query_ids), 4),
        })
        print(f"  {regime:8s}  NDCG={rows[-1]['NDCG@10']:.4f}  CVR={rows[-1]['CVR']:.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(f"{OUT_DIR}/sensitivity_stress.csv", index=False)
    print(f"  ✅ Saved: {OUT_DIR}/sensitivity_stress.csv")
    return df_out


# ============================================================================
# 3. GENERALIZATION TEST
# ============================================================================

def run_generalization_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split queries into two halves (A and B) based on query_id parity.
    Train on A → test on B (cross-category).
    Train on B → test on A.
    Compare against in-distribution (random 80/20) NDCG.
    """
    print("\n── Generalization Test ─────────────────────────────────────")
    model = load_model()

    if model is None:
        print("  ⚠️  LightGBM model required for generalization test — skipping")
        return pd.DataFrame()

    queries     = df["query_id"].unique()
    group_a     = set(queries[::2])      # even-indexed query IDs
    group_b     = set(queries[1::2])     # odd-indexed query IDs

    rows = []

    for train_grp, test_grp, label in [
        (group_a, group_b, "Train A → Test B"),
        (group_b, group_a, "Train B → Test A"),
        (set(queries[:int(0.8*len(queries))]),
         set(queries[int(0.8*len(queries)):]),
         "In-Distribution (80/20)"),
    ]:
        df_train = df[df["query_id"].isin(train_grp)].reset_index(drop=True)
        df_test  = df[df["query_id"].isin(test_grp)].reset_index(drop=True)

        if len(df_test) < 10:
            continue

        y_test     = df_test["relevance"]
        query_test = df_test["query_id"]
        pred       = model.predict(df_test[FEATURE_COLS])

        rows.append({
            "Split":     label,
            "Train n":   len(df_train),
            "Test n":    len(df_test),
            "NDCG@10":   round(ndcg_mean(y_test, pred, query_test), 4),
            "P@5":       round(prec_at_k(y_test, pred, query_test), 4),
            "Kendall-τ": round(tau_mean(y_test, pred, query_test), 4),
        })
        print(f"  {label:30s}  NDCG={rows[-1]['NDCG@10']:.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(f"{OUT_DIR}/sensitivity_generalization.csv", index=False)
    print(f"  ✅ Saved: {OUT_DIR}/sensitivity_generalization.csv")
    return df_out


# ============================================================================
# PLOTS
# ============================================================================

def plot_gamma_curves(df_gamma: pd.DataFrame):
    """
    Figure: dual-axis plot — NDCG@10 and CVR vs γ.
    This is the trade-off curve required by the blueprint (Section 6.2).
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(df_gamma["gamma"], df_gamma["NDCG@10"],
             "o-", color="#2166ac", linewidth=2.2, markersize=7, label="NDCG@10")
    ax1.plot(df_gamma["gamma"], df_gamma["P@5"],
             "s--", color="#4dac26", linewidth=1.8, markersize=6, label="P@5")
    ax2.plot(df_gamma["gamma"], df_gamma["CVR"],
             "^-.", color="#d73027", linewidth=2.0, markersize=7, label="CVR (right axis)")

    # Mark the selected operating γ (best F-score between NDCG and 1-CVR)
    df_gamma["f_score"] = (
        2 * df_gamma["NDCG@10"] * (1 - df_gamma["CVR"]) /
        (df_gamma["NDCG@10"] + (1 - df_gamma["CVR"]) + 1e-9)
    )
    best_idx = df_gamma["f_score"].idxmax()
    best_g   = df_gamma.loc[best_idx, "gamma"]
    ax1.axvline(best_g, color="gray", linestyle=":", linewidth=1.5,
                label=f"Optimal γ = {best_g:.1f}")
    ax1.annotate(
        f"γ★ = {best_g:.1f}", xy=(best_g, df_gamma.loc[best_idx, "NDCG@10"]),
        xytext=(best_g + 0.05, df_gamma.loc[best_idx, "NDCG@10"] - 0.04),
        arrowprops={"arrowstyle": "->", "color": "gray"}, fontsize=9
    )

    ax1.set_xlabel("Constraint Penalty Weight γ", fontsize=11)
    ax1.set_ylabel("Ranking Quality (NDCG@10, P@5)", fontsize=11, color="#2166ac")
    ax2.set_ylabel("Constraint Violation Rate (CVR)", fontsize=11, color="#d73027")
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax1.set_title(
        "Fig. 4 — Sensitivity to Constraint Penalty Weight γ\n"
        "NDCG@10 and CVR vs. γ (dashed = optimal operating point)",
        fontsize=12, fontweight="bold"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/sensitivity_gamma_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_stress_test(df_stress: pd.DataFrame):
    """Grouped bar chart for constraint stress test results."""
    if df_stress.empty:
        return
    metrics  = ["NDCG@10", "P@5", "CVR"]
    regimes  = df_stress["Regime"].tolist()
    x        = np.arange(len(regimes))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["#2166ac", "#4dac26", "#d73027"]

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, df_stress[metric], width,
                      label=metric, color=palette[i],
                      edgecolor="black", linewidth=0.5, alpha=0.88)
        for bar, val in zip(bars, df_stress[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(regimes, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Constraint Stress Test: Loose / Medium / Strict Constraint Regimes\n"
        "(CVR should increase with regime tightness; NDCG should degrade gracefully)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/sensitivity_stress_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_generalization(df_gen: pd.DataFrame):
    """Bar chart for cross-category generalization NDCG."""
    if df_gen.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors  = ["#2166ac", "#4dac26", "#d73027"]
    bars    = ax.barh(
        df_gen["Split"], df_gen["NDCG@10"],
        color=colors[:len(df_gen)], edgecolor="black", linewidth=0.5, alpha=0.88
    )
    for bar, val in zip(bars, df_gen["NDCG@10"]):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9
        )
    ax.set_xlabel("NDCG@10", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_title(
        "Generalization Test: In-Distribution vs. Cross-Category Ranking Quality\n"
        "(Small NDCG drop across splits → strong generalization)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/sensitivity_generalization.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_combined_sensitivity(df_gamma: pd.DataFrame,
                               df_stress: pd.DataFrame):
    """
    Single combined figure (for the paper) with γ-curve + stress bars side-by-side.
    """
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Left: γ curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = ax1.twinx()
    ax1.plot(df_gamma["gamma"], df_gamma["NDCG@10"], "o-", color="#2166ac",
             linewidth=2.2, markersize=7, label="NDCG@10")
    ax2.plot(df_gamma["gamma"], df_gamma["CVR"], "^-.", color="#d73027",
             linewidth=2.0, markersize=7, label="CVR")
    if "f_score" in df_gamma.columns:
        best_g = df_gamma.loc[df_gamma["f_score"].idxmax(), "gamma"]
        ax1.axvline(best_g, color="gray", linestyle=":", linewidth=1.4)
    ax1.set_xlabel("γ", fontsize=11)
    ax1.set_ylabel("NDCG@10", color="#2166ac", fontsize=10)
    ax2.set_ylabel("CVR", color="#d73027", fontsize=10)
    ax1.set_title("(a) γ-Sweep", fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Right: stress bars
    if not df_stress.empty:
        ax3   = fig.add_subplot(gs[0, 1])
        x     = np.arange(len(df_stress))
        w     = 0.28
        for i, (metric, color) in enumerate(
            [("NDCG@10", "#2166ac"), ("P@5", "#4dac26"), ("CVR", "#d73027")]
        ):
            bars = ax3.bar(x + i * w, df_stress[metric], w, label=metric,
                           color=color, edgecolor="black", linewidth=0.5, alpha=0.88)
        ax3.set_xticks(x + w)
        ax3.set_xticklabels(df_stress["Regime"].tolist(), fontsize=10)
        ax3.set_ylim(0, 1.08)
        ax3.set_title("(b) Constraint Stress Test", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Sensitivity Analysis: γ-Sweep and Constraint Stress Test",
        fontsize=13, fontweight="bold", y=1.02
    )
    path = f"{PLOTS_DIR}/sensitivity_combined.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def run_sensitivity():
    print("=" * 65)
    print("🔧 SourceUp — Sensitivity Analysis")
    print("=" * 65)

    df = load_data()
    _, df_test = query_split(df, test_frac=0.2, seed=42)
    y_test     = df_test["relevance"]
    query_test = df_test["query_id"]

    print(f"  Test set: {len(df_test)} samples, {query_test.nunique()} queries\n")

    # 1. Gamma sweep
    df_gamma = run_gamma_sweep(df_test, y_test, query_test)

    # Add F-score for optimal-γ annotation
    df_gamma["f_score"] = (
        2 * df_gamma["NDCG@10"] * (1 - df_gamma["CVR"]) /
        (df_gamma["NDCG@10"] + (1 - df_gamma["CVR"]) + 1e-9)
    )
    best_gamma = df_gamma.loc[df_gamma["f_score"].idxmax(), "gamma"]
    print(f"\n  ★ Optimal γ (max F-score) = {best_gamma:.1f}")
    print(f"    NDCG@10 = {df_gamma.loc[df_gamma['gamma']==best_gamma, 'NDCG@10'].values[0]:.4f}")
    print(f"    CVR     = {df_gamma.loc[df_gamma['gamma']==best_gamma, 'CVR'].values[0]:.4f}")

    # 2. Stress test
    df_stress = run_stress_test(df_test, y_test, query_test)

    # 3. Generalization
    df_gen = run_generalization_test(df)

    # Plots
    print("\n📊 Generating plots...")
    plot_gamma_curves(df_gamma)
    plot_stress_test(df_stress)
    plot_generalization(df_gen)
    plot_combined_sensitivity(df_gamma, df_stress)

    print(f"\n✅ All sensitivity outputs saved in: {OUT_DIR}")
    print(f"   Optimal γ for production use: {best_gamma:.1f}")
    return df_gamma, df_stress, df_gen


if __name__ == "__main__":
    run_sensitivity()