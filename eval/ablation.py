"""
Ablation Study - SourceUp Supplier Ranking
------------------------------------------
IEEE-compliant ablation across 5 system variants:

  V1: Full Model         (SBERT + Constraint Engine + LightGBM LambdaRank)
  V2: No Constraints     (SBERT + LightGBM, no constraint filtering/penalty)
  V3: No LTR             (SBERT + Constraints + Rule-based scorer)
  V4: No Semantic        (BM25 retrieval + Constraints + LightGBM)
  V5: Rule-Based Only    (Rule-based scorer, no SBERT, no LTR, no constraints)

Metrics reported per variant:
  - NDCG@10   (primary ranking quality)
  - Precision@5
  - MAP (Mean Average Precision)
  - Constraint Violation Rate (CVR)
  - Kendall's Tau (rank correlation with ground truth)

Output:
  data/eval/ablation_results.csv
  data/eval/plots/ablation_table.png
  data/eval/plots/ablation_ndcg_bar.png
  data/eval/plots/ablation_tradeoff.png
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  — adjust BASE_DIR to match your machine
# ---------------------------------------------------------------------------
BASE_DIR   = os.getenv("SOURCEUP_DIR", "C:/Users/somas/PycharmProjects/SourceUp")
CLEAN_DATA = f"{BASE_DIR}/data/clean/suppliers_clean.csv"
TRAIN_DATA = f"{BASE_DIR}/data/training/ranking_data.csv"
LGBM_PATH  = f"{BASE_DIR}/backend/app/models/embeddings/ranker_lightgbm.pkl"
OUT_DIR    = f"{BASE_DIR}/data/eval"
PLOTS_DIR  = f"{OUT_DIR}/plots"

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Optional heavy imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("⚠️  LightGBM not found — V1/V4 will use XGBoost fallback")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  rank_bm25 not found — V4 will approximate BM25 via TF-IDF")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False

# ============================================================================
# SHARED HELPERS  (replicated from train_ranker.py to keep eval self-contained)
# ============================================================================

FEATURE_COLS = [
    "price_match", "price_ratio", "price_distance",
    "location_match", "cert_match", "years_normalized",
    "is_manufacturer", "is_trading_company",
    "faiss_score", "faiss_rank",
]

CONSTRAINT_COLS = ["price_match", "location_match", "cert_match"]


def parse_price(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    try:
        s = str(v).strip()
        return float(s.split("-")[0].strip()) if "-" in s else float(s)
    except Exception:
        return 0.0


def load_training_data() -> pd.DataFrame:
    """Load or regenerate the training / evaluation dataset."""
    if os.path.exists(TRAIN_DATA):
        df = pd.read_csv(TRAIN_DATA)
        print(f"✅ Loaded training data: {len(df)} rows, {df['query_id'].nunique()} queries")
        return df
    raise FileNotFoundError(
        f"Training data not found at {TRAIN_DATA}.\n"
        "Run: python backend/app/models/train_ranker.py"
    )


def query_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Split by query ID so no query appears in both train and test."""
    queries = df["query_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(queries)
    split = int(len(queries) * (1 - test_frac))
    train_qids = set(queries[:split])
    test_qids  = set(queries[split:])
    return (
        df[df["query_id"].isin(train_qids)].reset_index(drop=True),
        df[df["query_id"].isin(test_qids)].reset_index(drop=True),
    )


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def ndcg_at_k(y_true: pd.Series, y_pred: np.ndarray,
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


def precision_at_k(y_true: pd.Series, y_pred: np.ndarray,
                   query_ids: pd.Series, k: int = 5,
                   threshold: int = 3) -> float:
    scores = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        true_vals = y_true[m].values
        pred_vals = y_pred[m]
        top_k_idx = np.argsort(pred_vals)[::-1][:k]
        relevant  = (true_vals >= threshold)
        if relevant.sum() == 0:
            continue
        scores.append(relevant[top_k_idx].sum() / min(k, len(top_k_idx)))
    return float(np.mean(scores)) if scores else 0.0


def mean_average_precision(y_true: pd.Series, y_pred: np.ndarray,
                           query_ids: pd.Series, threshold: int = 3) -> float:
    ap_scores = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        true_vals = y_true[m].values
        pred_vals = y_pred[m]
        order     = np.argsort(pred_vals)[::-1]
        relevant  = (true_vals[order] >= threshold)
        if relevant.sum() == 0:
            continue
        precisions = []
        hits = 0
        for i, rel in enumerate(relevant):
            if rel:
                hits += 1
                precisions.append(hits / (i + 1))
        ap_scores.append(np.mean(precisions) if precisions else 0.0)
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def constraint_violation_rate(df_test: pd.DataFrame,
                              y_pred: np.ndarray,
                              query_ids: pd.Series,
                              top_k: int = 5) -> float:
    """
    CVR = fraction of top-k retrieved suppliers that fail at least one constraint.
    Constraints (from feature columns): price_match, location_match, cert_match.
    A supplier passes a constraint if the feature value >= 0.5 (binary columns)
    or the column is neutral (0.5 exactly, meaning no filter was specified).
    """
    violations = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        pred_vals = y_pred[m]
        top_idx   = np.argsort(pred_vals)[::-1][:top_k]
        sub_df    = df_test[m].reset_index(drop=True)
        for idx in top_idx:
            row = sub_df.iloc[idx]
            # A "violation" means the supplier was shown despite failing a
            # hard constraint (value strictly < 0.5 means filter was active
            # and the supplier didn't pass it)
            failed = any(
                row.get(c, 0.5) < 0.5
                for c in CONSTRAINT_COLS
                if row.get(c, 0.5) != 0.5    # 0.5 = no filter specified
            )
            violations.append(int(failed))
    return float(np.mean(violations)) if violations else 0.0


def avg_kendall_tau(y_true: pd.Series, y_pred: np.ndarray,
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


def evaluate_variant(name: str, y_true: pd.Series, y_pred: np.ndarray,
                     df_test: pd.DataFrame, query_ids: pd.Series) -> Dict:
    return {
        "Variant":    name,
        "NDCG@10":    round(ndcg_at_k(y_true, y_pred, query_ids, k=10), 4),
        "P@5":        round(precision_at_k(y_true, y_pred, query_ids, k=5), 4),
        "MAP":        round(mean_average_precision(y_true, y_pred, query_ids), 4),
        "CVR":        round(constraint_violation_rate(df_test, y_pred, query_ids), 4),
        "Kendall-τ":  round(avg_kendall_tau(y_true, y_pred, query_ids), 4),
    }


# ============================================================================
# VARIANT SCORERS
# ============================================================================

def load_lgbm_model():
    """Load saved LightGBM model from disk."""
    if not os.path.exists(LGBM_PATH):
        return None
    with open(LGBM_PATH, "rb") as f:
        return pickle.load(f)


def score_v1_full_model(df_test: pd.DataFrame) -> np.ndarray:
    """
    V1: Full Model — LightGBM LambdaRank on all features.
    Constraint engine already baked into training labels and features.
    """
    model = load_lgbm_model()
    if model is not None:
        return model.predict(df_test[FEATURE_COLS])
    # Fallback: rule-based if model not available
    print("   ⚠️  LightGBM model not found for V1 — using rule-based fallback")
    return score_rule_based(df_test)


def score_v2_no_constraints(df_test: pd.DataFrame) -> np.ndarray:
    """
    V2: No Constraints — same LightGBM model but constraint features
    zeroed out / set to neutral (0.5), simulating a system that ignores
    budget, location, and certification filters.
    """
    df_nc = df_test.copy()
    # Set constraint features to neutral (as if no filter was specified)
    for col in CONSTRAINT_COLS:
        df_nc[col] = 0.5
    # Also zero price_distance and price_ratio (price constraint inactive)
    df_nc["price_distance"] = 0.0
    df_nc["price_ratio"]    = 1.0

    model = load_lgbm_model()
    if model is not None:
        return model.predict(df_nc[FEATURE_COLS])
    return score_rule_based(df_nc)


def score_v3_no_ltr(df_test: pd.DataFrame) -> np.ndarray:
    """
    V3: No LTR — rule-based scorer with constraint features intact.
    Mimics the system without the ML ranker.
    """
    return score_rule_based(df_test)


def score_rule_based(df: pd.DataFrame) -> np.ndarray:
    """
    Rule-based scoring matching ranker.py weights exactly.
    Used by V3 (No LTR) and V5 (Rule-Based Only).
    """
    scores = (
        df["price_match"]        * 0.35 +
        (1 - df["price_distance"]) * 0.10 +
        df["location_match"]     * 0.20 +
        df["cert_match"]         * 0.20 +
        df["years_normalized"]   * 0.05 +
        df["is_manufacturer"]    * 0.05 +
        df["faiss_score"]        * 0.05
    )
    return scores.values


def score_v4_no_semantic(df_train: pd.DataFrame, df_test: pd.DataFrame) -> np.ndarray:
    """
    V4: No Semantic Retrieval — replaces FAISS cosine similarity with
    BM25 (or TF-IDF cosine) over product name text, then re-scores using
    LightGBM on modified features.
    """
    # Build a text corpus from training products for BM25
    if "product name" not in df_train.columns:
        # Graceful: if product name not in data, simulate with random noise
        print("   ⚠️  'product name' column not found — simulating BM25 with noise")
        bm25_scores = np.random.uniform(0.3, 0.7, len(df_test))
    else:
        corpus_train = df_train["product name"].fillna("").astype(str).tolist()
        corpus_test  = df_test["product name"].fillna("").astype(str).tolist()

        if BM25_AVAILABLE and corpus_train:
            tokenized = [doc.lower().split() for doc in corpus_train]
            bm25 = BM25Okapi(tokenized)
            bm25_scores = np.array([
                np.mean(bm25.get_scores(doc.lower().split()))
                for doc in corpus_test
            ])
        elif TFIDF_AVAILABLE and corpus_train:
            vec = TfidfVectorizer(max_features=5000)
            train_mat = vec.fit_transform(corpus_train)
            test_mat  = vec.transform(corpus_test)
            sim = sk_cosine(test_mat, train_mat).mean(axis=1)
            bm25_scores = sim
        else:
            bm25_scores = np.random.uniform(0.3, 0.7, len(df_test))

        # Normalise to [0, 1]
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (
                bm25_scores.max() - bm25_scores.min()
            )

    df_ns = df_test.copy()
    df_ns["faiss_score"] = bm25_scores
    df_ns["faiss_rank"]  = len(df_ns) - np.argsort(bm25_scores)  # invert: lower rank = better

    model = load_lgbm_model()
    if model is not None:
        return model.predict(df_ns[FEATURE_COLS])
    return score_rule_based(df_ns)


def score_v5_rule_only(df_test: pd.DataFrame) -> np.ndarray:
    """
    V5: Rule-Based Only — no SBERT, no LTR, no constraint engine.
    FAISS score set to 0.5 (neutral), constraints all neutral.
    """
    df_ro = df_test.copy()
    df_ro["faiss_score"] = 0.5
    df_ro["faiss_rank"]  = 500
    for col in CONSTRAINT_COLS:
        df_ro[col] = 0.5
    df_ro["price_distance"] = 0.0
    df_ro["price_ratio"]    = 1.0
    return score_rule_based(df_ro)


# ============================================================================
# VISUALIZATION
# ============================================================================

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

_VARIANT_COLORS = {
    "V1: Full Model":          "#2166ac",
    "V2: No Constraints":      "#d73027",
    "V3: No LTR":              "#f46d43",
    "V4: No Semantic":         "#fdae61",
    "V5: Rule-Based Only":     "#abd9e9",
}


def plot_ablation_bar(results_df: pd.DataFrame):
    """Primary ablation bar chart: NDCG@10, P@5, MAP side-by-side."""
    metrics  = ["NDCG@10", "P@5", "MAP"]
    variants = results_df["Variant"].tolist()
    x        = np.arange(len(variants))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = ["#2166ac", "#4dac26", "#d01c8b"]

    for i, metric in enumerate(metrics):
        vals = results_df[metric].values
        bars = ax.bar(x + i * width, vals, width, label=metric,
                      color=palette[i], edgecolor="black", linewidth=0.6, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Ablation Study: Ranking Quality Across System Variants\n"
        "(V1 = Full Model; lower NDCG/P@5/MAP = contribution confirmed)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(results_df.loc[results_df["Variant"] == "V1: Full Model", "NDCG@10"].values[0],
               color="#2166ac", linestyle="--", linewidth=1.0, alpha=0.5)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/ablation_ndcg_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_ablation_tradeoff(results_df: pd.DataFrame):
    """Trade-off scatter: NDCG@10 vs Constraint Violation Rate."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in results_df.iterrows():
        color = _VARIANT_COLORS.get(row["Variant"], "#888888")
        ax.scatter(row["CVR"], row["NDCG@10"], s=160,
                   color=color, edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(
            row["Variant"],
            (row["CVR"], row["NDCG@10"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8.5
        )

    ax.set_xlabel("Constraint Violation Rate (CVR) — lower is better", fontsize=11)
    ax.set_ylabel("NDCG@10 — higher is better", fontsize=11)
    ax.set_title(
        "Ranking Quality vs. Feasibility Trade-off\n"
        "(Ideal system: top-left corner — high NDCG, low CVR)",
        fontsize=12, fontweight="bold"
    )
    ax.invert_xaxis()   # lower CVR → right visually
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/ablation_tradeoff.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_ablation_table(results_df: pd.DataFrame):
    """Render results as a publication-ready table figure."""
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")

    col_labels = results_df.columns.tolist()
    cell_text  = [
        [str(v) for v in row] for row in results_df.values
    ]

    # Highlight best value per numeric column
    cell_colors = [["white"] * len(col_labels) for _ in range(len(results_df))]
    for ci, col in enumerate(col_labels[1:], start=1):    # skip "Variant"
        col_vals = results_df[col].values.astype(float)
        # For CVR lower is better; for all others higher is better
        best_idx = int(np.argmin(col_vals)) if col == "CVR" else int(np.argmax(col_vals))
        cell_colors[best_idx][ci] = "#d4edda"             # light green highlight

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Bold header
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#ddeeff")

    ax.set_title(
        "Table I — Ablation Study Results  "
        "(green = best per metric; CVR lower is better)",
        fontsize=11, fontweight="bold", pad=12
    )
    plt.tight_layout()
    path = f"{PLOTS_DIR}/ablation_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def run_ablation():
    print("=" * 65)
    print("🔬 SourceUp — Ablation Study")
    print("=" * 65)

    df = load_training_data()

    # Round relevance to integers (0-5) for ranking metrics
    df["relevance"] = df["relevance"].round().clip(0, 5).astype(int)

    df_train, df_test = query_split(df, test_frac=0.2, seed=42)
    y_test     = df_test["relevance"]
    query_test = df_test["query_id"]

    print(f"  Train: {len(df_train)} samples | Test: {len(df_test)} samples")
    print(f"  Test queries: {query_test.nunique()}")
    print()

    results = []

    # ------------------------------------------------------------------
    # V1: Full Model
    # ------------------------------------------------------------------
    print("▶ V1: Full Model (SBERT + Constraint Engine + LightGBM LambdaRank)")
    pred_v1 = score_v1_full_model(df_test)
    results.append(evaluate_variant("V1: Full Model", y_test, pred_v1, df_test, query_test))
    print(f"   NDCG@10 = {results[-1]['NDCG@10']:.4f}   CVR = {results[-1]['CVR']:.4f}")

    # ------------------------------------------------------------------
    # V2: No Constraints
    # ------------------------------------------------------------------
    print("▶ V2: No Constraints (constraints neutralised)")
    pred_v2 = score_v2_no_constraints(df_test)
    results.append(evaluate_variant("V2: No Constraints", y_test, pred_v2, df_test, query_test))
    print(f"   NDCG@10 = {results[-1]['NDCG@10']:.4f}   CVR = {results[-1]['CVR']:.4f}")

    # ------------------------------------------------------------------
    # V3: No LTR
    # ------------------------------------------------------------------
    print("▶ V3: No LTR (rule-based scorer only)")
    pred_v3 = score_v3_no_ltr(df_test)
    results.append(evaluate_variant("V3: No LTR", y_test, pred_v3, df_test, query_test))
    print(f"   NDCG@10 = {results[-1]['NDCG@10']:.4f}   CVR = {results[-1]['CVR']:.4f}")

    # ------------------------------------------------------------------
    # V4: No Semantic Retrieval
    # ------------------------------------------------------------------
    print("▶ V4: No Semantic Retrieval (BM25/TF-IDF instead of SBERT+FAISS)")
    pred_v4 = score_v4_no_semantic(df_train, df_test)
    results.append(evaluate_variant("V4: No Semantic", y_test, pred_v4, df_test, query_test))
    print(f"   NDCG@10 = {results[-1]['NDCG@10']:.4f}   CVR = {results[-1]['CVR']:.4f}")

    # ------------------------------------------------------------------
    # V5: Rule-Based Only
    # ------------------------------------------------------------------
    print("▶ V5: Rule-Based Only (no SBERT, no LTR, no constraints)")
    pred_v5 = score_v5_rule_only(df_test)
    results.append(evaluate_variant("V5: Rule-Based Only", y_test, pred_v5, df_test, query_test))
    print(f"   NDCG@10 = {results[-1]['NDCG@10']:.4f}   CVR = {results[-1]['CVR']:.4f}")

    # ------------------------------------------------------------------
    # Compile & save
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)

    csv_path = f"{OUT_DIR}/ablation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved: {csv_path}")

    # Compute delta vs V1 (contribution of each removed component)
    v1_ndcg = results_df.loc[results_df["Variant"] == "V1: Full Model", "NDCG@10"].values[0]
    results_df["ΔNDCG vs V1"] = (results_df["NDCG@10"] - v1_ndcg).round(4)

    print("\n" + "=" * 65)
    print("ABLATION RESULTS")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print("\nKey: ΔNDCG < 0 → removing that component hurts ranking quality")
    print("     CVR: Constraint Violation Rate (lower = more feasible results)")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n📊 Generating plots...")
    plot_ablation_bar(results_df)
    plot_ablation_tradeoff(results_df)
    plot_ablation_table(results_df)

    print(f"\n✅ All ablation outputs saved in: {OUT_DIR}")
    return results_df


if __name__ == "__main__":
    run_ablation()