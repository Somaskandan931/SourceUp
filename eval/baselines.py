"""
Baseline Comparisons - SourceUp Supplier Ranking
-------------------------------------------------
Compares SourceUp's full model against three standard baselines:

  B1: BM25 (keyword retrieval — Okapi BM25 over product name)
  B2: SBERT Cosine Similarity (semantic retrieval, no constraint engine)
  B3: Rule-Based Ranker (heuristic scorer from ranker.py, no LTR)
  B4: Random Ranking (theoretical lower bound)
  S1: SourceUp Full Model (constraint-aware LambdaRank — our system)

Metrics: NDCG@10, P@5, MAP, CVR, Kendall-τ, MRR
Statistical significance: paired Wilcoxon signed-rank test vs S1.

Output:
  data/eval/baseline_results.csv
  data/eval/plots/baseline_comparison_bar.png
  data/eval/plots/baseline_ndcg_per_query.png
  data/eval/plots/baseline_significance_table.png
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import wilcoxon, kendalltau
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
# Optional imports
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  rank_bm25 not found — pip install rank_bm25")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️  sentence_transformers not found — SBERT baseline will use TF-IDF")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine
    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

FEATURE_COLS = [
    "price_match", "price_ratio", "price_distance",
    "location_match", "cert_match", "years_normalized",
    "is_manufacturer", "is_trading_company",
    "faiss_score", "faiss_rank",
]

# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_data() -> pd.DataFrame:
    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_DATA}.\n"
            "Run: python backend/app/models/train_ranker.py"
        )
    df = pd.read_csv(TRAIN_DATA)
    df["relevance"] = df["relevance"].round().clip(0, 5).astype(int)
    print(f"✅ Loaded: {len(df)} rows, {df['query_id'].nunique()} queries")
    return df


def query_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    queries = df["query_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(queries)
    split    = int(len(queries) * (1 - test_frac))
    train_q  = set(queries[:split])
    test_q   = set(queries[split:])
    return (
        df[df["query_id"].isin(train_q)].reset_index(drop=True),
        df[df["query_id"].isin(test_q)].reset_index(drop=True),
    )


# ============================================================================
# METRICS
# ============================================================================

def ndcg_per_query(y_true: pd.Series, y_pred: np.ndarray,
                   query_ids: pd.Series, k: int = 10) -> List[float]:
    scores = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        t = y_true[m].values.reshape(1, -1)
        p = y_pred[m].reshape(1, -1)
        scores.append(ndcg_score(t, p, k=k))
    return scores


def precision_at_k(y_true, y_pred, query_ids, k=5, threshold=3):
    scores = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tv = y_true[m].values
        pv = y_pred[m]
        top = np.argsort(pv)[::-1][:k]
        rel = (tv >= threshold)
        if rel.sum() == 0:
            continue
        scores.append(rel[top].sum() / min(k, len(top)))
    return float(np.mean(scores)) if scores else 0.0


def mean_ap(y_true, y_pred, query_ids, threshold=3):
    ap_list = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tv = y_true[m].values
        pv = y_pred[m]
        order = np.argsort(pv)[::-1]
        rel   = (tv[order] >= threshold)
        if rel.sum() == 0:
            continue
        prec = []
        hits = 0
        for i, r in enumerate(rel):
            if r:
                hits += 1
                prec.append(hits / (i + 1))
        ap_list.append(np.mean(prec) if prec else 0.0)
    return float(np.mean(ap_list)) if ap_list else 0.0


def mean_reciprocal_rank(y_true, y_pred, query_ids, threshold=3):
    rr_list = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tv = y_true[m].values
        pv = y_pred[m]
        order = np.argsort(pv)[::-1]
        for rank, idx in enumerate(order, start=1):
            if tv[idx] >= threshold:
                rr_list.append(1.0 / rank)
                break
        else:
            rr_list.append(0.0)
    return float(np.mean(rr_list)) if rr_list else 0.0


def cvr(df_test, y_pred, query_ids, top_k=5):
    constraint_cols = ["price_match", "location_match", "cert_match"]
    violations = []
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
                for c in constraint_cols
                if row.get(c, 0.5) != 0.5
            )
            violations.append(int(failed))
    return float(np.mean(violations)) if violations else 0.0


def avg_tau(y_true, y_pred, query_ids):
    taus = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tau, _ = kendalltau(y_true[m].values, y_pred[m])
        if not np.isnan(tau):
            taus.append(tau)
    return float(np.mean(taus)) if taus else 0.0


def full_eval(name, y_true, y_pred, df_test, query_ids):
    ndcg_scores = ndcg_per_query(y_true, y_pred, query_ids)
    return {
        "System":    name,
        "NDCG@10":   round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0.0,
        "P@5":       round(precision_at_k(y_true, y_pred, query_ids), 4),
        "MAP":       round(mean_ap(y_true, y_pred, query_ids), 4),
        "MRR":       round(mean_reciprocal_rank(y_true, y_pred, query_ids), 4),
        "CVR":       round(cvr(df_test, y_pred, query_ids), 4),
        "Kendall-τ": round(avg_tau(y_true, y_pred, query_ids), 4),
        "_ndcg_per_query": ndcg_scores,   # kept for significance testing
    }


# ============================================================================
# BASELINE SCORERS
# ============================================================================

def score_sourceup(df_test: pd.DataFrame) -> np.ndarray:
    """S1: SourceUp Full Model (LightGBM LambdaRank)."""
    if not os.path.exists(LGBM_PATH):
        print("   ⚠️  LightGBM model not found — using rule-based fallback for S1")
        return score_rule_based_default(df_test)
    with open(LGBM_PATH, "rb") as f:
        model = pickle.load(f)
    return model.predict(df_test[FEATURE_COLS])


def score_rule_based_default(df: pd.DataFrame) -> np.ndarray:
    """B3 / Fallback rule-based scorer matching ranker.py."""
    return (
        df["price_match"]          * 0.35 +
        (1 - df["price_distance"]) * 0.10 +
        df["location_match"]       * 0.20 +
        df["cert_match"]           * 0.20 +
        df["years_normalized"]     * 0.05 +
        df["is_manufacturer"]      * 0.05 +
        df["faiss_score"]          * 0.05
    ).values


def score_bm25(df_train: pd.DataFrame, df_test: pd.DataFrame) -> np.ndarray:
    """
    B1: BM25 baseline.
    Treats each supplier's product name as a document and each test
    product name as the query. Returns BM25 scores (normalised to [0,1]).
    Falls back to TF-IDF cosine if rank_bm25 is not installed.
    """
    if "product name" not in df_test.columns:
        print("   ⚠️  'product name' column missing — returning uniform BM25 score")
        return np.full(len(df_test), 0.5)

    corpus = df_train["product name"].fillna("unknown").astype(str).tolist()
    queries = df_test["product name"].fillna("unknown").astype(str).tolist()

    if BM25_AVAILABLE:
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        raw = np.array([
            float(np.mean(bm25.get_scores(q.lower().split())))
            for q in queries
        ])
    elif TFIDF_AVAILABLE:
        vec  = TfidfVectorizer(max_features=10000, sublinear_tf=True)
        Xc   = vec.fit_transform(corpus)
        Xq   = vec.transform(queries)
        raw  = np.asarray(tfidf_cosine(Xq, Xc).mean(axis=1)).ravel()
    else:
        print("   ⚠️  Neither rank_bm25 nor TF-IDF available — random BM25 scores")
        return np.random.uniform(0.3, 0.7, len(df_test))

    if raw.max() > raw.min():
        raw = (raw - raw.min()) / (raw.max() - raw.min())
    return raw


def score_sbert_cosine(df_train: pd.DataFrame, df_test: pd.DataFrame) -> np.ndarray:
    """
    B2: Pure SBERT cosine similarity.
    No constraint engine — semantic relevance only.
    """
    if "product name" not in df_test.columns:
        print("   ⚠️  'product name' column missing — returning FAISS scores as proxy")
        return df_test["faiss_score"].values

    corpus  = df_train["product name"].fillna("").astype(str).tolist()
    queries = df_test["product name"].fillna("").astype(str).tolist()

    if SBERT_AVAILABLE:
        print("   Loading SBERT model (all-MiniLM-L6-v2)...")
        model   = SentenceTransformer("all-MiniLM-L6-v2")
        c_emb   = model.encode(corpus, show_progress_bar=False, convert_to_numpy=True)
        q_emb   = model.encode(queries, show_progress_bar=False, convert_to_numpy=True)
        sims    = cosine_similarity(q_emb, c_emb).max(axis=1)   # best match per test item
    elif TFIDF_AVAILABLE:
        print("   SBERT not available — using TF-IDF cosine as proxy")
        vec   = TfidfVectorizer(max_features=10000)
        Xc    = vec.fit_transform(corpus)
        Xq    = vec.transform(queries)
        sims  = np.asarray(tfidf_cosine(Xq, Xc).max(axis=1)).ravel()
    else:
        print("   ⚠️  Fallback: using pre-computed FAISS scores from dataset")
        sims = df_test["faiss_score"].values

    if sims.max() > sims.min():
        sims = (sims - sims.min()) / (sims.max() - sims.min())
    return sims


def score_random(n: int, seed: int = 0) -> np.ndarray:
    """B4: Random ranking (theoretical lower bound)."""
    rng = np.random.default_rng(seed)
    return rng.random(n)


# ============================================================================
# STATISTICAL SIGNIFICANCE
# ============================================================================

def wilcoxon_vs_sourceup(sourceup_ndcg: List[float],
                          baseline_ndcg: List[float],
                          baseline_name: str) -> Dict:
    """
    Paired Wilcoxon signed-rank test (non-parametric, appropriate for
    per-query NDCG distributions which may not be normal).
    """
    n = min(len(sourceup_ndcg), len(baseline_ndcg))
    if n < 10:
        return {"baseline": baseline_name, "p_value": None, "significant": None,
                "mean_diff": 0.0, "n_queries": n}
    s = np.array(sourceup_ndcg[:n])
    b = np.array(baseline_ndcg[:n])
    try:
        stat, p = wilcoxon(s, b, alternative="greater")
    except Exception:
        stat, p = 0.0, 1.0
    return {
        "Baseline":       baseline_name,
        "Mean ΔNDCG":     round(float(np.mean(s - b)), 4),
        "p-value":        round(float(p), 4),
        "Significant":    "Yes (p<0.05)" if p < 0.05 else "No",
        "n queries":      n,
    }


# ============================================================================
# PLOTS
# ============================================================================

_PALETTE = {
    "S1: SourceUp Full":   "#2166ac",
    "B1: BM25":            "#d73027",
    "B2: SBERT Cosine":    "#f46d43",
    "B3: Rule-Based":      "#fdae61",
    "B4: Random":          "#cccccc",
}


def plot_comparison_bar(results_df: pd.DataFrame):
    """Grouped bar chart for all systems across all metrics."""
    metrics  = ["NDCG@10", "P@5", "MAP", "MRR"]
    systems  = results_df["System"].tolist()
    x        = np.arange(len(systems))
    width    = 0.18

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = ["#2166ac", "#4dac26", "#d01c8b", "#e6ab02"]

    for i, metric in enumerate(metrics):
        bars = ax.bar(
            x + i * width,
            results_df[metric].values,
            width, label=metric,
            color=palette[i], edgecolor="black", linewidth=0.5, alpha=0.9
        )
        for bar, val in zip(bars, results_df[metric].values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(systems, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Table II — Baseline Comparison: SourceUp vs Standard Retrieval Methods\n"
        "(S1 = SourceUp; B1–B4 = baselines)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Reference line at S1 NDCG
    s1_ndcg = results_df.loc[results_df["System"] == "S1: SourceUp Full", "NDCG@10"].values
    if len(s1_ndcg):
        ax.axhline(s1_ndcg[0], color="#2166ac", linestyle="--", linewidth=1.2, alpha=0.6,
                   label="S1 NDCG reference")

    plt.tight_layout()
    path = f"{PLOTS_DIR}/baseline_comparison_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_ndcg_per_query_distributions(system_ndcgs: Dict[str, List[float]]):
    """Box plot showing per-query NDCG distribution for each system."""
    systems  = list(system_ndcgs.keys())
    data_raw = [system_ndcgs[s] for s in systems]

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(
        data_raw, labels=systems, patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )
    colors = ["#2166ac", "#d73027", "#f46d43", "#fdae61", "#aaaaaa"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Per-Query NDCG@10", fontsize=11)
    ax.set_title(
        "Distribution of Per-Query NDCG@10 Across Systems\n"
        "(Higher median and tighter box → more consistent ranking quality)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xticklabels(systems, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/baseline_ndcg_per_query.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_significance_table(sig_df: pd.DataFrame):
    """Render statistical significance results as a table figure."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    cell_text   = [[str(v) for v in row] for row in sig_df.values]
    cell_colors = [["white"] * len(sig_df.columns) for _ in range(len(sig_df))]
    for i, row in sig_df.iterrows():
        if row.get("Significant", "No") == "Yes (p<0.05)":
            cell_colors[i] = ["#d4edda"] * len(sig_df.columns)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=sig_df.columns.tolist(),
        cellColours=cell_colors,
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#ddeeff")

    ax.set_title(
        "Statistical Significance: Wilcoxon Signed-Rank Test "
        "(SourceUp vs each baseline; one-sided, alternative='greater')\n"
        "Green = p < 0.05",
        fontsize=10, fontweight="bold", pad=12
    )
    plt.tight_layout()
    path = f"{PLOTS_DIR}/baseline_significance_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def run_baselines():
    print("=" * 65)
    print("📊 SourceUp — Baseline Comparison Study")
    print("=" * 65)

    df = load_data()
    df_train, df_test = query_split(df, test_frac=0.2, seed=42)
    y_test      = df_test["relevance"]
    query_test  = df_test["query_id"]

    print(f"  Train: {len(df_train)} rows | Test: {len(df_test)} rows")
    print()

    results      = []
    ndcg_dists   = {}

    # ------------------------------------------------------------------
    # S1: SourceUp Full Model
    # ------------------------------------------------------------------
    print("▶ S1: SourceUp Full Model")
    pred_s1 = score_sourceup(df_test)
    r = full_eval("S1: SourceUp Full", y_test, pred_s1, df_test, query_test)
    ndcg_dists["S1: SourceUp Full"] = r.pop("_ndcg_per_query")
    results.append(r)
    print(f"   NDCG@10 = {r['NDCG@10']:.4f}")

    # ------------------------------------------------------------------
    # B1: BM25
    # ------------------------------------------------------------------
    print("▶ B1: BM25 Baseline")
    pred_b1 = score_bm25(df_train, df_test)
    r = full_eval("B1: BM25", y_test, pred_b1, df_test, query_test)
    ndcg_dists["B1: BM25"] = r.pop("_ndcg_per_query")
    results.append(r)
    print(f"   NDCG@10 = {r['NDCG@10']:.4f}")

    # ------------------------------------------------------------------
    # B2: SBERT Cosine Similarity
    # ------------------------------------------------------------------
    print("▶ B2: SBERT Cosine Similarity")
    pred_b2 = score_sbert_cosine(df_train, df_test)
    r = full_eval("B2: SBERT Cosine", y_test, pred_b2, df_test, query_test)
    ndcg_dists["B2: SBERT Cosine"] = r.pop("_ndcg_per_query")
    results.append(r)
    print(f"   NDCG@10 = {r['NDCG@10']:.4f}")

    # ------------------------------------------------------------------
    # B3: Rule-Based Ranker
    # ------------------------------------------------------------------
    print("▶ B3: Rule-Based Ranker")
    pred_b3 = score_rule_based_default(df_test)
    r = full_eval("B3: Rule-Based", y_test, pred_b3, df_test, query_test)
    ndcg_dists["B3: Rule-Based"] = r.pop("_ndcg_per_query")
    results.append(r)
    print(f"   NDCG@10 = {r['NDCG@10']:.4f}")

    # ------------------------------------------------------------------
    # B4: Random
    # ------------------------------------------------------------------
    print("▶ B4: Random Ranking (lower bound)")
    pred_b4 = score_random(len(df_test))
    r = full_eval("B4: Random", y_test, pred_b4, df_test, query_test)
    ndcg_dists["B4: Random"] = r.pop("_ndcg_per_query")
    results.append(r)
    print(f"   NDCG@10 = {r['NDCG@10']:.4f}")

    # ------------------------------------------------------------------
    # Compile results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df["ΔNDCG vs B3"] = (
        results_df["NDCG@10"] -
        results_df.loc[results_df["System"] == "B3: Rule-Based", "NDCG@10"].values[0]
    ).round(4)

    csv_path = f"{OUT_DIR}/baseline_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved: {csv_path}")

    print("\n" + "=" * 65)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 65)
    display_df = results_df.drop(columns=["ΔNDCG vs B3"], errors="ignore")
    print(display_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Statistical significance
    # ------------------------------------------------------------------
    print("\n📈 Statistical Significance (Wilcoxon signed-rank, SourceUp > Baseline):")
    sourceup_ndcg = ndcg_dists["S1: SourceUp Full"]
    sig_rows = []
    for bname in ["B1: BM25", "B2: SBERT Cosine", "B3: Rule-Based", "B4: Random"]:
        sig_rows.append(wilcoxon_vs_sourceup(sourceup_ndcg, ndcg_dists[bname], bname))
    sig_df = pd.DataFrame(sig_rows)
    print(sig_df.to_string(index=False))
    sig_df.to_csv(f"{OUT_DIR}/significance_results.csv", index=False)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n📊 Generating plots...")
    plot_comparison_bar(results_df)
    plot_ndcg_per_query_distributions(ndcg_dists)
    plot_significance_table(sig_df)

    print(f"\n✅ All baseline outputs saved in: {OUT_DIR}")
    return results_df, sig_df


if __name__ == "__main__":
    run_baselines()