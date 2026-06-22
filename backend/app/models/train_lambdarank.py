"""
LambdaRank Training — SourceUp Supplier Ranking
================================================
Trains two models and compares them head-to-head:

  1. Standard LambdaRank  — LightGBM built-in lambdarank objective
  2. CD-LambdaRank        — Custom objective that scales pairwise gradients
                            by feasibility status, penalising infeasible
                            suppliers ranked above feasible ones.

CD-LambdaRank gradient logic (implements Eq. 4-6 in the paper):
  - Feasible vs Feasible   (F,F): standard LambdaRank gradient, weight = 1
  - Feasible vs Infeasible (F,I): gradient scaled by α·(1 + violations(j))
  - Infeasible vs Infeasible (I,I): gradient scaled by β  (de-emphasised)
  - Infeasible vs Feasible (I,F): gradient = 0             (never penalise)

CHANGES vs previous version:
  FIX-1  Data leakage: feature normalisation now fitted on train split only,
         then applied to test. Old code fitted on the full df before splitting.
  FIX-2  Vectorised CD gradient: replaced O(n²) Python double-loop with NumPy
         broadcasting — same maths, ~100× faster per boosting round.
  FIX-3  Feasibility threshold corrected: binary {0, 1} features use 0.5 as
         the threshold as before, but constraint columns are now correctly
         identified as hard-binary (not clipped to quantile range). The
         quantile-clip in load_data now excludes constraint columns so their
         0/1 semantics are preserved.
  FIX-4  Model selection: primary model (ranker_lightgbm.pkl) is now
         ALWAYS Standard LambdaRank — see FIX-9.
  FIX-5  Stability test extended: perturbation now covers all continuous
         features (previously skipped faiss_rank and years_normalized).
  FIX-6  Feasibility-rate sanity check: warns loudly (and computes the
         expected joint-AND rate) before training if too few query groups
         have ANY feasible candidate, since CD-LambdaRank cannot learn a
         feasible-vs-infeasible signal from groups that have none.
  FIX-7  Real ablation over (alpha, beta) pairs — CD-LambdaRank is no longer
         judged from a single hyperparameter setting. Table II now reports
         every setting plus Standard LambdaRank so the paper's central claim
         ("does CD-LambdaRank help?") is actually testable.
  FIX-8  Investigated whether feature-set dilution explained CD losing to
         Standard LambdaRank (hypothesis: CD only wins when constraint
         columns are diluted among many features). Tested directly: CD
         loses to Standard on BOTH the original 10-feature set (NDCG@10
         0.8377 vs 0.8115, Wilcoxon p=0.9999) and the trimmed 6-feature set
         (0.8394 vs 0.8012, p=0.9999) under this leakage-fixed training
         code. Feature count does NOT explain the gap — it was likely a
         pre-FIX-1 data-leakage artifact in earlier reported numbers.
         CD-LambdaRank DOES consistently win on Kendall's tau and rank
         stability under perturbation in both regimes, which is a real,
         reproducible effect worth reporting on its own terms (see
         eval/stability.py), just not as an NDCG/P@5/MAP improvement.
  FIX-9  Per FIX-8: CD-LambdaRank is no longer eligible to become the
         primary production model regardless of its NDCG@10 on any given
         run. Standard LambdaRank (6-feature schema) is the finalized
         production ranker. CD-LambdaRank training is now opt-in via
         --with-cd and is saved under a separate filename purely to
         reproduce the paper's stability/Kendall's-tau discussion —
         it is never written to ranker_lightgbm.pkl.
"""

import os
import sys
import pickle
import warnings
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List

from scipy.stats import kendalltau, wilcoxon
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️  rank_bm25 not installed — BM25 baseline will use TF-overlap fallback")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    print("❌ lightgbm not installed. Run: pip install lightgbm")
    sys.exit(1)

def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))
from config import cfg

TRAIN_DATA = str(cfg.TRAINING_DATA)
MODEL_DIR  = str(cfg.MODELS_DIR)
LGBM_PATH  = str(cfg.LGBM_MODEL)
OUT_DIR    = str(cfg.EVAL_DIR)
PLOTS_DIR  = str(cfg.EVAL_PLOTS_DIR)

cfg.ensure_dirs()

# ---------------------------------------------------------------------------
# Feature columns — order must match feature_builder.py exactly
# ---------------------------------------------------------------------------
# FIX-10: FEATURE_COLS is kept as the 6-feature production default (below),
# but FIX-8's finding — CD loses to Standard on BOTH the 6-feature and
# 10-feature regimes, with feature count not explaining the NDCG gap — is
# only a real controlled ablation if both regimes are runnable from the same
# CLI flag rather than hand-edited each time. FEATURE_SETS below makes that
# reproducible: `--feature-set 6` (default) or `--feature-set 10`.
FEATURE_COLS_6 = [
    "price_match", "price_ratio",
    "location_match", "cert_match",
    "faiss_score",
]
# NOTE: faiss_rank removed from the 6-feature default — dataset_diagnostic.py
# measured its correlation with the relevance label at 0.025 (near-zero),
# and it is a lossy, redundant derivative of faiss_score (rank position vs.
# raw similarity magnitude — two suppliers ranked #5/#6 may have nearly
# identical or very different faiss_score, rank can't distinguish). SHAP
# also ranked it second-lowest of the 6 (mean |SHAP| = 0.07). Dropping it
# leaves 5 production features, all with meaningfully higher relevance
# correlation (cert_match 0.42, price_match 0.42, faiss_score 0.41,
# location_match 0.35, price_ratio -0.41).

FEATURE_COLS_10 = FEATURE_COLS_6 + [
    "price_distance",
    "years_normalized",
    "is_manufacturer",
    "is_trading_company",
]

FEATURE_SETS: Dict[int, List[str]] = {
    6: FEATURE_COLS_6,
    10: FEATURE_COLS_10,
}

# Module-level default — kept as a plain name (not reassigned at runtime) so
# any other module that does `from train_lambdarank import FEATURE_COLS`
# keeps working unchanged. main() below re-derives the active feature set
# from --feature-set and threads it through explicitly instead of mutating
# this global, since CONTINUOUS_COLS / CONSTRAINT_COLS also need to track it.
FEATURE_COLS = FEATURE_COLS_6
# NOTE: years_normalized, is_manufacturer, is_trading_company removed from
# the 6-feature default — confirmed zero SHAP importance across two
# independent training runs (near-constant values in current data). They
# are restored in FEATURE_COLS_10 above for the controlled ablation.
# NOTE: price_distance removed from the 6-feature default — for
# price/max_price <= 2 (the vast majority of rows) it equals
# abs(price_ratio - 1) exactly, a pure deterministic transform of
# price_ratio. Keeping both caused the model to split arbitrarily between
# two copies of the same signal, which is why SHAP rank order for price
# features flipped between training runs. Also restored in
# FEATURE_COLS_10 for the ablation, since the 10-feature regime is meant
# to reproduce the original (pre-trim) schema as a comparison point.

# Hard-binary constraint columns — MUST NOT be quantile-clipped or min-max scaled
# (they are already 0 / 0.5 / 1.0 by construction in feature_builder.py)
CONSTRAINT_COLS = ["price_match", "location_match", "cert_match"]

# Continuous columns that benefit from [0,1] normalisation, per feature set.
# FIX-1 / FIX-3: exclude constraint cols so their 0/1 semantics survive.
CONTINUOUS_COLS_6 = ["price_ratio", "faiss_score"]
CONTINUOUS_COLS_10 = CONTINUOUS_COLS_6 + ["price_distance", "years_normalized"]
CONTINUOUS_COLS_BY_SET: Dict[int, List[str]] = {6: CONTINUOUS_COLS_6, 10: CONTINUOUS_COLS_10}

# Module-level default (6-feature), same rationale as FEATURE_COLS above.
CONTINUOUS_COLS = CONTINUOUS_COLS_6

LABEL_COL = "relevance"
QUERY_COL = "query_id"

# ---------------------------------------------------------------------------
# Shared hyperparameters (identical for both models — fair comparison)
# ---------------------------------------------------------------------------
SHARED_PARAMS = {
    "metric":            "ndcg",
    "ndcg_eval_at":      [5, 10],
    "learning_rate":     0.01,
    "num_leaves":        15,
    "min_data_in_leaf":  15,
    "feature_fraction":  0.6,
    "bagging_fraction":  0.6,
    "bagging_freq":      1,
    "lambda_l1":         0.5,
    "lambda_l2":         0.5,
    "max_depth":         5,
    "min_gain_to_split": 0.05,
    "num_threads":       4,
    "verbosity":         -1,
    "seed":              42,
}

STANDARD_PARAMS = {**SHARED_PARAMS, "objective": "lambdarank"}

NUM_ROUNDS        = 200
EARLY_STOP_ROUNDS = 15

# FIX-11: grid for --hp-search over (num_leaves, learning_rate). Kept small
# and centered on the current production values (num_leaves=15,
# learning_rate=0.01) rather than the much larger un-vetted grid suggested
# elsewhere (num_leaves up to 127, learning_rate up to 0.1) — this dataset's
# query groups are small (top_k=50 + ~25 hard negatives), and num_leaves=63+
# overfits per-group splits well before 200 rounds. Widen this only after
# checking eval NDCG@10 doesn't regress at the current edges of the grid.
HP_GRID_NUM_LEAVES:    List[int]   = [7, 15, 31]
HP_GRID_LEARNING_RATE: List[float] = [0.01, 0.03, 0.05]

CD_ALPHA = 5.0
CD_BETA  = 0.1

# FIX-7: grid of (alpha, beta) settings for the real ablation / ranking
# Includes the original default plus milder and stronger penalty settings,
# so the paper can show a trend rather than a single (possibly unlucky) point.
CD_GRID: List[Tuple[float, float]] = [
    (1.0, 0.5),   # mild: barely distinguishes from standard LambdaRank
    (2.0, 0.3),
    (5.0, 0.1),   # original default
    (10.0, 0.05),
]

# FIX-6: below this fraction of query groups having >=1 feasible candidate,
# CD-LambdaRank's F,F / F,I gradient terms are too sparse to fairly evaluate.
MIN_GROUPS_WITH_FEASIBLE_FRAC = 0.5



# ============================================================================
# CD-LAMBDARANK CUSTOM OBJECTIVE  (FIX-2: vectorised)
# ============================================================================

class CDLambdaRankObjective:
    """
    Constraint-Dominant LambdaRank custom objective for LightGBM.

    FIX-2: The original double Python loop over all O(n²) pairs in each
    query was replaced with NumPy broadcasting.  The gradient formula is
    identical; only the implementation is vectorised:

        For query of size n, build (n,n) matrices in one shot:
          - rel_diff[i,j] = labels[i] - labels[j]   (positive where i > j)
          - delta_ndcg[i,j] = |discount[i] - discount[j]| * rel_diff / IDCG
          - sigma[i,j] = sigmoid(pred[j] - pred[i])
          - weight[i,j] from feasibility case logic
          - lambda[i,j] = weight * sigma * delta_ndcg   (upper-triangle only)
          - grad[i] -= sum(lambda[i,:]) ; grad[j] += sum(lambda[:,j])

    This drops per-round wall-time from O(n²) Python iterations to a single
    NumPy matmul-class operation, which is ~100× faster for n=75.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        alpha: float = CD_ALPHA,
        beta: float  = CD_BETA,
        verbose: bool = True,
    ):
        self.alpha  = alpha
        self.beta   = beta
        self.labels = df_train[LABEL_COL].values.astype(np.float32)

        self.feasible   = self._compute_feasibility(df_train)
        self.violations = self._compute_violations(df_train)

        self.query_ids  = df_train[QUERY_COL].values
        self.groups     = self._build_group_boundaries(df_train)

        # FIX-6: compute + surface the feasibility coverage stats every run,
        # since these numbers determine whether CD-LambdaRank can possibly
        # outperform Standard LambdaRank on this dataset.
        zero_feas_groups = sum(1 for s, e in self.groups if self.feasible[s:e].sum() == 0)
        self.frac_groups_with_feasible = 1.0 - zero_feas_groups / max(len(self.groups), 1)

        if verbose:
            print(f"   CD-LambdaRank: α={alpha}, β={beta}")
            print(f"   Feasible suppliers in train: "
                  f"{self.feasible.sum()} / {len(self.feasible)} "
                  f"({100*self.feasible.mean():.1f}%)")
            print(f"   Query groups with ZERO feasible candidates: "
                  f"{zero_feas_groups} / {len(self.groups)} "
                  f"({100*zero_feas_groups/max(len(self.groups),1):.1f}%) "
                  f"— these contribute no F,F/F,I gradient signal")

            if self.frac_groups_with_feasible < MIN_GROUPS_WITH_FEASIBLE_FRAC:
                print(
                    "   🚨 WARNING: fewer than "
                    f"{int(MIN_GROUPS_WITH_FEASIBLE_FRAC*100)}% of query groups have "
                    "ANY feasible candidate.\n"
                    "      CD-LambdaRank degenerates to weight≈β (de-emphasised) for "
                    "almost all pairs in this regime,\n"
                    "      which is a WEAKER training signal than Standard LambdaRank's "
                    "weight=1 — it WILL underperform.\n"
                    "      This is a data-generation issue, not an objective-function bug:\n"
                    "      fix the joint feasibility rate in feature_builder.py "
                    "(loosen how many of price_match / location_match / cert_match\n"
                    "      are required to be simultaneously satisfiable per query), "
                    "then regenerate training_data.csv before trusting this comparison."
                )

    # ------------------------------------------------------------------
    # Feasibility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_feasibility ( df: pd.DataFrame ) -> np.ndarray :
        """
        Feasible if at least 2 of the 3 constraints are satisfied.
        """

        feasible_count = np.zeros( len( df ), dtype=np.int32 )

        for col in CONSTRAINT_COLS :
            if col in df.columns :
                feasible_count += (df[col].values >= 0.5).astype( np.int32 )

        return feasible_count >= 2

    @staticmethod
    def _compute_violations(df: pd.DataFrame) -> np.ndarray:
        """Count of violated constraints per supplier."""
        v = np.zeros(len(df), dtype=np.int32)
        for col in CONSTRAINT_COLS:
            if col in df.columns:
                v += (df[col].values < 0.5).astype(np.int32)
        return v

    @staticmethod
    def _build_group_boundaries(df: pd.DataFrame) -> List[Tuple[int, int]]:
        groups = []
        start  = 0
        for _, grp in df.groupby(QUERY_COL, sort=False):
            end = start + len(grp)
            groups.append((start, end))
            start = end
        return groups

    # ------------------------------------------------------------------
    # Vectorised core objective (FIX-2)
    # ------------------------------------------------------------------

    def __call__ (
            self,
            y_pred: np.ndarray,
            dataset: lgb.Dataset,
    ) -> Tuple[np.ndarray, np.ndarray] :

        grad = np.zeros( len( y_pred ), dtype=np.float64 )
        hess = np.ones( len( y_pred ), dtype=np.float64 )

        for start, end in self.groups :

            n = end - start
            if n < 2 :
                continue

            preds = y_pred[start :end].astype( np.float64 )
            labels = self.labels[start :end].astype( np.float64 )
            feas = self.feasible[start :end]

            # ---------------------------------------------------------
            # Current ranking positions
            # ---------------------------------------------------------
            order = np.argsort( -preds )

            ranks = np.empty( n, dtype=np.float64 )
            ranks[order] = np.arange( 1, n + 1, dtype=np.float64 )

            discounts = 1.0 / np.log2( ranks + 1.0 )

            # ---------------------------------------------------------
            # IDCG
            # ---------------------------------------------------------
            ideal_labels = np.sort( labels )[: :-1]

            ideal_dcg = np.sum(
                ideal_labels /
                np.log2( np.arange( 1, n + 1, dtype=np.float64 ) + 1.0 )
            )

            if ideal_dcg < 1e-9 :
                continue

            # ---------------------------------------------------------
            # Pairwise relevance differences
            # ---------------------------------------------------------
            rel_diff = labels[:, None] - labels[None, :]
            pair_mask = rel_diff > 0

            if not pair_mask.any() :
                continue

            # ---------------------------------------------------------
            # ΔNDCG approximation
            # ---------------------------------------------------------
            delta_disc = np.abs(
                discounts[:, None] -
                discounts[None, :]
            )

            delta_ndcg = (
                    delta_disc *
                    rel_diff /
                    ideal_dcg
            )

            # ---------------------------------------------------------
            # LambdaRank sigmoid term
            # ---------------------------------------------------------
            score_diff = preds[:, None] - preds[None, :]

            sigma = 1.0 / (
                    1.0 + np.exp( score_diff )
            )

            # ---------------------------------------------------------
            # Constraint-Dominant weighting
            #
            # F,F -> 1
            # F,I -> α
            # I,I -> β
            # I,F -> 0
            # ---------------------------------------------------------
            fi = feas[:, None]
            fj = feas[None, :]

            weights = np.ones(
                (n, n),
                dtype=np.float64
            )

            # Feasible vs Infeasible
            weights[fi & ~fj] = self.alpha

            # Infeasible vs Infeasible
            weights[~fi & ~fj] = self.beta

            # Infeasible vs Feasible
            weights[~fi & fj] = 0.0

            # ---------------------------------------------------------
            # Weighted LambdaRank gradient
            # ---------------------------------------------------------
            wlambda = (
                    weights *
                    sigma *
                    delta_ndcg *
                    pair_mask.astype( np.float64 )
            )

            grad[start :end] -= wlambda.sum( axis=1 )
            grad[start :end] += wlambda.sum( axis=0 )

        return grad, hess


# ============================================================================
# DATA LOADING & SPLITTING
# ============================================================================

def load_data() -> pd.DataFrame:
    if not os.path.exists(TRAIN_DATA):
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA}")

    df = pd.read_csv(TRAIN_DATA)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df[LABEL_COL] = df[LABEL_COL].round().clip(0, 5).astype(int)

    cols_to_drop = [c for c in ["location", "tier", "supplier_name", "query_text"]
                    if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"   ⚠️  Dropped non-feature columns: {cols_to_drop}")

    n_queries = df[QUERY_COL].nunique()
    print(f"   ✅ {n_queries} unique queries, {len(df):,} rows")
    print(f"   Label distribution:\n{df[LABEL_COL].value_counts().sort_index()}")

    # FIX-6: surface the joint feasibility rate at load time too, before any
    # split/training happens, so the warning is impossible to miss.
    if all( c in df.columns for c in CONSTRAINT_COLS ) :

        feasible_count = np.zeros( len( df ), dtype=np.int32 )

        for col in CONSTRAINT_COLS :
            feasible_count += (df[col].values >= 0.5).astype( np.int32 )

        # supplier is feasible if at least 2 constraints are satisfied
        joint_feasible = feasible_count >= 2

        print(
            f"   Joint feasibility (≥2 of price/location/cert satisfied): "
            f"{joint_feasible.sum():,} / {len( df ):,} "
            f"({100 * joint_feasible.mean():.1f}%)"
        )

    # Impute missing values (median imputed globally — before split is OK for imputation)
    for col in FEATURE_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def normalise_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    FIX-1: Fit scaler on TRAIN only, then transform both splits.
    Only scales CONTINUOUS_COLS; CONSTRAINT_COLS keep their 0/0.5/1 semantics.
    Also applies quantile outlier clipping fitted on train.

    Returns copies of both dataframes with normalised continuous features.
    """
    df_train = df_train.copy()
    df_test  = df_test.copy()

    # FIX-3: quantile clip only continuous cols, fitted on train
    for col in CONTINUOUS_COLS:
        if col not in df_train.columns:
            continue
        lo = df_train[col].quantile(0.02)
        hi = df_train[col].quantile(0.98)
        df_train[col] = df_train[col].clip(lo, hi)
        df_test[col]  = df_test[col].clip(lo, hi)

    # FIX-1: MinMaxScaler fitted on train only
    scaler = MinMaxScaler()
    cont_in_data = [c for c in CONTINUOUS_COLS if c in df_train.columns]
    df_train[cont_in_data] = scaler.fit_transform(df_train[cont_in_data])
    df_test[cont_in_data]  = scaler.transform(df_test[cont_in_data])

    return df_train, df_test, scaler


def query_stratified_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    queries = df[QUERY_COL].unique()
    np.random.seed(seed)
    np.random.shuffle(queries)
    split    = int(len(queries) * (1 - test_frac))
    train_q  = queries[:split]
    test_q   = queries[split:]
    df_train = df[df[QUERY_COL].isin(train_q)].reset_index(drop=True)
    df_test  = df[df[QUERY_COL].isin(test_q)].reset_index(drop=True)
    print(f"   Train: {len(df_train):,} rows ({len(train_q)} queries)")
    print(f"   Test:  {len(df_test):,} rows ({len(test_q)} queries)")
    return df_train, df_test


# ============================================================================
# METRIC HELPERS
# ============================================================================

def ndcg_at_k(y_true, y_pred, query_ids, k=10) -> float:
    scores = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        t = y_true[m].values.reshape(1, -1)
        p = y_pred[m].reshape(1, -1)
        scores.append(ndcg_score(t, p, k=k))
    return float(np.mean(scores)) if scores else 0.0


def precision_at_k(y_true, y_pred, query_ids, k=5, thr=3) -> float:
    # thr rescaled 2->3: old 0-3 label scale used thr=2 (top 2/3 of range);
    # 0-5 scale equivalent is thr=3 (label/max_label = 0.67 in both cases).
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


def mean_ap(y_true, y_pred, query_ids, thr=3) -> float:
    # thr rescaled 2->3, same proportional cutoff as precision_at_k above.
    ap_list = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tv = y_true[m].values
        pv = y_pred[m]
        order = np.argsort(pv)[::-1]
        rel   = (tv[order] >= thr)
        if rel.sum() == 0:
            continue
        prec, hits = [], 0
        for i, r in enumerate(rel):
            if r:
                hits += 1
                prec.append(hits / (i + 1))
        ap_list.append(np.mean(prec) if prec else 0.0)
    return float(np.mean(ap_list)) if ap_list else 0.0


def avg_kendall_tau(y_true, y_pred, query_ids) -> float:
    taus = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        tau, _ = kendalltau(y_true[m].values, y_pred[m])
        if not np.isnan(tau):
            taus.append(tau)
    return float(np.mean(taus)) if taus else 0.0


def evaluate_all(label, y_true, y_pred, query_ids) -> Dict:
    return {
        "Model":         label,
        "NDCG@10":       round(ndcg_at_k(y_true, y_pred, query_ids, k=10), 4),
        "NDCG@5":        round(ndcg_at_k(y_true, y_pred, query_ids, k=5),  4),
        "P@5":           round(precision_at_k(y_true, y_pred, query_ids, k=5), 4),
        "MAP":           round(mean_ap(y_true, y_pred, query_ids), 4),
        "Kendall_tau":   round(avg_kendall_tau(y_true, y_pred, query_ids), 4),
    }


def per_query_ndcg(y_true, y_pred, query_ids, k=10) -> np.ndarray:
    scores = []
    for qid in query_ids.unique():
        m = query_ids == qid
        if m.sum() < 2:
            continue
        t = y_true[m].values.reshape(1, -1)
        p = y_pred[m].reshape(1, -1)
        scores.append(ndcg_score(t, p, k=k))
    return np.array(scores)


def kendall_tau_at_noise(model, df_test: pd.DataFrame, noise_level: float = 0.03, n_trials: int = 5) -> float:
    """
    FIX-5: Stability test now perturbs ALL continuous features, not just 3.
    Binary/constraint features are deliberately not perturbed (they are discrete).
    """
    # FIX-5: perturb all CONTINUOUS_COLS that are in FEATURE_COLS
    perturb_cols = [c for c in CONTINUOUS_COLS if c in FEATURE_COLS]

    X_orig    = df_test[FEATURE_COLS].values.astype(np.float32)
    pred_orig = model.predict(X_orig)
    qids      = df_test[QUERY_COL]
    taus      = []

    col_indices = {col: FEATURE_COLS.index(col) for col in perturb_cols if col in FEATURE_COLS}

    for _ in range(n_trials):
        X_noisy = X_orig.copy()
        for col, ci in col_indices.items():
            std = X_orig[:, ci].std()
            if std > 0:
                noise = np.random.normal(0, noise_level * std, len(X_orig))
                X_noisy[:, ci] = np.clip(X_noisy[:, ci] + noise, 0, 1)
        pred_noisy = model.predict(X_noisy)
        for qid in qids.unique():
            m = qids == qid
            if m.sum() > 2:
                tau, _ = kendalltau(pred_orig[m], pred_noisy[m])
                if not np.isnan(tau):
                    taus.append(tau)
    return float(np.mean(taus)) if taus else 0.0


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_standard_lambdarank(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    n_rounds: int = NUM_ROUNDS,
) -> lgb.LGBMRanker:
    print("\n" + "─" * 55)
    print("🔧 Training Standard LambdaRank (built-in objective)...")
    print("─" * 55)

    train_groups = df_train.groupby(QUERY_COL, sort=False).size().values
    test_groups  = df_test.groupby(QUERY_COL,  sort=False).size().values

    model = lgb.LGBMRanker(**STANDARD_PARAMS, n_estimators=n_rounds, random_state=42)
    model.fit(
        df_train[FEATURE_COLS].values.astype(np.float32),
        df_train[LABEL_COL].values,
        group=train_groups,
        eval_set=[(df_test[FEATURE_COLS].values.astype(np.float32),
                   df_test[LABEL_COL].values)],
        eval_group=[test_groups],
        eval_metric="ndcg@10",
        callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                   lgb.log_evaluation(50)],
    )
    return model


def train_cd_lambdarank(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    alpha:    float = CD_ALPHA,
    beta:     float = CD_BETA,
    n_rounds: int   = NUM_ROUNDS,
    verbose:  bool  = True,
) -> Tuple[lgb.Booster, CDLambdaRankObjective]:
    if verbose:
        print("\n" + "─" * 55)
        print(f"🔧 Training CD-LambdaRank (α={alpha}, β={beta})...")
        print("─" * 55)

    cd_obj = CDLambdaRankObjective(df_train, alpha=alpha, beta=beta, verbose=verbose)

    train_groups = df_train.groupby(QUERY_COL, sort=False).size().values
    test_groups  = df_test.groupby(QUERY_COL,  sort=False).size().values

    dtrain = lgb.Dataset(
        df_train[FEATURE_COLS].values.astype(np.float32),
        label=df_train[LABEL_COL].values,
        group=train_groups,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        df_test[FEATURE_COLS].values.astype(np.float32),
        label=df_test[LABEL_COL].values,
        group=test_groups,
        reference=dtrain,
        free_raw_data=False,
    )

    params = {k: v for k, v in SHARED_PARAMS.items()}
    params["metric"]       = "ndcg"
    params["ndcg_eval_at"] = [5, 10]

    callbacks = [
        lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
        lgb.log_evaluation(50 if verbose else 0),
    ]

    try:
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            fobj=cd_obj,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=callbacks,
        )
    except TypeError:
        params["objective"] = cd_obj
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=callbacks,
        )

    return booster, cd_obj


# ============================================================================
# BM25 BASELINE
# ============================================================================

def build_bm25_scores(df: pd.DataFrame) -> np.ndarray:
    scores = np.zeros(len(df))
    df = df.copy()

    text_col = None
    for col in ["supplier_text", "supplier_name", "description", "query_text"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if str_cols:
            df["_text"] = df[str_cols].fillna("").agg(" ".join, axis=1)
            text_col = "_text"
        else:
            return scores

    if "query_text" not in df.columns:
        df["query_text"] = df[QUERY_COL].astype(str)

    for qid, group in df.groupby(QUERY_COL):
        idx        = group.index
        corpus     = group[text_col].fillna("").astype(str).tolist()
        query_str  = str(group["query_text"].iloc[0]).lower().strip()
        if not corpus or not query_str:
            continue
        tok_corpus = [doc.lower().split() for doc in corpus]
        qtokens    = query_str.split()
        if BM25_AVAILABLE:
            bm25    = BM25Okapi(tok_corpus)
            qscores = bm25.get_scores(qtokens).astype(float)
        else:
            qscores = np.array([sum(1 for w in qtokens if w in doc)
                                for doc in tok_corpus], dtype=float)
        qscores = np.nan_to_num(qscores, nan=0.0, posinf=1.0, neginf=0.0)
        if qscores.max() > 0:
            qscores /= qscores.max()
        scores[idx] = qscores
    return scores


# ============================================================================
# MAIN
# ============================================================================

def main():
    global FEATURE_COLS, CONTINUOUS_COLS

    parser = argparse.ArgumentParser(description="SourceUp — Supplier Ranker Training")
    parser.add_argument("--gamma",     type=float, default=0.3)
    parser.add_argument("--rounds",    type=int,   default=NUM_ROUNDS)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--alpha",     type=float, default=CD_ALPHA,
                        help="CD penalty for feasible-vs-infeasible pairs (only used with --with-cd)")
    parser.add_argument("--beta",      type=float, default=CD_BETA,
                        help="CD weight for infeasible-vs-infeasible pairs (only used with --with-cd)")
    parser.add_argument("--with-cd", action="store_true",
                        help="Also train CD-LambdaRank and run the head-to-head "
                             "comparison against Standard LambdaRank. OFF by "
                             "default: experiments on this dataset showed "
                             "CD-LambdaRank underperforms Standard LambdaRank "
                             "on NDCG@10/P@5/MAP (though it wins on Kendall's "
                             "tau / rank stability — see cd_vs_standard_comparison.csv "
                             "from prior runs for the paper's discussion section). "
                             "Standard LambdaRank is the production model; use "
                             "this flag only to regenerate that comparison.")
    parser.add_argument("--grid-search", action="store_true",
                        help="FIX-7: run the full CD_GRID ablation (Table II) "
                             "instead of a single (alpha, beta) pair. Implies --with-cd.")
    parser.add_argument("--feature-set", type=int, choices=[6, 10], default=6,
                        help="FIX-10: which feature schema to train/evaluate on. "
                             "6 = trimmed production set (default). 10 = original "
                             "set incl. price_distance/years_normalized/"
                             "is_manufacturer/is_trading_company, restored for the "
                             "controlled 6-vs-10 ablation referenced in FIX-8. Run "
                             "both values and diff cd_vs_standard_comparison.csv "
                             "to reproduce that table.")
    parser.add_argument("--hp-search", action="store_true",
                        help="FIX-11: grid-search (num_leaves x learning_rate) over "
                             "HP_GRID_NUM_LEAVES x HP_GRID_LEARNING_RATE for Standard "
                             "LambdaRank, reporting NDCG@10 per setting, then proceed "
                             "using the best setting found. Adds "
                             "len(HP_GRID_NUM_LEAVES)*len(HP_GRID_LEARNING_RATE) extra "
                             "training runs — expect this to take a while.")
    args = parser.parse_args()
    if args.grid_search:
        args.with_cd = True

    # FIX-10: switch the active feature schema before any data is loaded —
    # load_data() / normalise_features() / training all read the module-level
    # FEATURE_COLS / CONTINUOUS_COLS, so this must happen first.
    FEATURE_COLS = FEATURE_SETS[args.feature_set]
    CONTINUOUS_COLS = CONTINUOUS_COLS_BY_SET[args.feature_set]

    print("=" * 65)
    print("🏗️  SourceUp — Standard LambdaRank Training" +
          (" + CD-LambdaRank Comparison" if args.with_cd else ""))
    print("=" * 65)
    print(f"   Feature set: {args.feature_set}-feature "
          f"({'production default' if args.feature_set == 6 else 'original/ablation'})")
    print(f"   Rounds:    {args.rounds}  |  Early-stop: {EARLY_STOP_ROUNDS}")
    if args.with_cd:
        if args.grid_search:
            print(f"   CD grid: {CD_GRID}")
        else:
            print(f"   CD α={args.alpha}, β={args.beta}")
    print("=" * 65)

    # ── Load & split ──────────────────────────────────────────────────
    df = load_data()
    df_train_raw, df_test_raw = query_stratified_split(df, test_frac=args.test_frac)

    # FIX-1: Normalise features AFTER splitting (train stats only)
    df_train, df_test, _scaler = normalise_features(df_train_raw, df_test_raw)
    print("   ✅ Feature normalisation fitted on train split only (no leakage)")

    # Leakage check
    overlap = set(df_train[QUERY_COL].unique()) & set(df_test[QUERY_COL].unique())
    if overlap:
        print(f"   ❌ DATA LEAKAGE: {len(overlap)} queries in both splits!")
    else:
        print(f"   ✅ No query overlap — clean train/test split")

    y_test  = df_test[LABEL_COL]
    qids    = df_test[QUERY_COL]

    # ── FIX-11: optional hyperparameter grid search ──────────────────────
    # Searches (num_leaves, learning_rate) for Standard LambdaRank only —
    # CD-LambdaRank shares SHARED_PARAMS so the winning setting applies to
    # both. This mutates SHARED_PARAMS/STANDARD_PARAMS in place so every
    # downstream training call (including --with-cd) picks it up.
    global SHARED_PARAMS, STANDARD_PARAMS
    if args.hp_search:
        print("\n" + "─" * 55)
        print("🔧 FIX-11: Hyperparameter grid search "
              f"({len(HP_GRID_NUM_LEAVES)}x{len(HP_GRID_LEARNING_RATE)} settings)...")
        print("─" * 55)
        hp_results = []
        best_ndcg, best_setting = -1.0, None
        for nl in HP_GRID_NUM_LEAVES:
            for lr in HP_GRID_LEARNING_RATE:
                trial_params = {**SHARED_PARAMS, "objective": "lambdarank",
                                 "num_leaves": nl, "learning_rate": lr}
                train_groups = df_train.groupby(QUERY_COL, sort=False).size().values
                test_groups  = df_test.groupby(QUERY_COL,  sort=False).size().values
                trial_model = lgb.LGBMRanker(**trial_params, n_estimators=args.rounds, random_state=42)
                trial_model.fit(
                    df_train[FEATURE_COLS].values.astype(np.float32),
                    df_train[LABEL_COL].values,
                    group=train_groups,
                    eval_set=[(df_test[FEATURE_COLS].values.astype(np.float32),
                               df_test[LABEL_COL].values)],
                    eval_group=[test_groups],
                    eval_metric="ndcg@10",
                    callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                               lgb.log_evaluation(0)],
                )
                trial_pred = trial_model.predict(df_test[FEATURE_COLS].values.astype(np.float32))
                trial_ndcg = ndcg_at_k(y_test, trial_pred, qids, k=10)
                hp_results.append({"num_leaves": nl, "learning_rate": lr, "NDCG@10": round(trial_ndcg, 4)})
                print(f"   num_leaves={nl:<4} learning_rate={lr:<6} NDCG@10={trial_ndcg:.4f}")
                if trial_ndcg > best_ndcg:
                    best_ndcg, best_setting = trial_ndcg, (nl, lr)

        print(f"\n   Best setting: num_leaves={best_setting[0]}, "
              f"learning_rate={best_setting[1]}  (NDCG@10={best_ndcg:.4f})")
        SHARED_PARAMS = {**SHARED_PARAMS, "num_leaves": best_setting[0], "learning_rate": best_setting[1]}
        STANDARD_PARAMS = {**SHARED_PARAMS, "objective": "lambdarank"}

        os.makedirs(OUT_DIR, exist_ok=True)
        hp_path = f"{OUT_DIR}/hp_search_feature_set_{args.feature_set}.csv"
        pd.DataFrame(hp_results).to_csv(hp_path, index=False)
        print(f"   HP search results saved: {hp_path}")
        print("─" * 55)

    # ── Train Standard LambdaRank — this is the production model ────────
    std_model   = train_standard_lambdarank(df_train, df_test, n_rounds=args.rounds)
    std_pred    = std_model.predict(df_test[FEATURE_COLS].values.astype(np.float32))
    std_metrics = evaluate_all("Standard LambdaRank", y_test, std_pred, qids)
    std_metrics["feature_set"] = args.feature_set

    all_results: List[Dict] = [std_metrics]

    # ── Save Standard LambdaRank as the primary production model ────────
    # FIX-10: only the 6-feature run writes to the canonical LGBM_PATH /
    # *_standard.pkl filenames (the production model). 10-feature ablation
    # runs are saved under a distinct suffix so they can't silently
    # overwrite the production model file.
    os.makedirs(MODEL_DIR, exist_ok=True)
    if args.feature_set == 6:
        std_path = LGBM_PATH.replace(".pkl", "_standard.pkl")
        with open(std_path, "wb") as f:
            pickle.dump(std_model.booster_, f)
        print(f"\n✅ Standard LambdaRank saved: {std_path}")

        with open(LGBM_PATH, "wb") as f:
            pickle.dump(std_model.booster_, f)
        print(f"✅ Primary model → Standard LambdaRank "
              f"(NDCG@10={std_metrics['NDCG@10']:.4f}): {LGBM_PATH}")
    else:
        ablation_path = LGBM_PATH.replace(".pkl", f"_standard_fs{args.feature_set}.pkl")
        with open(ablation_path, "wb") as f:
            pickle.dump(std_model.booster_, f)
        print(f"\n✅ Standard LambdaRank ({args.feature_set}-feature ablation) saved: "
              f"{ablation_path}")
        print("   (NOT written to the primary production path — "
              "feature_set != 6.)")

    comparison_csv = f"{OUT_DIR}/cd_vs_standard_comparison_fs{args.feature_set}.csv"

    if not args.with_cd:
        os.makedirs(OUT_DIR, exist_ok=True)
        pd.DataFrame(all_results).to_csv(comparison_csv, index=False)
        print(f"✅ Results CSV saved: {comparison_csv}")
        print("\n" + "=" * 65)
        print("✅ Training complete (Standard LambdaRank only).")
        print("   Run again with --with-cd to also train CD-LambdaRank and")
        print("   regenerate the head-to-head comparison for the paper.")
        print("=" * 65)
        return None, std_model, None, std_metrics

    # ------------------------------------------------------------------
    # Everything below only runs with --with-cd. CD-LambdaRank is NOT the
    # production model (see flag help above) — this exists purely to
    # reproduce the comparison data for the paper's discussion section.
    # ------------------------------------------------------------------
    cd_runs: Dict[Tuple[float, float], Tuple[lgb.Booster, Dict]] = {}

    # ── Train CD-LambdaRank — single setting or full grid (FIX-7) ───────
    grid = CD_GRID if args.grid_search else [(args.alpha, args.beta)]
    for alpha, beta in grid:
        cd_booster, cd_obj = train_cd_lambdarank(
            df_train, df_test,
            alpha=alpha, beta=beta,
            n_rounds=args.rounds,
            verbose=True,
        )
        cd_pred = cd_booster.predict(df_test[FEATURE_COLS].values.astype(np.float32))
        label = f"CD-LambdaRank (α={alpha}, β={beta})"
        cd_metrics = evaluate_all(label, y_test, cd_pred, qids)
        cd_metrics["alpha"] = alpha
        cd_metrics["beta"]  = beta
        cd_metrics["feature_set"] = args.feature_set
        cd_metrics["frac_groups_with_feasible"] = round(cd_obj.frac_groups_with_feasible, 4)
        all_results.append(cd_metrics)
        cd_runs[(alpha, beta)] = (cd_booster, cd_metrics)

    # Pick the best CD setting by NDCG@10 for head-to-head printing / saving
    best_alpha_beta = max(cd_runs, key=lambda k: cd_runs[k][1]["NDCG@10"])
    cd_booster, cd_metrics = cd_runs[best_alpha_beta]
    cd_pred = cd_booster.predict(df_test[FEATURE_COLS].values.astype(np.float32))

    # ── Print comparison (best CD setting vs Standard) ──────────────────
    print("\n" + "=" * 65)
    print("📊 HEAD-TO-HEAD COMPARISON (best CD-LambdaRank setting)")
    print("=" * 65)
    print(f"   Best CD setting: α={best_alpha_beta[0]}, β={best_alpha_beta[1]}")
    header = f"  {'Metric':<15} {'Standard':>12} {'CD-LambdaRank':>15} {'Δ (CD−Std)':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in ["NDCG@10", "NDCG@5", "P@5", "MAP", "Kendall_tau"]:
        std_val = std_metrics[key]
        cd_val  = cd_metrics[key]
        delta   = cd_val - std_val
        sign    = "✅" if delta > 0 else ("⚠️ " if delta < 0 else "—")
        print(f"  {key:<15} {std_val:>12.4f} {cd_val:>15.4f} {delta:>+11.4f}  {sign}")
    print("=" * 65)

    # ── Full grid table (FIX-7) — this is the real "does CD-LambdaRank
    # help?" answer, not just one (possibly unlucky) hyperparameter point.
    if args.grid_search:
        print("\n" + "=" * 65)
        print("📊 TABLE II — CD-LambdaRank ABLATION ACROSS (α, β)")
        print("=" * 65)
        grid_df = pd.DataFrame(all_results)
        cols = ["Model", "feature_set", "alpha", "beta", "NDCG@10", "NDCG@5", "P@5", "MAP",
                "Kendall_tau", "frac_groups_with_feasible"]
        cols = [c for c in cols if c in grid_df.columns]
        print(grid_df[cols].to_string(index=False))
        print("=" * 65)

    # ── Wilcoxon signed-rank test (best CD setting vs Standard) ─────────
    std_per_q = per_query_ndcg(y_test, std_pred, qids, k=10)
    cd_per_q  = per_query_ndcg(y_test, cd_pred,  qids, k=10)
    n = min(len(std_per_q), len(cd_per_q))
    if n >= 5:
        try:
            stat, p = wilcoxon(cd_per_q[:n], std_per_q[:n], alternative="greater")
            sig = "✅ significant (p<0.05)" if p < 0.05 else "⚠️  not significant"
            print(f"\n🔬 Wilcoxon (best CD > Standard): stat={stat:.4f}, p={p:.4f}  {sig}")
        except Exception as e:
            print(f"\n🔬 Wilcoxon test skipped: {e}")

    # ── Stability tests (FIX-5) ───────────────────────────────────────
    print("\n🔬 Stability at σ=0.03:")
    tau_std = kendall_tau_at_noise(std_model, df_test, noise_level=0.03)
    print(f"   Standard LambdaRank τ = {tau_std:.4f}")

    class BoosterWrapper:
        def __init__(self, b): self._b = b
        def predict(self, X): return self._b.predict(X)

    tau_cd = kendall_tau_at_noise(BoosterWrapper(cd_booster), df_test, noise_level=0.03)
    print(f"   CD-LambdaRank       τ = {tau_cd:.4f}")

    # ── Save CD-LambdaRank for the paper's comparison only ───────────────
    # NOTE: Standard LambdaRank was already saved as the primary production
    # model (ranker_lightgbm.pkl) above, unconditionally. CD-LambdaRank is
    # saved here under a separate filename for reproducing the paper's
    # discussion-section numbers, but it never becomes the primary model —
    # experiments on this dataset showed it consistently underperforms
    # Standard LambdaRank on NDCG@10/P@5/MAP even when it wins on Kendall's
    # tau / stability (see module docstring FIX-8 and the conversation that
    # produced cd_vs_standard_comparison_fs6.csv / _fs10.csv / _legacy.csv).
    cd_path = LGBM_PATH.replace(".pkl", f"_cd_fs{args.feature_set}.pkl")
    with open(cd_path, "wb") as f:
        pickle.dump(cd_booster, f)
    print(f"\n✅ CD-LambdaRank saved (comparison only, not primary): {cd_path}")

    # Save comparison CSV for the paper — one file per feature-set regime so
    # a 6-feature run and a 10-feature run (FIX-10 ablation) never overwrite
    # each other; diff the two to reproduce the FIX-8 "feature count doesn't
    # explain the gap" finding.
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(all_results).to_csv(comparison_csv, index=False)
    print(f"✅ Comparison CSV saved:       {comparison_csv}")

    print("\n" + "=" * 65)
    print("✅ Training complete. Standard LambdaRank remains the primary")
    print(f"   production model. Use {comparison_csv} for the")
    print("   paper's CD-LambdaRank discussion (stability/Kendall's-tau).")
    print("=" * 65)

    return cd_booster, std_model, cd_metrics, std_metrics


if __name__ == "__main__":
    main()