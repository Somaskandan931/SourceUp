# backend/app/training/weak_label_generator.py
"""
Weak Label Generator — SourceUp Supplier Ranking
-------------------------------------------------
SourceUp has no click logs or human-annotated relevance judgments
available at this stage. Rather than blocking training on the
collection of such a dataset, this module implements a **weak
supervision** strategy: relevance labels are generated automatically
from domain-informed procurement heuristics (semantic similarity,
price compatibility, location match, certification match, and
supplier tenure), combined with a small amount of injected noise to
emulate the imperfection of a real signal.

This module exists separately from `features/feature_builder.py` so
that the label-generation methodology is easy to find, review, and
cite from the paper's Methodology / Limitations sections.

Public API:
    generate_weak_labels(df)        -> df with weak_label_score / weak_label columns
    compute_weak_label(...)         -> per-row scalar scoring function
    load_weak_label_config()        -> dict loaded from configs/weak_labels.yaml
    save_weak_label_metadata(path)  -> writes reproducibility metadata JSON
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))

from config import cfg

LABEL_TYPE = "weak_supervision"
LABEL_METHOD = "heuristic"
LABEL_VERSION = "3.1"  # bumped: added feasibility gate to rebucket_by_quantile
                        # (v3.0's quantile fix solved label skew but let
                        # semantically-strong/constraint-violating rows into
                        # labels 4-5, raising CVR from ~0.37 to ~0.45-0.47
                        # reproducibly; the gate caps label tier by minimum
                        # constraint satisfaction).


def load_weak_label_config() -> dict:
    """Load weak-label weights/thresholds/noise params from YAML config."""
    config_path = cfg.WEAK_LABEL_CONFIG
    if not config_path.exists():
        # Sensible fallback so the pipeline still runs if the config is missing.
        return {
            "weak_labels": {
                "semantic_similarity": 0.25,
                "price_match": 0.20,
                "location_match": 0.20,
                "certification_match": 0.20,
                "years_on_platform": 0.10,
                "hidden_factor": 0.05,
            },
            "thresholds": {
                "label_5": 0.85, "label_4": 0.70, "label_3": 0.55,
                "label_2": 0.40, "label_1": 0.25,
            },
            "noise": {
                "hidden_factor_std": 0.30,
                "random_perturbation_prob": 0.15,
                "random_perturbation_std": 0.15,
            },
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_weak_label(
    semantic_similarity,
    price_match,
    location_match,
    certification_match,
    years_on_platform,
    config: dict | None = None,
):
    """
    Compute a single weakly-supervised relevance label (0-5) for one
    (query, supplier) pair from heuristic procurement signals.

    Returns:
        (weak_label_score, weak_label): the raw continuous score in
        [0, 1] and the bucketed ordinal label in {0, 1, 2, 3, 4, 5}.
    """
    cfg_data = config or load_weak_label_config()
    w = cfg_data["weak_labels"]
    thresholds = cfg_data["thresholds"]
    noise_cfg = cfg_data["noise"]

    hidden_factor = np.random.normal(0, noise_cfg["hidden_factor_std"])

    weak_label_score = (
        w["semantic_similarity"] * semantic_similarity +
        w["price_match"] * price_match +
        w["location_match"] * location_match +
        w["certification_match"] * certification_match +
        w["years_on_platform"] * years_on_platform +
        w["hidden_factor"] * hidden_factor
    )

    if np.random.rand() < noise_cfg["random_perturbation_prob"]:
        weak_label_score += np.random.normal(0, noise_cfg["random_perturbation_std"])

    weak_label_score = float(np.clip(weak_label_score, 0, 1))

    if weak_label_score >= thresholds["label_5"]:
        weak_label = 5
    elif weak_label_score >= thresholds["label_4"]:
        weak_label = 4
    elif weak_label_score >= thresholds["label_3"]:
        weak_label = 3
    elif weak_label_score >= thresholds["label_2"]:
        weak_label = 2
    elif weak_label_score >= thresholds["label_1"]:
        weak_label = 1
    else:
        weak_label = 0

    return weak_label_score, weak_label


def generate_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized convenience wrapper: given a DataFrame that already has
    columns `faiss_score` (or `semantic_score`), `price_match`,
    `location_match`, `cert_match`, and `years_normalized`, return a
    copy of df with `weak_label_score` and `weak_label` columns added.

    Row-by-row callers (e.g. feature_builder.build_training_data, which
    interleaves label generation with other per-candidate feature
    computation) can instead call `compute_weak_label(...)` directly
    per row using the same config.
    """
    config = load_weak_label_config()
    sem_col = "faiss_score" if "faiss_score" in df.columns else "semantic_score"

    scores, labels = [], []
    for _, row in df.iterrows():
        score, label = compute_weak_label(
            row[sem_col],
            row["price_match"],
            row["location_match"],
            row["cert_match"],
            row["years_normalized"],
            config=config,
        )
        scores.append(score)
        labels.append(label)

    out = df.copy()
    out["weak_label_score"] = scores
    out["weak_label"] = labels
    return out


def rebucket_by_quantile(
    scores: pd.Series,
    config: dict | None = None,
    feasibility_count: pd.Series | None = None,
) -> pd.Series:
    """
    Recompute ordinal weak labels (0-5) from the *empirical distribution*
    of weak_label_score, instead of the fixed absolute cutoffs used by
    compute_weak_label().

    WHY THIS EXISTS:
        compute_weak_label() buckets each row independently against fixed
        thresholds (e.g. score >= 0.85 -> label 5) on the assumption that
        weak_label_score is roughly uniform over [0, 1]. It isn't: most
        component signals (location_match, cert_match) are sparse/binary
        and faiss_score averages ~0.25, so the weighted composite rarely
        clears 0.70-0.85 even for rows that are the *best available*
        candidates for their query. Diagnosed via eval/dataset_diagnostic.py:
        labels {1,2} held 83%+ of rows, labels {4,5} held ~1.1%, even though
        nothing in the data is "near-constant" and faiss_score isn't simply
        re-deriving the label (correlation ~0.41-0.42, not dominant).

        Quantile-based bucketing fixes this by defining "label 5" as
        "top ~2% of candidates *for this dataset*" rather than "score above
        an absolute number nothing reaches". This preserves the original
        rank ordering of weak_label_score (a monotonic recalibration, not a
        new heuristic) while giving LambdaRank usable examples of every
        relevance tier.

    FEASIBILITY GATE (v3.1 fix):
        After deploying the quantile fix, eval/ablation.py and eval/
        baselines.py showed the Constraint Violation Rate (CVR) rose from
        ~0.36-0.38 (pre-relabel) to a stably-reproduced ~0.44-0.47
        (post-relabel, confirmed across two seeded reruns — not sampling
        noise). Root cause: faiss_score has much higher variance/spread
        than the sparse binary location_match/cert_match signals, so rows
        that are semantically excellent but constraint-mediocre can still
        land in the top weak_label_score quantiles, earning labels 4-5
        despite violating constraints. The fixed absolute thresholds
        (pre-v3.0) accidentally avoided this because few rows could reach
        0.70-0.85 *at all* without strong constraint signals contributing
        to the composite; quantile bucketing removed that side-effect
        along with the label-skew problem it was meant to fix.

        `feasibility_count` (0-3: how many of price_match/location_match/
        cert_match are >= 0.5 for that row) lets us require a minimum
        constraint-satisfaction floor per label tier. A row that scores in
        the top 2% of weak_label_score but satisfies 0 constraints gets
        downgraded until it reaches a label tier whose floor it actually
        meets — it does NOT get deleted or reweighted, just relabeled
        consistently with how feasible it actually is. This keeps the
        v3.0 label-balance fix intact while preventing label 5 from
        being awarded to constraint-violating rows.

    Args:
        scores: the full weak_label_score column (continuous, in [0, 1]),
                computed as before via compute_weak_label().
        config: weak_labels.yaml dict. Reads config["quantile_thresholds"]
                (cumulative percentiles) and config["feasibility_floor"]
                (minimum feasibility_count required per label, 0-3).
                Both fall back to sensible defaults if absent, so this is
                safe to call even on older configs.
        feasibility_count: optional pd.Series, same index as `scores`,
                with values in {0,1,2,3} = how many of
                price_match/location_match/cert_match are >= 0.5 for that
                row. If None, the feasibility gate is skipped entirely
                (pure quantile bucketing, pre-v3.1 behaviour).

    Returns:
        (labels, cutoffs): pd.Series of ints in {0..5} (same index as
        `scores`), and the dict of resolved absolute score cutoffs used.
    """
    cfg_data = config or load_weak_label_config()
    q = cfg_data.get("quantile_thresholds", {
        "label_5": 98, "label_4": 90, "label_3": 70,
        "label_2": 35, "label_1": 10,
    })
    floors = cfg_data.get("feasibility_floor", {
        "label_5": 2, "label_4": 2, "label_3": 1, "label_2": 0, "label_1": 0,
    })

    cutoffs = {
        label: float(np.percentile(scores, pct))
        for label, pct in q.items()
    }

    def _bucket(s):
        if s >= cutoffs["label_5"]:
            return 5
        elif s >= cutoffs["label_4"]:
            return 4
        elif s >= cutoffs["label_3"]:
            return 3
        elif s >= cutoffs["label_2"]:
            return 2
        elif s >= cutoffs["label_1"]:
            return 1
        return 0

    raw_labels = scores.apply(_bucket).astype(int)

    if feasibility_count is None:
        return raw_labels, cutoffs

    floor_by_label = {
        5: floors.get("label_5", 2),
        4: floors.get("label_4", 2),
        3: floors.get("label_3", 1),
        2: floors.get("label_2", 0),
        1: floors.get("label_1", 0),
        0: 0,
    }

    def _gate(label: int, feas: float) -> int:
        # Downgrade one tier at a time until the feasibility floor for the
        # current tier is met. feasibility_count can be NaN for rows where
        # constraint features weren't computed (defensive: treat as 0).
        feas = 0 if pd.isna(feas) else feas
        while label > 0 and feas < floor_by_label[label]:
            label -= 1
        return label

    gated_labels = pd.Series(
        [_gate(lbl, feas) for lbl, feas in zip(raw_labels, feasibility_count)],
        index=scores.index,
    ).astype(int)

    return gated_labels, cutoffs


def save_weak_label_metadata(
    path: Path | str | None = None,
    resolved_cutoffs: dict | None = None,
) -> Path:
    """
    Write a reproducibility metadata JSON describing the labeling run.

    Args:
        resolved_cutoffs: optional dict of the *actual* score cutoffs
            produced by rebucket_by_quantile() for this dataset (the
            quantile percentiles in weak_labels.yaml are relative, so the
            resulting absolute score cutoffs differ run-to-run as the
            underlying data changes; recording them here keeps the paper's
            methodology section reproducible).
    """
    config = load_weak_label_config()
    metadata = {
        "label_type": LABEL_TYPE,
        "method": LABEL_METHOD,
        "version": LABEL_VERSION,
        "weights": config["weak_labels"],
        "thresholds": config["thresholds"],
        "quantile_thresholds": config.get("quantile_thresholds"),
        "feasibility_floor": config.get("feasibility_floor"),
        "resolved_quantile_cutoffs": resolved_cutoffs,
        "noise": config["noise"],
    }
    out_path = Path(path) if path else cfg.WEAK_LABEL_METADATA
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=4)
    return out_path