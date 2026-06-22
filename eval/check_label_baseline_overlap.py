"""
check_label_baseline_overlap.py
--------------------------------
Quantifies how much the rule-based evaluation baseline (rule_baseline.py)
overlaps with the weak-supervision label formula (weak_label_generator.py
/ weak_labels.yaml) it's being compared against.

WHY THIS EXISTS:
    The weak labels used as ground truth throughout eval/ are generated
    from a weighted blend of semantic_similarity, price_match,
    location_match, certification_match, and years_on_platform, then
    quantile-rebucketed and feasibility-gated so that the top relevance
    tiers (4-5) REQUIRE >=2 of {price_match, location_match, cert_match}
    to be satisfied (see weak_labels.yaml's feasibility_floor).

    The rule-based baseline (B3 / V3 / V5 in baselines.py / ablation.py)
    scores candidates using price_match (0.40) + location_match (0.30)
    + cert_match (0.25) + faiss_score (0.05) — i.e. 95% of its weight is
    on the same three signals that gate the top label tiers.

    That overlap is a plausible structural reason B3/V3 outscore the
    full learned model (S1/V1) on NDCG@10 in your run (0.8720 vs 0.8505):
    the metric is, to some degree, rewarding agreement with the same
    signals that defined "relevant" in the first place, not purely
    "ability to find relevant suppliers."

    This script doesn't change any numbers — it gives you a citable,
    quantified overlap score to report transparently in the paper's
    Methodology or Limitations section, instead of presenting the
    rule-vs-ML comparison as a clean, independent result.

USAGE:
    python eval/check_label_baseline_overlap.py
"""

import sys
from pathlib import Path

import numpy as np
import yaml


def _find_project_root(marker: str = "config.py") -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))

from config import cfg
from rule_baseline import RULE_BASELINE_WEIGHTS


def load_label_weights() -> dict:
    """Pull the weak-label component weights from weak_labels.yaml,
    mapped onto the same column names rule_baseline.py uses."""
    with open(cfg.WEAK_LABEL_CONFIG, "r") as f:
        raw = yaml.safe_load(f)["weak_labels"]
    return {
        "faiss_score": raw.get("semantic_similarity", 0.0),
        "price_match": raw.get("price_match", 0.0),
        "location_match": raw.get("location_match", 0.0),
        "cert_match": raw.get("certification_match", 0.0),
        # years_on_platform / hidden_factor have no counterpart in the
        # rule baseline's feature set — they contribute to label
        # construction but the baseline can't "see" them either way, so
        # they're excluded from the comparison vector rather than
        # silently zero-padded into a misleading cosine score.
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def main():
    label_weights = load_label_weights()
    keys = list(label_weights.keys())

    label_vec = np.array([label_weights[k] for k in keys], dtype=np.float64)
    baseline_vec = np.array([RULE_BASELINE_WEIGHTS.get(k, 0.0) for k in keys], dtype=np.float64)

    # Renormalise each vector to sum to 1 over the shared key set, so the
    # comparison isn't skewed by years_on_platform/hidden_factor weight
    # that the label formula has but the baseline structurally can't.
    label_vec_norm = label_vec / label_vec.sum() if label_vec.sum() > 0 else label_vec
    baseline_vec_norm = baseline_vec / baseline_vec.sum() if baseline_vec.sum() > 0 else baseline_vec

    sim = cosine_similarity(label_vec_norm, baseline_vec_norm)
    # Fraction of each formula's weight sitting on the 3 constraint
    # signals (price/location/cert) vs. semantic similarity.
    constraint_keys = ["price_match", "location_match", "cert_match"]
    label_constraint_share = sum(label_weights[k] for k in constraint_keys) / label_vec.sum()
    baseline_constraint_share = sum(RULE_BASELINE_WEIGHTS.get(k, 0.0) for k in constraint_keys) / baseline_vec.sum()

    print("=" * 70)
    print("Label Formula vs. Rule-Baseline Overlap Check")
    print("=" * 70)
    print(f"{'Signal':<16}{'Label weight':>14}{'Baseline weight':>18}")
    for k in keys:
        print(f"{k:<16}{label_weights[k]:>14.3f}{RULE_BASELINE_WEIGHTS.get(k, 0.0):>18.3f}")
    print("-" * 70)
    print(f"Cosine similarity (renormalised over shared signals): {sim:.3f}")
    print(f"Label formula's weight on constraint signals (price/loc/cert): {label_constraint_share:.1%}")
    print(f"Baseline's weight on constraint signals (price/loc/cert):      {baseline_constraint_share:.1%}")
    print()
    if sim >= 0.7 or baseline_constraint_share >= 0.7:
        print("⚠️  HIGH OVERLAP: the rule-based baseline leans on the same")
        print("    signals used to gate top relevance labels. NDCG comparisons")
        print("    against this baseline should be reported with that caveat —")
        print("    do not present 'beats/loses to rule-based' as an independent")
        print("    validation of the learned ranker without disclosing this.")
    else:
        print("✅ Overlap below the 0.7 flag threshold.")
    print("=" * 70)


if __name__ == "__main__":
    main()
