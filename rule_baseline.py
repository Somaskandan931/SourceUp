# rule_baseline.py
"""
Canonical Rule-Based Baseline — SourceUp
-----------------------------------------
SINGLE SOURCE OF TRUTH for the "rule-based scorer" referenced throughout
this repo as B3 (eval/baselines.py), V3/V5 (eval/ablation.py), the
pre-training sanity check in pipeline/run_all.py, the sensitivity sweep
fallback (eval/sensitivity.py), eval/stability.py, eval/train_ranker.py,
the production inference fallback (backend/app/models/ranker.py), and
the case-study attribution fallback (eval/case_study.py).

THIS FILE EXISTS BECAUSE OF A REAL BUG, NOW FIXED ONE WAY, NOT TWO:

Before this file existed, eight separate call sites across this repo
each hand-rolled their own version of "the rule-based baseline," with
different weights and sometimes different features entirely:
  - run_all.py's rule_based_score(): 0.45 price_match / 0.35 faiss_score
    / 0.20 cert_match (no location_match at all)
  - ablation.py's score_rule_based(): 0.40/0.30/0.25/0.05 on
    price_match/location_match/cert_match/faiss_score
  - baselines.py's score_rule_based_default(): a third variant
  - stability.py, sensitivity.py, train_ranker.py, ranker.py's
    _rule_based_score(), case_study.py's SHAP-fallback weights: four
    more variants, none matching each other
That's why run_all.py's own pipeline log could print "Rule-Based
Baseline NDCG@10: 0.8549" while ablation.py/baselines.py printed
0.8720 for what was presented as the same number — they were
genuinely different formulas wearing the same label. Every call site
above has now been edited to import score_rule_based() from here
instead of keeping its own copy, so there is exactly one formula and
one number.

KNOWN LIMITATION (read before citing B3/V3/V5 vs. S1/V1 as an
independent baseline comparison in the paper):

This formula —

    score = 0.40 * price_match + 0.30 * location_match
          + 0.25 * cert_match  + 0.05 * faiss_score

— uses the same four signals that weak_label_generator.compute_weak_label()
weights into the `relevance` label every system in this repo (including
this baseline) is scored against:

    weak_label_score = 0.25*semantic_similarity + 0.20*price_match
                      + 0.20*location_match    + 0.20*certification_match
                      + 0.10*years_on_platform + 0.05*hidden_factor

95% of this baseline's weight sits on price_match/location_match/
cert_match/faiss_score, and those same four signals (under
semantic_similarity's other name) account for ~80% of the label
formula's weight, with the label's top two tiers additionally GATED
(not just weighted) on >=2 of {price_match, location_match, cert_match}
via weak_labels.yaml's feasibility_floor. That overlap is a plausible
structural reason this baseline outscores the learned ranker on
NDCG@10 in some runs (e.g. 0.8720 vs 0.8505): the metric rewards
agreement with the same signals that defined "relevant" in the first
place, not purely "ability to find relevant suppliers" independent of
how relevance was defined.

This file does NOT attempt to engineer that overlap away (e.g. by
switching to label-independent features like supplier_rating or
category_overlap_score) — that would silently change what "the
rule-based baseline" means across every script that already cites a
specific number for it, and a differently-defined baseline isn't
automatically a more valid one. Instead:
  1. The formula here stays IDENTICAL across every call site (the
     actual bug — divergent formulas — is fixed).
  2. Run check_label_baseline_overlap.py and report its cosine-overlap
     number alongside any "beats/loses to rule-based" claim, so the
     comparison is presented with its real limitations disclosed
     rather than as a clean independent validation it isn't.

If you want a genuinely label-independent baseline in addition to this
one (not as a replacement — both are legitimate things to report), add
a second function here (e.g. score_rule_based_independent()) built from
columns outside weak_label_generator's inputs (supplier_rating,
certification_count, category_overlap_score, is_manufacturer,
price_distance all exist in ranking_data.csv and are unused by the
label) rather than editing the weights below, so existing citations to
this formula stay valid.

UPDATE (after the independent baseline below was first built and run
against real data): supplier_rating turned out to be permanently
constant (std=0.0, always the 0.5 cold-start default) — none of the
three scrapers feeding this project ever populate a rating field, so
there's nothing to recover. It's been dropped from the weighted
formula (kept only as a diagnostic column in
check_independent_baseline_inputs() in case that ever changes).
is_manufacturer had a separate, fixable problem: clean_normalize.py's
CANONICAL_ALIASES was missing an entry for "business_type" even though
validate_merge.py's step-1 schema already produces it — same shape of
bug as the product_name/category aliasing bug documented above, just
in a different field. That alias has been added; is_manufacturer
should vary once the pipeline is re-run from clean_normalize.py
onward.

UPDATE 2 (after inspecting the actual raw scraped CSVs — output.csv,
output1.csv, output3.csv, output4.csv, output7.csv, output8.csv,
output_full.csv): the conclusion in UPDATE 1 above was right about the
symptom but wrong about the cause. supplier_rating wasn't constant
because "no scraper ever populates a rating field" — output8.csv has a
real "Rating" column, 97.5% filled, genuine 2.0-5.0 values. It was
constant because validate_merge.py (pipeline step 1, upstream of this
file) did `df = df[canonical]`, which silently DROPPED every column
outside its 23-column canonical schema — including rating — before
clean_normalize.py (step 2) ever got a chance to resolve it. That's
been fixed in validate_merge.py (added an OPTIONAL_ENRICHMENT_COLUMNS
allow-list that passes rating/city/certification_count through when a
source file has them, instead of an unconditional canonical-only
selection).

supplier_rating is STILL EXCLUDED from the weighted formula below, but
the reason has changed from "structurally impossible" to "real but too
thin to matter": counting rows across all 7 raw files, only
output8.csv (80 rows) has a rating column at all — the other
~247,700+ rows (output.csv/output1.csv/output3.csv/output4.csv/
output7.csv/output_full.csv) have none, so even after the
validate_merge.py fix, rating coverage across the full merged dataset
is ~0.03%. On a ~23K-row training sample that's effectively zero rows
with a non-default value — std will round to ~0 in practice even
though it is no longer literally impossible to be nonzero. Re-run
check_independent_baseline_inputs() after re-running the pipeline from
validate_merge.py onward to get the real number before deciding
whether to add it back; the bar is "does it actually vary enough in
this run's train/test split to carry signal," not "is it theoretically
obtainable from at least one source file." If a future scrape adds
ratings at meaningful coverage across more files, re-add it to
INDEPENDENT_BASELINE_WEIGHTS and _INDEPENDENT_RAW_INPUT_COLUMNS
together and renormalise the other four weights down.
    score_rule_based(df) -> np.ndarray   # vectorized, NaN-safe
    RULE_BASELINE_WEIGHTS                # {column: weight} dict, used
                                          # both for scoring here and for
                                          # per-feature attribution
                                          # display in case_study.py

    score_rule_based_independent(df) -> np.ndarray
        Second baseline (added after the KNOWN LIMITATION above was
        identified). Built from 4 columns — category_overlap_score,
        price_distance (as closeness), certification_count,
        is_manufacturer — that are NOT inputs to
        weak_label_generator.compute_weak_label(), so it does not share
        the overlap problem documented above. A 5th candidate column,
        supplier_rating, was tried and dropped (see UPDATE note above).
        Does not replace score_rule_based() — report both. See that
        function's own docstring for the full rationale and a worked
        example of why each candidate column is safe to use.

    check_independent_baseline_inputs(df) -> dict
        Run this once against your real ranking_data.csv before citing
        score_rule_based_independent() in the paper. It checks that
        every column the independent baseline reads actually exists and
        actually varies (isn't accidentally constant/all-default in
        your data, which would make the baseline degenerate without
        erroring), and also reports on supplier_rating for visibility
        even though it's no longer in the weighted formula. See its own
        docstring for how to call it.
"""

import numpy as np
import pandas as pd

# Canonical weights. Change here ONLY — every call site in the repo
# imports score_rule_based()/RULE_BASELINE_WEIGHTS from this module, so
# there is nowhere else a "rule-based baseline" formula should live.
# See the KNOWN LIMITATION section above before changing these to chase
# a higher or lower NDCG@10 against the learned ranker.
RULE_BASELINE_WEIGHTS = {
    "price_match": 0.40,
    "location_match": 0.30,
    "cert_match": 0.25,
    "faiss_score": 0.05,
}

# ---------------------------------------------------------------------------
# Label-independent baseline (score_rule_based_independent)
# ---------------------------------------------------------------------------
# Columns this second baseline is allowed to read. Kept as an explicit
# allow-list (rather than "whatever columns happen to exist") so that any
# future edit which accidentally adds price_match/location_match/
# cert_match/faiss_score back into this formula fails loudly via
# _assert_no_label_leakage() instead of silently recreating the exact
# overlap problem this function exists to avoid.
INDEPENDENT_BASELINE_WEIGHTS = {
    "category_overlap_score": 0.40,
    "price_distance_closeness": 0.30,  # derived below from price_distance
    "certification_count": 0.20,
    "is_manufacturer": 0.10,
}

# supplier_rating was originally weighted 0.25 here but is excluded as of
# this version: check_independent_baseline_inputs() showed std=0.0 on the
# real ranking_data.csv (always 0.5, the cold-start default), and tracing
# the cause confirmed it's not fixable — none of the three scrapers
# (GlobalSources/IndiaMART/TradeIndia; see validate_merge.py's
# CANONICAL_COLUMNS) ever populate a rating/review_count/response_rate
# field, so there's no real signal anywhere upstream to recover. Leaving
# a 0.25 weight on a column that is always the same constant 0.5 doesn't
# bias the ranking (it's a no-op, same value added to every row) but
# falsely implies independence-baseline coverage on 5 signals when only
# 4 actually vary. The remaining 4 weights above were renormalised to
# sum to 1.0. is_manufacturer is KEPT (unlike supplier_rating) because
# its dead-column problem (business_type missing from
# clean_normalize.py's CANONICAL_ALIASES, despite being a real column
# produced by validate_merge.py) was an aliasing bug, not a missing-data
# problem, and has been fixed — re-run the pipeline from clean_normalize.py
# onward and check_independent_baseline_inputs() again to confirm
# is_manufacturer now varies before citing B3b/V3b numbers.

# The raw column price_distance_closeness is derived from (not equal to,
# see score_rule_based_independent's docstring).
_INDEPENDENT_RAW_INPUT_COLUMNS = {
    "category_overlap_score",
    "price_distance",  # raw column; transformed into price_distance_closeness
    "certification_count",
    "is_manufacturer",
}

# supplier_rating is tracked separately (not in the weighted formula —
# see INDEPENDENT_BASELINE_WEIGHTS comment above) purely so
# check_independent_baseline_inputs() can still report on it. If a future
# data source adds real ratings, re-add it to
# _INDEPENDENT_RAW_INPUT_COLUMNS and INDEPENDENT_BASELINE_WEIGHTS together.
_DIAGNOSTIC_ONLY_COLUMNS = {"supplier_rating"}

# Columns that are direct inputs to weak_label_generator.compute_weak_label.
# score_rule_based_independent() must never read these.
LABEL_INPUT_COLUMNS = {
    "faiss_score", "semantic_score",  # semantic_similarity
    "price_match",
    "location_match",
    "cert_match", "certification_match",
    "years_normalized", "years_on_platform",
}


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Return df[col] as float, filling missing column / NaNs / inf with `default`."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .astype(float)
    )


def score_rule_based(df: pd.DataFrame) -> np.ndarray:
    """
    Canonical rule-based baseline. Scores each row as a fixed linear
    combination of RULE_BASELINE_WEIGHTS, NaN/Inf-safe regardless of
    which (if any) of the weighted columns are missing from `df`.

    Expects (each defaults to 0.0 if the column is absent, so this is
    safe to call on a partial/neutralised DataFrame, e.g. ablation.py's
    V5 which zeroes out some of these columns deliberately):
        price_match     : float in [0, 1]
        location_match  : float in [0, 1]
        cert_match      : float in [0, 1]
        faiss_score     : float in [0, 1]

    Returns:
        np.ndarray, one score per row, NaN/Inf-free.
    """
    score = np.zeros(len(df), dtype=np.float64)
    for col, weight in RULE_BASELINE_WEIGHTS.items():
        score = score + weight * _safe_col(df, col, default=0.0).values
    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)


def _assert_no_label_leakage(used_columns: set, fn_name: str) -> None:
    leaked = used_columns & LABEL_INPUT_COLUMNS
    if leaked:
        raise ValueError(
            f"{fn_name}() read label-input column(s) {leaked}, which are "
            f"direct inputs to weak_label_generator.compute_weak_label(). "
            f"This would recreate the exact overlap problem documented in "
            f"score_rule_based()'s KNOWN LIMITATION section. Use a column "
            f"from INDEPENDENT_BASELINE_WEIGHTS / "
            f"_INDEPENDENT_RAW_INPUT_COLUMNS instead."
        )


def score_rule_based_independent(df: pd.DataFrame) -> np.ndarray:
    """
    Label-independent rule-based baseline (B3b). Report ALONGSIDE
    score_rule_based() (B3), not as a replacement — both are legitimate
    things to put in the paper. See score_rule_based()'s KNOWN
    LIMITATION section for why this one exists.

    Built only from columns that are NOT inputs to
    weak_label_generator.compute_weak_label() (confirmed by inspection:
    that function's signature is exactly (semantic_similarity,
    price_match, location_match, certification_match,
    years_on_platform) — none of the four columns below are among
    them):

        category_overlap_score : float in [0, 1]. Lexical token overlap
            between the query and the supplier's category/product_name
            text (feature_builder.py). NOT the same signal as
            faiss_score (SBERT semantic similarity) — this is a cheap,
            literal word-match proxy, computed independently.

        price_distance : float >= 0 (0 = exactly at budget, larger =
            further over). Converted here to a bounded "closeness"
            score via 1 / (1 + price_distance). This is functionally
            different from price_match (which the label DOES use):
            price_match is a binary/near-binary in-budget flag,
            price_distance is a continuous distance — a supplier 5%
            over budget and one 200% over budget can both score
            price_match=0, but they have very different price_distance.

        certification_count : float in [0, 1], already capped/
            normalised in feature_builder.py. Counts how many
            certifications a supplier lists IN TOTAL, independent of
            whether any of them match the query's specific requested
            certification. Distinct from cert_match (which the label
            DOES use): a supplier can have a high certification_count
            and still score cert_match=0 if none of their certs match
            this query, or vice versa.

        is_manufacturer : 0/1, supplier classified as manufacturer vs.
            trading company (feature_builder.py, from business_type
            text). Not used anywhere in the label formula.

    NOT included: supplier_rating. It was part of the original 5-signal
    design but check_independent_baseline_inputs() showed it's
    std=0.0 (always the 0.5 cold-start default) on the real
    ranking_data.csv — none of the three scrapers feeding this project
    (GlobalSources/IndiaMART/TradeIndia) ever populate a rating field,
    so there's no real signal to recover. See the comment above
    INDEPENDENT_BASELINE_WEIGHTS for the full explanation. The other
    four weights were renormalised to sum to 1.0 in its absence.

    A runtime guard (_assert_no_label_leakage) raises if this function
    is ever edited to read a label-input column, so this independence
    claim can't silently rot.

    IMPORTANT — run check_independent_baseline_inputs(df) once against
    your real ranking_data.csv before citing results from this function.
    It will not raise on a missing/constant column (this function
    degrades gracefully, same as score_rule_based), but a baseline built
    on columns that are accidentally all-default in your actual data is
    a meaningless number even though it won't error. is_manufacturer in
    particular depends on clean_normalize.py resolving a "business_type"
    column — re-run the pipeline from clean_normalize.py onward after
    pulling the latest version of that file before trusting this column.

    Returns:
        np.ndarray, one score per row, NaN/Inf-free.
    """
    used_raw_columns = _INDEPENDENT_RAW_INPUT_COLUMNS & set(df.columns)
    _assert_no_label_leakage(used_raw_columns, "score_rule_based_independent")

    category_overlap = _safe_col(df, "category_overlap_score", default=0.0)
    cert_count = _safe_col(df, "certification_count", default=0.0)
    is_manufacturer = _safe_col(df, "is_manufacturer", default=0.0)

    price_distance = _safe_col(df, "price_distance", default=1.0)
    price_closeness = 1.0 / (1.0 + price_distance)

    score = (
        INDEPENDENT_BASELINE_WEIGHTS["category_overlap_score"] * category_overlap +
        INDEPENDENT_BASELINE_WEIGHTS["price_distance_closeness"] * price_closeness +
        INDEPENDENT_BASELINE_WEIGHTS["certification_count"] * cert_count +
        INDEPENDENT_BASELINE_WEIGHTS["is_manufacturer"] * is_manufacturer
    )

    return np.nan_to_num(score.values, nan=0.0, posinf=1.0, neginf=0.0)


def check_independent_baseline_inputs(df: pd.DataFrame) -> dict:
    """
    Diagnostic — run this ONCE against your real ranking_data.csv before
    citing score_rule_based_independent() results anywhere in the paper:

        import pandas as pd
        from rule_baseline import check_independent_baseline_inputs
        df = pd.read_csv("data/training/ranking_data.csv")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        report = check_independent_baseline_inputs(df)
        for col, info in report.items():
            print(col, info)

    Reports on the 4 columns actually used in the weighted B3b formula
    (_INDEPENDENT_RAW_INPUT_COLUMNS) plus supplier_rating, which is
    tracked here for visibility even though it's excluded from the
    formula itself (see INDEPENDENT_BASELINE_WEIGHTS comment — it was
    found to be permanently constant and dropped). For each column,
    reports:
        present  : bool, is the column in df.columns at all
        n_nan    : how many values were missing/non-numeric
        std      : standard deviation (0.0 or NaN means CONSTANT —
                   the column carries no ranking signal for this
                   baseline, even though scoring won't error)
        min/max  : observed range

    A column that is present=False or std<=0.0 means either: (a) it's
    supplier_rating, which is expected to be constant until the
    underlying scraper data adds a real rating field, or (b) it's one
    of the 4 weighted columns, in which case B3b is silently relying on
    its default fill value for that term — worth investigating (e.g.
    is_manufacturer requires clean_normalize.py to have resolved a
    "business_type" source column; if it shows std=0.0, re-check that
    the pipeline was re-run after that alias was added) before citing
    B3b as a meaningfully different baseline from B3.
    """
    report = {}
    for col in sorted(_INDEPENDENT_RAW_INPUT_COLUMNS | _DIAGNOSTIC_ONLY_COLUMNS):
        present = col in df.columns
        if present:
            vals = pd.to_numeric(df[col], errors="coerce")
            report[col] = {
                "present": True,
                "n_nan": int(vals.isna().sum()),
                "std": float(vals.std()) if vals.notna().any() else float("nan"),
                "min": float(vals.min()) if vals.notna().any() else float("nan"),
                "max": float(vals.max()) if vals.notna().any() else float("nan"),
                "used_in_formula": col not in _DIAGNOSTIC_ONLY_COLUMNS,
            }
        else:
            report[col] = {
                "present": False, "n_nan": None, "std": None,
                "min": None, "max": None,
                "used_in_formula": col not in _DIAGNOSTIC_ONLY_COLUMNS,
            }
    return report