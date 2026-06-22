"""
verify_location_match.py
-------------------------
Independent sanity check on case_study_candidates.json (produced by
find_successful_case_study.py).

WHY: the 'packaging materials supplier / Mumbai' candidate reported
location_matches=100 while every supplier in its own top-10 table is
listed as "China". That is not possible if location_matches is really
counting suppliers whose location matches the query. This script
recomputes the match count directly from the *displayed* location field
in top_results, completely independent of constraint_engine.py's
internal bookkeeping, so you can see whether the engine's reported
number can be trusted.

This does NOT fix anything — it just tells you, per candidate, whether
the engine's claimed location_matches lines up with the actual location
strings attached to the results it returned.

Usage:
    python eval/verify_location_match.py
    python eval/verify_location_match.py --path data/eval/case_study_candidates.json
"""

import sys
import json
import argparse
from pathlib import Path


def _find_project_root(marker: str = "config.py") -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


def honest_location_match_count(query_location: str, top_results: list) -> int:
    """Case-insensitive substring check: does this result's own displayed
    location string contain (or get contained by) the requested location?
    Deliberately simple/conservative — if anything it should UNDER-count
    relative to a real city-aware matcher, so any number it produces that's
    still suspiciously high relative to the data (e.g. matching 'Mumbai'
    against 'China') is real evidence of a problem, not a false positive
    from this checker.
    """
    if not query_location:
        return None  # no location was requested; not applicable
    q = query_location.strip().lower()
    count = 0
    for r in top_results:
        loc = str(r.get("location", "") or "").strip().lower()
        if q in loc or loc in q:
            count += 1
    return count


def main():
    sys.path.insert(0, str(_find_project_root()))
    from config import cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(cfg.EVAL_DIR / "case_study_candidates.json"))
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        scored = json.load(f)

    print("=" * 100)
    print(f"{'Query location':<14}{'Engine loc_matches':>20}{'Honest top-10 match':>22}   Product")
    print("=" * 100)

    flagged = []
    for r in scored:
        cand = r["candidate"]
        trace = r["trace"]
        q_loc = cand.get("location", "")
        top_results = trace.get("top_results", [])

        engine_claim = trace["stage_3_constraints"]["location_matches"]
        honest_top10 = honest_location_match_count(q_loc, top_results)

        flag = ""
        if honest_top10 is not None and engine_claim > 0 and honest_top10 == 0:
            flag = "  ⚠️  ENGINE SAYS MATCH, TOP-10 SHOWS NONE"
            flagged.append((cand["product"], q_loc, engine_claim, honest_top10))

        print(f"{q_loc:<14}{engine_claim:>20}{str(honest_top10):>22}   {cand['product']}{flag}")

    print("=" * 100)
    if flagged:
        print(f"\n⚠️  {len(flagged)} candidate(s) where the constraint engine reported location "
              f"matches but none of the actual top-10 results are located anywhere near the "
              f"requested location. This points to a bug in constraint_engine.py's location "
              f"check (likely auto-passing location when location_mandatory=False) rather than "
              f"a genuine match. Recommend inspecting/sharing constraint_engine.py before citing "
              f"any location_matches / CVR numbers that depend on it — including the ablation "
              f"(Table tab:ablation) and constraint-stress (Table tab:stress) results in the paper, "
              f"since they're computed by the same engine.")
    else:
        print("\n✅ No discrepancy detected between engine-reported and observed location matches "
              "in this candidate set.")


if __name__ == "__main__":
    main()
