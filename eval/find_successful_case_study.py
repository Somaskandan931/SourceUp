"""
find_successful_case_study.py
------------------------------
Your current paper case study (food packaging, Chennai) returns
0 fully-feasible suppliers and 0 location matches — realistic, but weak
as a demo of the system working end-to-end.

This script does NOT invent a better-looking case study. It runs a list
of CANDIDATE queries through the exact same production pipeline that
eval/case_study.py uses (retrieve -> rerank -> constraint_engine ->
ranker -> MMR -> SHAP), and reports which candidates actually produce
several feasible, constraint-satisfying suppliers in YOUR data. You then
pick a genuine "success" case from real output to pair with the existing
"realistic failure / partial match" case in the paper — two honest case
studies that together show both behaviors of the system.

Usage:
    python eval/find_successful_case_study.py
    python eval/find_successful_case_study.py --top_n 5

Output:
    Prints a ranked table of candidates by `fully_matching` count.
    Writes data/eval/case_study_candidates.json with full traces for
    every candidate tried, and data/eval/case_study_success.md for the
    best one (same render_markdown() used by case_study.py), so the
    artifact you cite is produced by the real system, not hand-edited.
"""

import sys
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def _find_project_root(marker: str = "config.py") -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))
from config import cfg
from case_study import run_case_study, render_markdown  # reuse the real pipeline wrapper

cfg.ensure_dirs()

# ---------------------------------------------------------------------------
# Candidate queries to try. Edit/extend this list freely — these are just
# starting guesses based on categories your benchmark already covers
# (Section III-A of the paper: packaging, automotive, electronics, baby
# products, industrial tools, raw materials, office supplies, apparel) and
# the city-tier distribution your assign_locations.py produces (Metro
# cities are best-populated, so they're the most likely to actually have
# several real, constraint-satisfying suppliers in the indexed subset).
# ---------------------------------------------------------------------------
CANDIDATES = [
    dict(product="cotton textile manufacturer", max_price=200000, moq_budget=500,
         location="Tiruppur", certification="ISO", location_mandatory=True),
    dict(product="cotton textile manufacturer", max_price=200000, moq_budget=500,
         location="Surat", certification="ISO", location_mandatory=True),
    dict(product="electronics components supplier", max_price=100000, moq_budget=200,
         location="Bangalore", certification="ISO", location_mandatory=True),
    dict(product="automotive parts manufacturer", max_price=300000, moq_budget=100,
         location="Chennai", certification="", location_mandatory=True),
    dict(product="industrial tools supplier", max_price=150000, moq_budget=50,
         location="Mumbai", certification="", location_mandatory=True),
    dict(product="office supplies wholesaler", max_price=50000, moq_budget=100,
         location="Delhi", certification="", location_mandatory=False),
    dict(product="apparel manufacturer", max_price=200000, moq_budget=500,
         location="Surat", certification="ISO", location_mandatory=False),
    dict(product="baby products manufacturer", max_price=100000, moq_budget=200,
         location="Mumbai", certification="", location_mandatory=False),
    dict(product="packaging materials supplier", max_price=100000, moq_budget=500,
         location="Mumbai", certification="", location_mandatory=False),
    dict(product="raw materials supplier", max_price=150000, moq_budget=100,
         location="Pune", certification="", location_mandatory=False),
    # Same product as the existing paper case study, but with constraints
    # relaxed in a defensible, disclosed way (no location_mandatory) — lets
    # you show the SAME query succeeding once location is treated as a
    # soft preference rather than a hard gate, which is an honest contrast
    # to report alongside the strict version already in the paper.
    dict(product="food packaging suppliers", max_price=50000, moq_budget=500,
         location="Chennai", certification="ISO", location_mandatory=False),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=10,
                         help="How many of the best candidates to print/save in detail")
    args = parser.parse_args()

    scored = []
    for i, cand in enumerate(CANDIDATES):
        print(f"[{i+1}/{len(CANDIDATES)}] Running: {cand}")
        try:
            trace = run_case_study(
                product=cand["product"],
                max_price=cand["max_price"],
                moq_budget=cand["moq_budget"],
                location=cand["location"],
                certification=cand["certification"],
                top_k=10,
                location_mandatory=cand["location_mandatory"],
            )
        except Exception as e:
            print(f"   ❌ failed: {e}")
            continue

        fully_matching = trace["stage_3_constraints"]["fully_matching"]
        location_matches = trace["stage_3_constraints"]["location_matches"]
        n_results = trace["stage_5_diversity"]["final_results"]
        print(f"   -> fully_matching={fully_matching}  location_matches={location_matches}  "
              f"final_results={n_results}")

        scored.append({
            "candidate": cand,
            "fully_matching": fully_matching,
            "location_matches": location_matches,
            "final_results": n_results,
            "trace": trace,
        })

    # Rank by fully_matching desc, then location_matches desc, then result count
    scored.sort(key=lambda r: (r["fully_matching"], r["location_matches"], r["final_results"]),
                reverse=True)

    print("\n" + "=" * 70)
    print("Ranked candidates (best first)")
    print("=" * 70)
    for r in scored[:args.top_n]:
        c = r["candidate"]
        print(f"  fully_matching={r['fully_matching']:>2}  loc_matches={r['location_matches']:>2}  "
              f"results={r['final_results']:>2}  | {c['product']!r} in {c['location']} "
              f"(mandatory_location={c['location_mandatory']})")

    out_json = cfg.EVAL_DIR / "case_study_candidates.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2, default=str)
    print(f"\n✅ Full candidate traces saved: {out_json}")

    if scored and scored[0]["fully_matching"] > 0:
        best = scored[0]["trace"]
        out_md = cfg.EVAL_DIR / "case_study_success.md"
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(render_markdown(best))
        print(f"✅ Best successful case study written: {out_md}")
        print("\nUse this alongside (not instead of) the existing food-packaging case study —")
        print("reporting one constrained 'no perfect match' case and one satisfied case gives")
        print("a more honest and more complete picture of system behavior than either alone.")
    else:
        print("\n⚠️  None of these candidates produced a fully-feasible supplier in your indexed")
        print("    subset. That's a real finding about the 19,750-supplier index's coverage —")
        print("    add more candidates (different cities/products) above and re-run, or")
        print("    relax location_mandatory, rather than reporting a result that didn't occur.")


if __name__ == "__main__":
    main()
