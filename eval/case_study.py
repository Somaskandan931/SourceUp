"""
Procurement Case Study — SourceUp
-----------------------------------
Runs ONE realistic SME procurement query end-to-end through the actual
production pipeline (FAISS retrieval -> cross-encoder rerank -> constraint
engine -> LTR ranking -> MMR diversity -> SHAP explanation) and writes a
grounded, reproducible case study for the IEEE paper.

This is intentionally a thin wrapper around the real backend modules
(NOT a separate reimplementation), so the numbers reported in the paper
are guaranteed to match what the deployed system actually returns.

Outputs:
  data/eval/case_study.json   (full machine-readable trace)
  data/eval/case_study.md     (paper-ready narrative writeup)

Usage:
  python eval/case_study.py
  python eval/case_study.py --product "food packaging" --budget 50000 --moq 500 --location Chennai --certification ISO
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))
from config import cfg
from rule_baseline import RULE_BASELINE_WEIGHTS

from backend.app.models.retriever import retrieve, rerank_with_cross_encoder
from backend.app.models.constraint_engine import get_constraint_engine
from backend.app.models.ranker import get_ranker, extract_features_batch, apply_mmr, FEATURE_COLS

cfg.ensure_dirs()


def run_case_study(product: str, max_price: float, moq_budget: float,
                    location: str, certification: str, top_k: int = 10,
                    location_mandatory: bool = True) -> dict:

    trace = {"query": {
        "product": product,
        "max_price": max_price,
        "moq_budget": moq_budget,
        "location": location,
        "certification": certification,
        "location_mandatory": location_mandatory,
    }}

    # --- Stage 1: SBERT + FAISS retrieval ---------------------------------
    candidates = retrieve(product, k=100)
    trace["stage_1_retrieval"] = {"retrieved": len(candidates)}

    # --- Stage 2: Cross-encoder rerank (shortlist only) --------------------
    candidates = rerank_with_cross_encoder(product, candidates, top_n=100)
    trace["stage_2_cross_encoder"] = {"reranked": len(candidates)}

    # --- Stage 3: SME constraint filtering (soft, graduated penalties) ----
    # location_mandatory=True because this case study's whole premise is
    # "find me suppliers in <location>" — without it, location is just one
    # soft-weighted feature among several and can be silently overridden by
    # cert_match/faiss_score, producing top results nowhere near the
    # requested location while still being reported as "fully matching".
    constraint_engine = get_constraint_engine()
    constraints = {
        "max_price": max_price,
        "moq_budget": moq_budget,
        "preferred_location": location,
        "location_mandatory": location_mandatory,
        "required_certifications": [certification] if certification else [],
    }
    viable = constraint_engine.apply_constraints(candidates, constraints)
    filter_summary = constraint_engine.get_filter_summary()
    fully_matching = sum(1 for s in viable if not s.get("constraint_violated", False))
    # Honest location count, independent of the mandatory flag, so the
    # writeup never silently implies location was satisfied when it wasn't.
    location_matches = sum(
        1 for s in viable
        if s.get("constraint_results", {}).get("location", {}).get("passed", False)
    )
    trace["stage_3_constraints"] = {
        "after_constraints": len(viable),
        "fully_matching": fully_matching,
        "location_matches": location_matches,
        "filters_applied": filter_summary.get("filters_applied", []),
    }

    # --- Stage 4: LTR ranking ----------------------------------------------
    ranker = get_ranker()
    query_dict = {
        "product": product,
        "max_price": max_price,
        "location": (location or "").lower(),
        "certification": (certification or "").lower(),
    }
    ranked = ranker.rank(viable, query_dict)
    trace["stage_4_ranking"] = {
        "ranking_method": ranker.model_type if ranker.use_ml else "rule-based",
    }

    # --- Stage 5: MMR diversity --------------------------------------------
    diversified = apply_mmr(ranked, top_k=top_k, lambda_param=0.3)
    trace["stage_5_diversity"] = {"final_results": len(diversified[:top_k])}

    top_results = diversified[:top_k]

    # --- Stage 6: SHAP-style feature contribution for the #1 result -------
    top_supplier = top_results[0] if top_results else None
    shap_breakdown = None
    if top_supplier is not None:
        feats = extract_features_batch([top_supplier], query_dict)
        shap_breakdown = _approximate_contribution(ranker, feats)

    trace["top_results"] = [
        {
            "rank": i + 1,
            "supplier": s.get("supplier_name") or s.get("supplier name") or "Unknown",
            "product_name": s.get("product_name") or s.get("product name") or "",
            "price": s.get("price"),
            "location": s.get("supplier_location") or s.get("location"),
            "score": round(float(s.get("score", 0.0)), 4),
            "constraint_violated": s.get("constraint_violated", False),
        }
        for i, s in enumerate(top_results)
    ]
    trace["top_supplier_feature_contribution"] = shap_breakdown

    return trace


def _approximate_contribution(ranker, feats) -> dict:
    """
    Best-effort SHAP-style attribution for the case study writeup.
    Uses the real shap library against the loaded LightGBM model when
    available (matches eval/shap_analysis.py); falls back to the rule-based
    weighted contribution used in ranker.rule_based_score otherwise, so the
    case study still runs even without the ML model / shap installed.
    """
    try:
        if ranker.use_ml and ranker.model is not None:
            import shap
            explainer = shap.TreeExplainer(ranker.model)
            shap_values = explainer.shap_values(feats[FEATURE_COLS])
            row = shap_values[0] if hasattr(shap_values, "__len__") else shap_values
            return {col: round(float(val), 4) for col, val in zip(FEATURE_COLS, row)}
    except Exception:
        pass

    # Fallback: rule-based weights.
    # FIX: this previously claimed to "mirror ranker.rule_based_score" but
    # actually used its own 7-term weight set (price_match 0.25 /
    # price_distance -0.10 / location_match 0.15 / cert_match 0.15 /
    # years_normalized 0.10 / is_manufacturer 0.05 / faiss_score 0.20) —
    # an 8th divergent copy of the "rule-based" formula found in this
    # repo, none of which matched each other (see rule_baseline.py's
    # module docstring for the full inventory). Now uses
    # rule_baseline.RULE_BASELINE_WEIGHTS directly, so this case-study
    # writeup's attribution actually mirrors what ranker.py's production
    # fallback and the paper's NDCG tables describe.
    weights = RULE_BASELINE_WEIGHTS
    row = feats.iloc[0]
    return {k: round(float(row.get(k, 0.0)) * w, 4) for k, w in weights.items()}


def render_markdown(trace: dict) -> str:
    q = trace["query"]
    lines = [
        "# SourceUp Procurement Case Study\n",
        "## Input\n",
        "```text",
        f"Need: {q['product']}",
        f"Budget: ₹{q['max_price']}" if q["max_price"] else "Budget: (none specified)",
        f"MOQ budget: {q['moq_budget']}" if q["moq_budget"] else "",
        f"Location: {q['location']}" if q["location"] else "",
        f"Certification: {q['certification']}" if q["certification"] else "",
        "```\n",
        "## Pipeline Trace\n",
        f"- Retrieved (FAISS top-k): {trace['stage_1_retrieval']['retrieved']}",
        f"- After cross-encoder rerank: {trace['stage_2_cross_encoder']['reranked']}",
        f"- After constraints: {trace['stage_3_constraints']['after_constraints']} "
        f"(fully matching: {trace['stage_3_constraints']['fully_matching']}, "
        f"location matching: {trace['stage_3_constraints']['location_matches']})",
        f"- Filters applied: {', '.join(trace['stage_3_constraints']['filters_applied']) or 'none'}",
        f"- Ranking method: {trace['stage_4_ranking']['ranking_method']}",
        f"- Final diversified results: {trace['stage_5_diversity']['final_results']}\n",
        "## Top Results\n",
        "| Rank | Supplier | Product | Price | Location | Score |",
        "|---|---|---|---|---|---|",
    ]
    for r in trace["top_results"]:
        lines.append(
            f"| {r['rank']} | {r['supplier']} | {r['product_name']} | "
            f"{r['price']} | {r['location']} | {r['score']} |"
        )

    if trace.get("top_supplier_feature_contribution"):
        lines.append("\n## Feature Contribution — Top Result\n")
        lines.append("| Feature | Contribution |")
        lines.append("|---|---|")
        for feat, val in trace["top_supplier_feature_contribution"].items():
            lines.append(f"| {feat} | {val:+.4f} |")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SourceUp end-to-end procurement case study")
    parser.add_argument("--product", default="food packaging suppliers")
    parser.add_argument("--budget", type=float, default=50000)
    parser.add_argument("--moq", type=float, default=500)
    parser.add_argument("--location", default="Chennai")
    parser.add_argument("--certification", default="ISO")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--location-soft", action="store_true",
                         help="Treat location as a soft preference instead of mandatory "
                              "(default: mandatory, since the case study's premise is "
                              "finding suppliers IN the requested location)")
    args = parser.parse_args()

    result = run_case_study(
        product=args.product,
        max_price=args.budget,
        moq_budget=args.moq,
        location=args.location,
        certification=args.certification,
        top_k=args.top_k,
        location_mandatory=not args.location_soft,
    )

    json_path = cfg.EVAL_DIR / "case_study.json"
    md_path = cfg.EVAL_DIR / "case_study.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))

    print(f"✅ Case study written to:\n  {json_path}\n  {md_path}")