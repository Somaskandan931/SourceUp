"""
Recommendation API Endpoint - UPGRADED TO SOURCEUP-X (Async)
----------------------------------------------------
Explainable Procurement Intelligence API with:
- SME constraint filtering
- Transparent decision traces
- What-if simulation support
- Async request handling for better concurrency
"""

import time
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import traceback
import pandas as pd
import math
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.app.models.retriever import retrieve, retrieve_hybrid, retrieve_bm25, load_index
from backend.app.models.ranker import get_ranker, extract_features_batch
from backend.app.models.constraint_engine import get_constraint_engine
from backend.app.services.decision_trace import get_decision_trace
from backend.app.services.what_if_simulator import get_what_if_simulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# REQUEST MODELS
# ============================================================================

class SearchQuery(BaseModel):
    """Request model for supplier search with SME constraints"""
    # Core search
    product: str

    # Price constraints
    max_price: Optional[float] = None
    moq_budget: Optional[float] = None

    # Location preferences
    location: Optional[str] = None
    location_mandatory: Optional[bool] = False

    # Quality requirements
    certification: Optional[str] = None
    required_certifications: Optional[List[str]] = None
    min_years_experience: Optional[int] = None

    # Delivery constraints
    max_lead_time: Optional[int] = None

    # Retrieval mode
    retrieval_mode: Optional[str] = "hybrid"  # "faiss", "bm25", "hybrid"

    # Feature flags
    enable_explanations: Optional[bool] = True
    enable_what_if: Optional[bool] = False


class WhatIfQuery(BaseModel):
    """Request model for what-if simulation"""
    product: str
    constraints: Dict[str, Any]
    scenario: str


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_value(value):
    """
    Clean a value for JSON serialization.
    Replaces NaN, Infinity, and None with safe defaults.
    """
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (int, str, bool)):
        return value

    # For pandas types
    if pd.isna(value):
        return None

    return value


def clean_dict(data):
    """Recursively clean a dictionary for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_dict(item) for item in data]
    else:
        return clean_value(data)


def safe_float(value, default=0.0):
    """Convert to float, handling NaN/Infinity"""
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except (ValueError, TypeError):
        return default


def safe_int(value, default=None):
    """Convert to int, handling invalid values"""
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def get_supplier_name(supplier: Dict) -> Optional[str]:
    """Extract supplier name from various possible field names. Returns None if not found."""
    # NOTE: retriever.py normalises columns with .replace(" ", "_"), so we must
    # check underscore variants FIRST; space-variants kept for raw/un-normalised data.
    raw = (
        supplier.get("supplier_name") or
        supplier.get("supplier name") or
        supplier.get("Supplier Name") or
        supplier.get("company_name") or
        supplier.get("company name") or
        supplier.get("Company Name") or
        supplier.get("company") or
        supplier.get("name") or
        supplier.get("brand") or
        supplier.get("manufacturer") or
        None
    )
    if not raw:
        return None
    cleaned = str(raw).strip()
    # Reject known placeholder values
    if not cleaned or cleaned.lower() in {"unknown supplier", "unknown", "n/a", "nan", "-", "none"}:
        return None
    return cleaned


def get_product_name(supplier: Dict, default: str = "Product") -> str:
    """Extract product name from various possible field names."""
    return (
        supplier.get("product_name") or
        supplier.get("product name") or
        supplier.get("Product Name") or
        supplier.get("product") or
        supplier.get("Product") or
        default
    )


# ============================================================================
# MAIN RECOMMENDATION ENDPOINT (Async)
# ============================================================================

@router.post("/recommend")
async def recommend(q: SearchQuery):
    """
    UPGRADED: Explainable Procurement Intelligence API (Async)
    """
    start_time = time.time()

    try:
        logger.info(f"🔥 [SourceUp-X] Query started: {q.product}")
        logger.debug(f"Query parameters: {q.dict()}")

        # ====================================================================
        # STEP 1: Retrieve Candidates
        # ====================================================================
        step_start = time.time()
        logger.info("🔍 Retrieving candidates...")

        try:
            if q.retrieval_mode == "hybrid":
                candidates = retrieve_hybrid(q.product, k=100, alpha=0.7)
            elif q.retrieval_mode == "bm25":
                candidates = retrieve_bm25(q.product, k=100)
            else:
                candidates = retrieve(q.product, k=100)

            retrieve_time = (time.time() - step_start) * 1000
            logger.info(f"✅ Found {len(candidates)} candidates ({retrieve_time:.1f}ms)")
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}", exc_info=True)
            raise

        if not candidates:
            logger.warning(f"No candidates found for query: {q.product}")
            return {
                "results": [],
                "metadata": {
                    "total_candidates": 0,
                    "after_constraints": 0,
                    "latency_ms": round((time.time() - start_time) * 1000, 2)
                }
            }

        # ====================================================================
        # STEP 2: Apply SME Constraints
        # ====================================================================
        step_start = time.time()
        logger.info("🔒 Applying constraints...")

        try:
            constraint_engine = get_constraint_engine()

            constraints = {
                "max_price": q.max_price,
                "moq_budget": q.moq_budget,
                "max_lead_time": q.max_lead_time,
                "preferred_location": q.location,
                "location_mandatory": q.location_mandatory,
                "required_certifications": q.required_certifications or (
                    [q.certification] if q.certification else []
                ),
                "min_years_experience": q.min_years_experience
            }

            active_constraints = [k for k, v in constraints.items() if v is not None and v != [] and v is not False]
            logger.info(f"Active constraints: {active_constraints}")

            viable_suppliers = constraint_engine.apply_constraints(candidates, constraints)
            constraint_time = (time.time() - step_start) * 1000

            filter_summary = constraint_engine.get_filter_summary()
            logger.info(f"✅ {len(viable_suppliers)} suppliers pass constraints ({constraint_time:.1f}ms)")
            logger.info(f"   Filtered out: {filter_summary.get('total_filtered', 0)}")

        except Exception as e:
            logger.error(f"Constraint filtering failed: {str(e)}", exc_info=True)
            raise

        if not viable_suppliers:
            logger.warning(f"No suppliers meet constraints for query: {q.product}")
            return {
                "results": [],
                "message": "No suppliers meet your constraints",
                "filters_applied": filter_summary.get('filters_applied', []),
                "suggestion": "Try relaxing some constraints",
                "metadata": {
                    "total_candidates": len(candidates),
                    "after_constraints": 0,
                    "latency_ms": round((time.time() - start_time) * 1000, 2)
                }
            }

        # ====================================================================
        # STEP 3: Hybrid Ranking
        # ====================================================================
        step_start = time.time()
        logger.info("🎯 Ranking suppliers...")

        try:
            ranker = get_ranker()

            query_dict = {
                "product": q.product,
                "max_price": q.max_price,
                "location": q.location.lower() if q.location else "",
                "certification": q.certification.lower() if q.certification else ""
            }

            ranked_suppliers = ranker.rank(viable_suppliers, query_dict)
            ranking_time = (time.time() - step_start) * 1000
            logger.info(f"✅ Ranked {len(ranked_suppliers)} suppliers ({ranking_time:.1f}ms)")
            logger.info(f"   Ranking method: {ranker.model_type if hasattr(ranker, 'model_type') and ranker.use_ml else 'rule-based'}")

        except Exception as e:
            logger.error(f"Ranking failed: {str(e)}", exc_info=True)
            raise

        # ====================================================================
        # STEP 4: Generate Decision Traces
        # ====================================================================
        step_start = time.time()
        results = []
        features_df = None

        try:
            decision_tracer = get_decision_trace()

            if q.enable_explanations and len(ranked_suppliers) > 0:
                logger.info("📋 Generating decision traces...")
                features_df = extract_features_batch(ranked_suppliers[:50], query_dict)

            for idx, supplier in enumerate(ranked_suppliers[:20], 1):
                # Basic supplier info
                supplier_name = get_supplier_name(supplier)
                # Skip suppliers with no resolvable name
                if not supplier_name:
                    continue
                supplier_name = supplier_name  # already a clean string
                product_name = get_product_name(supplier, q.product)

                # Format display fields with safe conversions
                price_display = supplier.get("price", "Contact for pricing")
                location_display = (supplier.get("supplier_location") or supplier.get("supplier location") or supplier.get("location") or "")

                moq = safe_int(supplier.get("min order qty") or supplier.get("moq"))
                unit = supplier.get("unit", "pieces")
                moq_display = f"{moq} {unit}" if moq else None

                lead_time = safe_int(supplier.get("lead time") or supplier.get("lead_time"))
                lead_time_display = f"{lead_time} days" if lead_time else None

                product_url = supplier.get("product url", "") or supplier.get("url", "")

                # Generate decision trace
                decision_trace = None
                if q.enable_explanations and features_df is not None and idx - 1 < len(features_df):
                    try:
                        raw_trace = decision_tracer.generate_trace(
                            supplier=supplier,
                            query=query_dict,
                            features=features_df.iloc[idx - 1],
                            final_score=supplier.get('score', 0),
                            constraint_results=supplier.get('constraint_results')
                        )
                        decision_trace = clean_dict(raw_trace) if raw_trace else None
                    except Exception as trace_error:
                        logger.warning(f"Trace generation failed for {supplier_name}: {trace_error}")

                # Calculate confidence score
                constraint_score = safe_float(supplier.get('constraint_score', 1.0), 1.0)
                supplier_score = safe_float(supplier.get('score', 0.0), 0.0)
                confidence = constraint_score * supplier_score

                # Get reasons from decision trace or generate basic ones
                reasons = []
                if decision_trace and decision_trace.get('summary'):
                    reasons = decision_trace['summary']
                elif supplier.get('reasons'):
                    reasons = supplier['reasons']
                else:
                    # Generate basic reasons
                    if q.max_price and price_display != "Contact for pricing":
                        try:
                            price_val = float(price_display) if isinstance(price_display, (int, float)) else 0
                            if price_val <= q.max_price:
                                reasons.append(f"Within budget (${price_val})")
                        except:
                            pass
                    if location_display and q.location and q.location.lower() in location_display.lower():
                        reasons.append(f"Located in {location_display}")
                    if q.certification and supplier.get("certifications"):
                        if q.certification.lower() in str(supplier.get("certifications")).lower():
                            reasons.append(f"{q.certification.upper()} certified")

                result = {
                    "supplier": supplier_name,
                    "product": str(product_name),
                    "score": safe_float(supplier_score, 0.0),
                    "rank": idx,
                    "reasons": reasons[:5],  # Limit to top 5 reasons
                    "price": str(price_display),
                    "location": str(location_display),
                    "moq": moq_display,
                    "lead_time": lead_time_display,
                    "url": product_url if product_url else None,
                    "decision_trace": decision_trace,
                    "constraint_results": clean_dict(supplier.get('constraint_results')) if supplier.get('constraint_results') else None,
                    "confidence_score": round(confidence, 3)
                }

                results.append(result)

            trace_time = (time.time() - step_start) * 1000
            logger.info(f"✅ Generated {len(results)} decision traces ({trace_time:.1f}ms)")

        except Exception as e:
            logger.error(f"Decision trace generation failed: {str(e)}", exc_info=True)
            # Don't raise - return results without traces

        # ====================================================================
        # STEP 5: What-If Scenarios (Optional)
        # ====================================================================
        what_if_scenarios = None
        what_if_time = 0

        if q.enable_what_if and len(ranked_suppliers) >= 5:
            step_start = time.time()
            logger.info("🔮 Generating what-if scenarios...")

            try:
                simulator = get_what_if_simulator()

                raw_scenarios = {
                    "price_focused": simulator.simulate_trade_off(
                        ranked_suppliers[:20],
                        query_dict,
                        "price_over_speed"
                    ),
                    "speed_focused": simulator.simulate_trade_off(
                        ranked_suppliers[:20],
                        query_dict,
                        "speed_over_price"
                    ),
                    "quality_focused": simulator.simulate_trade_off(
                        ranked_suppliers[:20],
                        query_dict,
                        "quality_over_cost"
                    )
                }

                what_if_scenarios = clean_dict(raw_scenarios)
                what_if_time = (time.time() - step_start) * 1000
                logger.info(f"✅ Generated what-if scenarios ({what_if_time:.1f}ms)")

            except Exception as e:
                logger.warning(f"What-if simulation failed (non-critical): {str(e)}")

        # ====================================================================
        # RETURN RESPONSE with timing metrics
        # ====================================================================
        total_time = (time.time() - start_time) * 1000

        response = {
            "results": results,
            "metadata": {
                "total_candidates": len(candidates),
                "after_constraints": len(viable_suppliers),
                "filters_applied": filter_summary.get('filters_applied', []),
                "ranking_method": ranker.model_type if hasattr(ranker, 'model_type') and ranker.use_ml else "rule-based",
                "retrieval_mode": q.retrieval_mode,
                "latency_ms": round(total_time, 2),
                "timing_breakdown": {
                    "retrieval_ms": round(retrieve_time, 1),
                    "constraint_filtering_ms": round(constraint_time, 1),
                    "ranking_ms": round(ranking_time, 1),
                    "decision_trace_ms": round(trace_time, 1),
                    "what_if_ms": round(what_if_time, 1) if what_if_scenarios else 0,
                }
            }
        }

        if what_if_scenarios:
            response["what_if_scenarios"] = what_if_scenarios

        logger.info(f"✅ Returning {len(results)} results in {total_time:.1f}ms total")
        logger.info(f"   Breakdown: R={retrieve_time:.0f}ms | C={constraint_time:.0f}ms | "
                    f"RK={ranking_time:.0f}ms | T={trace_time:.0f}ms")

        return clean_dict(response)

    except Exception as e:
        error_msg = f"Recommendation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        total_time = (time.time() - start_time) * 1000
        raise HTTPException(
            status_code=500,
            detail={
                "error": error_msg,
                "latency_ms": round(total_time, 2)
            }
        )


# ============================================================================
# WHAT-IF SIMULATION ENDPOINT (Async)
# ============================================================================

@router.post("/what-if")
async def what_if_simulation(q: WhatIfQuery):
    """Interactive what-if simulation"""
    try:
        logger.info(f"🔮 What-if simulation: {q.scenario}")

        # Retrieve candidates
        candidates = retrieve(q.product, k=50)

        if not candidates:
            return {"error": "No candidates found", "scenario": q.scenario}

        # Apply constraints
        constraint_engine = get_constraint_engine()
        viable_suppliers = constraint_engine.apply_constraints(candidates, q.constraints)

        if not viable_suppliers:
            return {
                "error": "No suppliers meet the constraints",
                "scenario": q.scenario,
                "total_candidates": len(candidates),
                "after_constraints": 0
            }

        # Run simulation
        simulator = get_what_if_simulator()
        result = simulator.simulate_trade_off(viable_suppliers, q.constraints, q.scenario)

        # Add metadata
        result["total_candidates"] = len(candidates)
        result["after_constraints"] = len(viable_suppliers)
        result["scenario"] = q.scenario

        logger.info(f"✅ What-if simulation complete: {len(viable_suppliers)} suppliers analyzed")
        return clean_dict(result)

    except Exception as e:
        logger.error(f"What-if simulation failed: {str(e)}", exc_info=True)
        return {"error": str(e), "scenario": q.scenario if hasattr(q, 'scenario') else "unknown"}


# ============================================================================
# COMPARISON ENDPOINT (Async)
# ============================================================================

@router.post("/compare")
async def compare_suppliers(supplier_ids: List[int], query: Dict[str, Any]):
    """Compare two suppliers side-by-side"""
    try:
        if len(supplier_ids) != 2:
            logger.warning(f"Invalid comparison: expected 2 IDs, got {len(supplier_ids)}")
            return {"error": "Provide exactly 2 supplier IDs"}

        logger.info(f"🔍 Comparing suppliers: {supplier_ids}")

        index, meta = load_index()

        if supplier_ids[0] >= len(meta) or supplier_ids[1] >= len(meta):
            return {"error": "Invalid supplier IDs"}

        supplier_a = meta.iloc[supplier_ids[0]].to_dict()
        supplier_b = meta.iloc[supplier_ids[1]].to_dict()

        ranker = get_ranker()
        ranked = ranker.rank([supplier_a, supplier_b], query)

        decision_tracer = get_decision_trace()
        features_df = extract_features_batch(ranked, query)

        trace_a = decision_tracer.generate_trace(
            ranked[0], query, features_df.iloc[0], ranked[0].get('score', 0)
        )
        trace_b = decision_tracer.generate_trace(
            ranked[1], query, features_df.iloc[1], ranked[1].get('score', 0)
        )

        comparison = decision_tracer.generate_comparative_trace(
            ranked[0], ranked[1], trace_a, trace_b
        )

        logger.info(f"✅ Comparison complete - Winner: {comparison.get('winner', 'Unknown')}")
        return clean_dict(comparison)

    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LEGACY ENDPOINTS (Async)
# ============================================================================

@router.get("/recommend/test")
async def test_recommend():
    """Test endpoint to verify system is working"""
    try:
        logger.info("Testing recommendation system")

        index, meta = load_index()

        sample = meta.iloc[0].to_dict() if len(meta) > 0 else {}
        sample_clean = clean_dict(sample)

        logger.info(f"System ready - {len(meta):,} suppliers loaded")

        return {
            "status": "SourceUp-X Ready",
            "version": "2.0 (Explainable Intelligence)",
            "total_suppliers": len(meta),
            "features": [
                "SME Constraint Filtering",
                "Decision Transparency",
                "What-If Simulation",
                "Hybrid ML Ranking",
                "Async API",
                "Hybrid FAISS+BM25 Retrieval"
            ],
            "sample": sample_clean
        }
    except Exception as e:
        logger.error(f"Test endpoint failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        logger.info("Fetching system statistics")

        index, meta = load_index()

        total = len(meta)

        locations = {}
        if 'supplier location' in meta.columns:
            locations = {str(k): int(v) for k, v in meta['supplier location'].value_counts().head(10).items()}
        elif 'location' in meta.columns:
            locations = {str(k): int(v) for k, v in meta['location'].value_counts().head(10).items()}

        categories = {}
        if 'category l1' in meta.columns:
            categories = {str(k): int(v) for k, v in meta['category l1'].value_counts().head(10).items()}
        elif 'category' in meta.columns:
            categories = {str(k): int(v) for k, v in meta['category'].value_counts().head(10).items()}

        logger.info(f"Stats: {total:,} suppliers, {len(locations)} locations")

        return {
            "total_suppliers": total,
            "top_locations": locations,
            "top_categories": categories,
            "system": "SourceUp-X (Explainable Intelligence)",
            "retrieval_modes": ["faiss", "bm25", "hybrid"]
        }
    except Exception as e:
        logger.error(f"Stats endpoint failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))