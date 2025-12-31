"""
Recommendation API Endpoint - UPGRADED TO SOURCEUP-X
----------------------------------------------------
Explainable Procurement Intelligence API with:
- SME constraint filtering
- Transparent decision traces
- What-if simulation support
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import traceback
import pandas as pd
import math

from backend.app.models.retriever import retrieve
from backend.app.models.ranker import get_ranker, extract_features_batch
from backend.app.models.constraint_engine import get_constraint_engine
from backend.app.services.decision_trace import get_decision_trace
from backend.app.services.what_if_simulator import get_what_if_simulator

router = APIRouter()


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


# ============================================================================
# REQUEST MODELS
# ============================================================================

class SearchQuery(BaseModel):
    """Request model for supplier search with SME constraints"""
    # Core search
    product: str

    # Price constraints
    max_price: Optional[float] = None
    moq_budget: Optional[float] = None  # 🆕 Total budget for MOQ

    # Location preferences
    location: Optional[str] = None
    location_mandatory: Optional[bool] = False  # 🆕 Hard vs soft constraint

    # Quality requirements
    certification: Optional[str] = None
    required_certifications: Optional[List[str]] = None  # 🆕 Multiple certs
    min_years_experience: Optional[int] = None  # 🆕 Platform experience

    # Delivery constraints
    max_lead_time: Optional[int] = None  # 🆕 Urgency requirement

    # Feature flags
    enable_explanations: Optional[bool] = True  # 🆕 Return decision traces
    enable_what_if: Optional[bool] = False  # 🆕 Include what-if scenarios


class WhatIfQuery(BaseModel):
    """Request model for what-if simulation"""
    product: str
    constraints: Dict
    scenario: str  # 'price_over_speed', 'speed_over_price', etc.


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class SupplierResult(BaseModel):
    """Enhanced response with explanations"""
    # Basic info
    supplier: str
    product: str
    score: float
    rank: int

    # Details
    price: Optional[str] = None
    location: Optional[str] = None
    moq: Optional[str] = None
    lead_time: Optional[str] = None
    url: Optional[str] = None

    # 🆕 Explanations
    reasons: List[str]
    decision_trace: Optional[Dict] = None
    constraint_results: Optional[Dict] = None
    confidence_score: Optional[float] = None


# ============================================================================
# MAIN RECOMMENDATION ENDPOINT (UPGRADED)
# ============================================================================

@router.post("/recommend")
def recommend(q: SearchQuery):
    """
    🆕 UPGRADED: Explainable Procurement Intelligence API

    Flow:
    1. Semantic retrieval (FAISS)
    2. SME constraint filtering
    3. Hybrid ML + Rule ranking
    4. Decision trace generation
    5. Optional what-if scenarios

    Returns ranked suppliers with full transparency.
    """
    try:
        print(f"\n🔥 [SourceUp-X] Query: {q.product}")

        # ====================================================================
        # STEP 1: Retrieve Candidates (Unchanged)
        # ====================================================================
        print(f"🔍 Retrieving candidates...")
        candidates = retrieve(q.product, k=100)  # Get more for filtering
        print(f"✅ Found {len(candidates)} candidates")

        if not candidates:
            return []

        # ====================================================================
        # STEP 2: Apply SME Constraints (NEW)
        # ====================================================================
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

        print(f"🔒 Applying constraints: {[k for k, v in constraints.items() if v is not None]}")
        viable_suppliers = constraint_engine.apply_constraints(candidates, constraints)

        filter_summary = constraint_engine.get_filter_summary()
        print(f"✅ {len(viable_suppliers)} suppliers pass constraints")
        print(f"   Filtered out: {filter_summary['total_filtered']}")

        if not viable_suppliers:
            return {
                "results": [],
                "message": "No suppliers meet your constraints",
                "filters_applied": filter_summary['filters_applied'],
                "suggestion": "Try relaxing some constraints"
            }

        # ====================================================================
        # STEP 3: Hybrid Ranking (Enhanced)
        # ====================================================================
        print(f"🎯 Ranking suppliers...")
        ranker = get_ranker()

        query_dict = {
            "product": q.product,
            "max_price": q.max_price,
            "location": q.location.lower() if q.location else "",
            "certification": q.certification.lower() if q.certification else ""
        }

        ranked_suppliers = ranker.rank(viable_suppliers, query_dict)
        print(f"✅ Ranked {len(ranked_suppliers)} suppliers")

        # ====================================================================
        # STEP 4: Generate Decision Traces (NEW)
        # ====================================================================
        results = []
        decision_tracer = get_decision_trace()

        if q.enable_explanations:
            print(f"📋 Generating decision traces...")
            features_df = extract_features_batch(ranked_suppliers, query_dict)

        for idx, supplier in enumerate(ranked_suppliers[:20], 1):
            # Basic supplier info
            supplier_name = (
                supplier.get("supplier name") or
                supplier.get("Supplier Name") or
                "Unknown Supplier"
            )

            product_name = (
                supplier.get("product name") or
                supplier.get("Product Name") or
                q.product
            )

            # Format display fields with safe conversions
            price_display = supplier.get("price", "Contact for pricing")
            location_display = supplier.get("supplier location", "")

            moq = safe_int(supplier.get("min order qty"))
            unit = supplier.get("unit", "pieces")
            moq_display = f"{moq} {unit}" if moq else None

            lead_time = safe_int(supplier.get("lead time"))
            lead_time_display = f"{lead_time} days" if lead_time else None

            product_url = supplier.get("product url", "")

            # 🆕 Generate decision trace (with cleaning)
            decision_trace = None
            if q.enable_explanations:
                raw_trace = decision_tracer.generate_trace(
                    supplier=supplier,
                    query=query_dict,
                    features=features_df.iloc[idx-1],
                    final_score=supplier['score'],
                    constraint_results=supplier.get('constraint_results')
                )
                # Clean the trace for JSON
                decision_trace = clean_dict(raw_trace) if raw_trace else None

            # 🆕 Calculate confidence score (with safe conversion)
            constraint_score = safe_float(supplier.get('constraint_score', 1.0), 1.0)
            supplier_score = safe_float(supplier.get('score', 0.0), 0.0)
            confidence = constraint_score * supplier_score

            result = {
                "supplier": str(supplier_name),
                "product": str(product_name),
                "score": safe_float(supplier_score, 0.0),
                "rank": idx,
                "reasons": supplier.get('reasons', []),
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

        # ====================================================================
        # STEP 5: What-If Scenarios (Optional)
        # ====================================================================
        what_if_scenarios = None
        if q.enable_what_if and len(ranked_suppliers) >= 5:
            print(f"🔮 Generating what-if scenarios...")
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

            # Clean what-if scenarios
            what_if_scenarios = clean_dict(raw_scenarios)

        # ====================================================================
        # RETURN RESPONSE (with cleaning)
        # ====================================================================
        response = {
            "results": results,
            "metadata": {
                "total_candidates": len(candidates),
                "after_constraints": len(viable_suppliers),
                "filters_applied": filter_summary['filters_applied'],
                "ranking_method": ranker.model_type if ranker.use_ml else "rule-based"
            }
        }

        if what_if_scenarios:
            response["what_if_scenarios"] = what_if_scenarios

        print(f"✅ Returning {len(results)} results\n")

        # Final safety check
        return clean_dict(response)

    except Exception as e:
        error_msg = f"Recommendation failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


# ============================================================================
# WHAT-IF SIMULATION ENDPOINT (NEW)
# ============================================================================

@router.post("/what-if")
def what_if_simulation(q: WhatIfQuery):
    """
    🆕 NEW: Interactive what-if simulation

    Allows users to explore trade-offs without re-searching.
    """
    try:
        print(f"\n🔮 What-if simulation: {q.scenario}")

        # Retrieve candidates
        candidates = retrieve(q.product, k=50)

        if not candidates:
            return {"error": "No candidates found"}

        # Apply original constraints
        constraint_engine = get_constraint_engine()
        viable_suppliers = constraint_engine.apply_constraints(candidates, q.constraints)

        # Run simulation
        simulator = get_what_if_simulator()
        result = simulator.simulate_trade_off(viable_suppliers, q.constraints, q.scenario)

        return clean_dict(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# COMPARISON ENDPOINT (NEW)
# ============================================================================

@router.post("/compare")
def compare_suppliers(supplier_ids: List[int], query: Dict):
    """
    🆕 NEW: Compare two suppliers side-by-side

    Shows exactly why one ranks higher than another.
    """
    try:
        if len(supplier_ids) != 2:
            return {"error": "Provide exactly 2 supplier IDs"}

        # Retrieve both suppliers
        from backend.app.models.retriever import load_index
        _, meta = load_index()

        supplier_a = meta.iloc[supplier_ids[0]].to_dict()
        supplier_b = meta.iloc[supplier_ids[1]].to_dict()

        # Rank them
        ranker = get_ranker()
        ranked = ranker.rank([supplier_a, supplier_b], query)

        # Generate traces
        decision_tracer = get_decision_trace()
        features_df = extract_features_batch(ranked, query)

        trace_a = decision_tracer.generate_trace(
            ranked[0], query, features_df.iloc[0], ranked[0]['score']
        )
        trace_b = decision_tracer.generate_trace(
            ranked[1], query, features_df.iloc[1], ranked[1]['score']
        )

        # Comparative analysis
        comparison = decision_tracer.generate_comparative_trace(
            ranked[0], ranked[1], trace_a, trace_b
        )

        return clean_dict(comparison)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LEGACY ENDPOINTS (Keep for backward compatibility)
# ============================================================================

@router.get("/recommend/test")
def test_recommend():
    """Test endpoint to verify system is working"""
    try:
        from backend.app.models.retriever import load_index
        index, meta = load_index()

        sample = meta.iloc[0].to_dict() if len(meta) > 0 else {}

        # Clean the sample
        sample_clean = clean_dict(sample)

        return {
            "status": "SourceUp-X Ready",
            "version": "2.0 (Explainable Intelligence)",
            "total_suppliers": len(meta),
            "features": [
                "SME Constraint Filtering",
                "Decision Transparency",
                "What-If Simulation",
                "Hybrid ML Ranking"
            ],
            "sample": sample_clean
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
def get_stats():
    """Get database statistics"""
    try:
        from backend.app.models.retriever import load_index
        index, meta = load_index()

        total = len(meta)

        # Safe conversion of value_counts to dict
        locations = {}
        if 'supplier location' in meta.columns:
            locations = {str(k): int(v) for k, v in meta['supplier location'].value_counts().head(10).items()}

        categories = {}
        if 'category l1' in meta.columns:
            categories = {str(k): int(v) for k, v in meta['category l1'].value_counts().head(10).items()}

        return {
            "total_suppliers": total,
            "top_locations": locations,
            "top_categories": categories,
            "system": "SourceUp-X (Explainable Intelligence)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))