"""
Recommendation API Endpoint (Fixed for GlobalSources Data)
-----------------------------------------------------------
FastAPI endpoint for supplier recommendations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import traceback
from backend.app.models.retriever import retrieve
from backend.app.models.ranker import score
from backend.app.services.explanation import explain

router = APIRouter()


class SearchQuery(BaseModel):
    """Request model for supplier search"""
    product: str
    max_price: Optional[float] = None
    location: Optional[str] = None
    certification: Optional[str] = None


class SupplierResult(BaseModel):
    """Response model for supplier results"""
    supplier: str
    product: str
    score: float
    reasons: List[str]
    price: Optional[str] = None
    location: Optional[str] = None
    moq: Optional[str] = None
    lead_time: Optional[str] = None
    url: Optional[str] = None


@router.post("/recommend")
def recommend(q: SearchQuery):
    """
    Get supplier recommendations based on search criteria.
    Adapted for GlobalSources data structure.

    Args:
        q: SearchQuery with product, max_price, location, certification

    Returns:
        List of ranked suppliers with explanations
    """
    try:
        print(f"\n📥 Received query: {q.dict()}")

        # Retrieve candidates using semantic search
        print(f"🔍 Searching for: {q.product}")
        candidates = retrieve(q.product, k=50)  # Get more candidates
        print(f"✅ Found {len(candidates)} candidates")

        if not candidates:
            print("⚠️  No candidates found")
            return []

        # Convert query to dict for compatibility
        query_dict = {
            "product": q.product,
            "max_price": q.max_price if q.max_price else None,
            "location": q.location.lower() if q.location else "",
            "certification": q.certification.lower() if q.certification else ""
        }

        print(f"🎯 Query parameters: {query_dict}")

        # Score and rank candidates
        results = []
        errors = []

        for idx, supplier in enumerate(candidates):
            try:
                # Calculate score and generate explanations
                supplier_score = score(supplier, query_dict)
                reasons = explain(supplier, query_dict)

                # Get supplier name - use company_name or supplier_name
                supplier_name = (
                    supplier.get("supplier name") or
                    supplier.get("company name") or
                    supplier.get("supplier_name") or
                    "Unknown Supplier"
                )

                # Get product name
                product_name = (
                    supplier.get("product name") or
                    supplier.get("product_name") or
                    q.product
                )

                # Get additional details
                price_display = supplier.get("price", "Contact for pricing")
                location_display = supplier.get("supplier location") or supplier.get("location", "")

                moq = supplier.get("min order qty")
                unit = supplier.get("unit", "pieces")
                moq_display = f"{int(moq)} {unit}" if moq else None

                lead_time = supplier.get("lead time")
                lead_time_display = f"{int(lead_time)} days" if lead_time else None

                product_url = supplier.get("product url", "")

                results.append({
                    "supplier": str(supplier_name),
                    "product": str(product_name),
                    "score": float(supplier_score),
                    "reasons": reasons,
                    "price": str(price_display),
                    "location": str(location_display),
                    "moq": moq_display,
                    "lead_time": lead_time_display,
                    "url": product_url
                })

            except Exception as e:
                errors.append(f"Error processing supplier {idx}: {str(e)}")
                print(f"⚠️  Error processing supplier {idx}: {e}")
                continue

        if errors:
            print(f"⚠️  Encountered {len(errors)} errors while processing")

        if not results:
            print("❌ No valid results after processing")
            return []

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        print(f"✅ Returning {len(results[:20])} results")
        return results[:20]  # Return top 20

    except Exception as e:
        error_msg = f"Recommendation failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/recommend/test")
def test_recommend():
    """Test endpoint to verify retriever is working"""
    try:
        from backend.app.models.retriever import load_index
        index, meta = load_index()

        # Get sample with actual column names
        sample = meta.iloc[0].to_dict() if len(meta) > 0 else {}

        # Clean NaN values for JSON serialization
        sample_clean = {}
        for k, v in sample.items():
            if isinstance(v, float) and v != v:  # NaN check
                sample_clean[k] = None
            else:
                sample_clean[k] = v

        return {
            "status": "ok",
            "total_suppliers": len(meta),
            "columns": list(meta.columns),
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

        # Calculate some stats
        total = len(meta)

        # Count by location
        locations = meta['supplier location'].value_counts().head(10).to_dict() if 'supplier location' in meta.columns else {}

        # Count by category
        categories = meta['category l1'].value_counts().head(10).to_dict() if 'category l1' in meta.columns else {}

        return {
            "total_suppliers": total,
            "total_products": total,  # In your data, each row is a product
            "top_locations": locations,
            "top_categories": categories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))