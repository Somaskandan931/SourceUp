"""
fields.py — Safe field accessor for SourceUp supplier dicts
------------------------------------------------------------
Supplier dicts come from retriever.py which normalises CSV columns to
lowercase + underscore (e.g. "supplier_location", "years_with_gs").
Some consuming code still uses space/Title-Case variants. This helper
tries all common forms so nothing silently returns None.
"""
from typing import Any, Optional


# Map canonical underscore name → all aliases to try (order matters)
_ALIASES: dict[str, list[str]] = {
    "supplier_location": ["supplier_location", "supplier location", "Supplier Location", "location", "Location"],
    "location":         ["location", "Location", "supplier_location", "supplier location"],
    "price_min":        ["price_min", "price min", "Price Min", "price", "Price"],
    "price":            ["price", "Price", "price_min", "price min"],
    "min_order_qty":    ["min_order_qty", "min order qty", "Min Order Qty", "moq", "MOQ"],
    "lead_time":        ["lead_time", "lead time", "Lead Time"],
    "certifications":   ["certifications", "Certifications"],
    "years_with_gs":    ["years_with_gs", "years with gs", "Years with GS"],
    "years_on_platform":["years_on_platform", "years on platform", "Years on Platform", "years_with_gs", "years with gs"],
    "business_type":    ["business_type", "business type", "Business Type"],
    "supplier_name":    ["supplier_name", "supplier name", "Supplier Name", "company_name", "company name", "Company Name"],
    "product_name":     ["product_name", "product name", "Product Name", "product", "Product"],
    "product_url":      ["product_url", "product url", "Product URL", "url", "URL"],
    "unit":             ["unit", "Unit"],
    "faiss_score":      ["faiss_score", "semantic_score"],
    "semantic_score":   ["semantic_score", "faiss_score"],
}


def get_field(d: dict, *names: str, default: Any = None) -> Any:
    """
    Retrieve a field from a supplier dict, trying multiple name variants.

    Usage:
        get_field(supplier, "price_min")          # tries all price_min aliases
        get_field(supplier, "years_with_gs", "years_on_platform")  # tries both + their aliases
        get_field(supplier, "price", default=0)
    """
    for name in names:
        # Try the name itself first
        val = d.get(name)
        if val is not None and val == val:  # also filters NaN (float NaN != NaN)
            return val
        # Try all registered aliases
        for alias in _ALIASES.get(name, []):
            val = d.get(alias)
            if val is not None and val == val:
                return val
    return default
