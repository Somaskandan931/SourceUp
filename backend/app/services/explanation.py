"""
Explanation Module — SourceUp
------------------------------
Generates human-readable explanations for recommendations.
Uses the shared get_field accessor so underscore-normalised keys
from the retriever are resolved correctly.
"""
from backend.app.utils.fields import get_field


def parse_price(price_value) -> float:
    if price_value is None or (isinstance(price_value, float) and price_value != price_value):
        return 0.0
    try:
        if isinstance(price_value, (int, float)):
            return float(price_value)
        s = str(price_value).strip()
        return float(s.split('-')[0].strip()) if '-' in s else float(s)
    except Exception:
        return 0.0


def explain(supplier: dict, query: dict) -> list:
    reasons = []

    # Price
    supplier_price = parse_price(get_field(supplier, "price_min") or get_field(supplier, "price", default=0))
    query_max_price = query.get("max_price", 1e9)
    if supplier_price > 0:
        price_display = get_field(supplier, "price", default=f"${supplier_price}")
        if query_max_price and supplier_price <= query_max_price:
            reasons.append(f"Within budget ({price_display})")
        else:
            reasons.append(f"Price: {price_display}")

    # Location
    supplier_location = str(
        get_field(supplier, "supplier_location") or
        get_field(supplier, "location", default="") or ""
    ).strip()
    query_location = str(query.get("location", "")).strip()
    if supplier_location:
        if query_location and query_location.lower() in supplier_location.lower():
            reasons.append(f"Located in {supplier_location}")
        else:
            reasons.append(f"From {supplier_location}")

    # Certifications
    supplier_certs = str(get_field(supplier, "certifications", default="") or "")
    query_cert = str(query.get("certification", "")).lower().strip()
    if query_cert and supplier_certs and supplier_certs.lower() not in ('nan', ''):
        if query_cert in supplier_certs.lower():
            reasons.append(f"{query_cert.upper()} certified")

    # Business type
    business_type = str(get_field(supplier, "business_type", default="") or "")
    if "manufacturer" in business_type.lower():
        reasons.append("Direct manufacturer")

    # Years on platform
    years = get_field(supplier, "years_with_gs", "years_on_platform", default=0) or 0
    try:
        years_int = int(float(years))
        if years_int >= 5:
            reasons.append(f"{years_int}+ years verified")
    except Exception:
        pass

    # Lead time
    lead_time = get_field(supplier, "lead_time")
    if lead_time:
        try:
            days = int(float(lead_time))
            if days <= 15:
                reasons.append(f"Quick delivery ({days} days)")
        except Exception:
            pass

    # MOQ
    moq = get_field(supplier, "min_order_qty")
    if moq:
        try:
            moq_int = int(float(moq))
            unit = str(get_field(supplier, "unit", default="pieces") or "pieces")
            reasons.append(f"MOQ: {moq_int} {unit.lower()}")
        except Exception:
            pass

    if not reasons:
        reasons.append("Matches your search")

    return reasons
