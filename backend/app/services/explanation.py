"""
Explanation Module (Fixed for GlobalSources Data)
--------------------------------------------------
Generates human-readable explanations for recommendations.
"""


def parse_price(price_value) -> float:
    """Parse price from various formats"""
    if price_value is None or (isinstance(price_value, float) and price_value != price_value):
        return 0.0

    try:
        if isinstance(price_value, (int, float)):
            return float(price_value)

        price_str = str(price_value).strip()
        if '-' in price_str:
            parts = price_str.split('-')
            return float(parts[0].strip())

        return float(price_str)
    except:
        return 0.0


def explain(supplier: dict, query: dict) -> list:
    """
    Generate explanation for why a supplier was recommended.
    Adapted for GlobalSources data structure.

    Args:
        supplier: Supplier data dictionary
        query: Query parameters dictionary

    Returns:
        List of reason strings
    """
    reasons = []

    # Price check
    supplier_price = supplier.get("price min", None)
    if supplier_price is None or (isinstance(supplier_price, float) and supplier_price != supplier_price):
        supplier_price = parse_price(supplier.get("price", 0))
    else:
        supplier_price = float(supplier_price)

    query_max_price = query.get("max_price", 1e9)

    if supplier_price > 0:
        if query_max_price and supplier_price <= query_max_price:
            price_display = supplier.get("price", f"${supplier_price}")
            reasons.append(f"Within budget ({price_display})")
        else:
            price_display = supplier.get("price", f"${supplier_price}")
            reasons.append(f"Price: {price_display}")

    # Location check
    supplier_location = str(supplier.get("supplier location") or supplier.get("location", "")).strip()
    query_location = str(query.get("location", "")).strip()

    if supplier_location:
        if query_location and query_location.lower() in supplier_location.lower():
            reasons.append(f"Located in {supplier_location}")
        else:
            reasons.append(f"From {supplier_location}")

    # Certification check
    supplier_certs = str(supplier.get("certifications", ""))
    query_cert = str(query.get("certification", "")).lower().strip()

    if query_cert and supplier_certs and supplier_certs.lower() != 'nan':
        if query_cert in supplier_certs.lower():
            reasons.append(f"{query_cert.upper()} certified")

    # Business type
    business_type = str(supplier.get("business type", ""))
    if "manufacturer" in business_type.lower():
        reasons.append("Direct manufacturer")

    # Years with platform
    years = supplier.get("years with gs", 0)
    if years:
        try:
            years_int = int(years)
            if years_int >= 5:
                reasons.append(f"{years_int}+ years verified")
        except:
            pass

    # Lead time
    lead_time = supplier.get("lead time")
    if lead_time:
        try:
            days = int(lead_time)
            if days <= 15:
                reasons.append(f"Quick delivery ({days} days)")
        except:
            pass

    # Minimum order quantity
    moq = supplier.get("min order qty")
    if moq:
        try:
            moq_int = int(moq)
            unit = supplier.get("unit", "pieces")
            reasons.append(f"MOQ: {moq_int} {unit.lower()}")
        except:
            pass

    # If no specific reasons, add semantic match
    if not reasons:
        reasons.append("Matches your search")

    return reasons