"""
Rule-based Parser - FIXED
--------------------------
Prevents garbage product extraction from questions.
"""

import re


# ============================================================================
# FIX 6: BLOCK QUESTION STARTERS
# ============================================================================
QUESTION_STARTERS = ['who', 'what', 'why', 'how', 'when', 'where']


def rule_parse(text: str) -> dict:
    """
    Extract intent using simple rules and regex patterns.
    FIXED to prevent extracting questions as products.

    Args:
        text: User query text

    Returns:
        Dictionary with extracted fields
    """
    text_lower = text.lower()

    # ============================================================================
    # NEW: Block question starters from becoming products
    # ============================================================================
    words = text_lower.split()
    if words and words[0] in QUESTION_STARTERS:
        return {
            "product": None,
            "max_price": None,
            "location": "",
            "certification": ""
        }

    # ============================================================================
    # EXISTING: Extract product (everything before price/location keywords)
    # ============================================================================
    product = text
    for keyword in ['under', 'below', 'in', 'from', 'with', 'certified']:
        if keyword in text_lower:
            product = text[:text_lower.index(keyword)].strip()
            break

    # Clean up common search prefixes
    for prefix in ['find', 'search for', 'looking for', 'need', 'want']:
        if product.lower().startswith(prefix):
            product = product[len(prefix):].strip()

    # Extract price
    max_price = None
    price_match = re.search(r'(\$|usd|dollar)?[\s]*(\d+(?:\.\d+)?)', text_lower)
    if price_match:
        max_price = float(price_match.group(2))

    # Extract location
    location = ""
    location_patterns = [
        r'in\s+([a-z]+)',
        r'from\s+([a-z]+)',
        r'\b(chennai|vietnam|china|india|usa|uk)\b'
    ]
    for pattern in location_patterns:
        location_match = re.search(pattern, text_lower)
        if location_match:
            location = location_match.group(1)
            break

    # Extract certification
    certification = ""
    cert_patterns = ['iso', 'ce', 'fda', 'rohs', 'ul']
    for cert in cert_patterns:
        if cert in text_lower:
            certification = cert.upper()
            break

    return {
        "product": product,
        "max_price": max_price,
        "location": location,
        "certification": certification
    }