"""
Parser Module - FIXED
----------------------
Combines rule-based and GPT-based parsing with ENHANCED intent classification.
Prevents prompt variable mismatches and handles general questions properly.
"""

from sourcebot.nlu.rules import rule_parse
from sourcebot.nlu.gpt_fallback import gpt_parse


def classify_intent(text: str) -> dict:
    """
    Classify user intent BEFORE parsing with EXPANDED patterns.

    Returns:
        {
            'intent': 'product_search' | 'information' | 'conversation',
            'confidence': float
        }
    """
    text_lower = text.lower().strip()

    # ============================================================================
    # FIX 1: ADD EDUCATION/DEFINITION INTENT PATTERNS
    # ============================================================================
    education_patterns = [
        'who is', 'who are',
        'meaning of', 'define',
        'difference between',
        'buyer and seller',
        'what does', 'role of'
    ]

    # Check education patterns FIRST (highest priority)
    if any(p in text_lower for p in education_patterns):
        return {'intent': 'information', 'confidence': 0.95}

    # ============================================================================
    # EXISTING: Informational question patterns (EXPANDED)
    # ============================================================================
    info_patterns = [
        'what is', 'what are', 'what\'s', 'how do', 'how to', 'how can',
        'why', 'when', 'where', 'explain', 'tell me about',
        'requirements for', 'process of', 'steps to', 'difference between',
        'define', 'meaning of', 'help me understand',
        'is it hard', 'is it difficult', 'is it easy', 'is it worth',
        'should i', 'can i', 'do i need'
    ]

    # Check for informational intent
    if any(pattern in text_lower for pattern in info_patterns):
        # Exception: "What are the best containers?" is still a search
        if any(word in text_lower for word in ['best', 'top', 'recommended', 'good']):
            return {'intent': 'product_search', 'confidence': 0.8}
        return {'intent': 'information', 'confidence': 0.9}

    # ============================================================================
    # Product search patterns
    # ============================================================================
    search_patterns = [
        'find', 'search', 'looking for', 'need', 'want', 'buy',
        'purchase', 'supplier', 'manufacturer', 'get me', 'show me',
        'recommend', 'suggest', 'quote'
    ]

    # Check for explicit search intent
    if any(pattern in text_lower for pattern in search_patterns):
        return {'intent': 'product_search', 'confidence': 0.95}

    # Check for product terms (containers, packaging, etc.)
    product_terms = [
        'container', 'box', 'packaging', 'bottle', 'bag', 'wrap',
        'equipment', 'machine', 'device', 'tool', 'component', 'part',
        'material', 'fabric', 'plastic', 'metal', 'electronics'
    ]
    if any(term in text_lower for term in product_terms):
        return {'intent': 'product_search', 'confidence': 0.75}

    # Check if very short (likely greeting/conversation)
    # BUT exclude if it contains sourcing/business terms
    if len(text.split()) <= 3:
        sourcing_terms = ['sourcing', 'supplier', 'buyer', 'seller', 'moq', 'iso', 'fda']
        if not any(term in text_lower for term in sourcing_terms):
            return {'intent': 'conversation', 'confidence': 0.7}

    # Default: conversation
    return {'intent': 'conversation', 'confidence': 0.5}


def parse(text: str) -> dict:
    """
    Parse user query with intent classification first.
    FIXED to prevent prompt variable mismatches.

    Args:
        text: User query text

    Returns:
        Dictionary with intent, product, max_price, location, certification
    """
    # Step 1: Classify intent
    intent = classify_intent(text)

    # ============================================================================
    # FIX 2: HARD STOP FOR NON-SEARCH QUERIES
    # ============================================================================
    # If not a product search, return immediately WITHOUT parsing
    if intent['intent'] != 'product_search':
        return {
            'intent': intent['intent'],
            'confidence': intent['confidence'],
            'original_query': text
            # ❌ DO NOT include product, max_price, location, certification
            # These fields belong ONLY to product_search intent
        }

    # ============================================================================
    # Step 3: For product searches ONLY, parse normally
    # ============================================================================
    result = rule_parse(text)

    # If product is empty or just the raw text, use GPT fallback
    if not result.get("product") or result["product"] == text:
        try:
            gpt_result = gpt_parse(text)
            # Merge results, preferring GPT for missing fields
            for key, value in gpt_result.items():
                if value and not result.get(key):
                    result[key] = value
        except Exception as e:
            print(f"GPT parsing failed: {e}")

    # Add intent information
    result['intent'] = intent['intent']
    result['confidence'] = intent['confidence']

    return result