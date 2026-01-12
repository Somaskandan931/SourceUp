"""
Orchestrator Module - FIXED
----------------------------
CRITICAL FIX: Only passes raw text to info responses, not parsed_data or session state.
This prevents prompt variable mismatch errors.
"""

import requests
from sourcebot.memory.session import get_session, set_session
from sourcebot.nlu.parser import parse
from sourcebot.responses.info_responses import generate_info_response_enhanced

API = "http://localhost:8000/recommend"
WHAT_IF_API = "http://localhost:8000/what-if"


def handle(sid: str, text: str):
    """
    Handle user query with enhanced explanation support and AI-generated answers.
    FIXED to prevent prompt variable collisions.

    New capabilities:
    - Dynamic informational responses via Groq/Ollama
    - "Why was X ranked higher than Y?"
    - "What if I prioritize price over speed?"
    - "Explain why this supplier was recommended"
    """
    # Parse user input
    parsed_data = parse(text)

    print(f"💬 Message: {text}")
    print(f"🎯 Intent: {parsed_data.get('intent', 'unknown')}")

    intent = parsed_data.get('intent', 'product_search')

    # ========================================================================
    # NEW: EXPLANATION QUERIES
    # ========================================================================
    if intent == 'explanation_request':
        return handle_explanation_request(sid, text, parsed_data)

    # ========================================================================
    # NEW: WHAT-IF QUERIES
    # ========================================================================
    if intent == 'what_if':
        return handle_what_if_query(sid, text, parsed_data)

    # ========================================================================
    # FIX 3: INFORMATIONAL QUERIES - PASS ONLY TEXT (NOT parsed_data)
    # ========================================================================
    if intent == 'information':
        print(f"🤖 Generating AI response for: {text}")
        try:
            # ✅ CRITICAL FIX: Pass ONLY the raw text string
            # ❌ DO NOT pass parsed_data
            # ❌ DO NOT pass session state
            # ❌ DO NOT pass any dictionary with 'product', 'location', etc.
            return generate_info_response_enhanced(text)
        except Exception as e:
            print(f"❌ AI response failed: {e}")
            return (
                "I apologize, but I'm having trouble answering that question right now. "
                "You can try:\n"
                "• Rephrasing your question\n"
                "• Searching for suppliers: 'Find [product] from [location]'\n"
                "• Asking about specific topics like ISO, FDA, MOQ, etc."
            )

    # ========================================================================
    # CONVERSATIONAL
    # ========================================================================
    if intent == 'conversation':
        return (
            "👋 Hi! I'm SourceBot-X. I can help you:\n\n"
            "• Find suppliers: 'Find plastic containers from China'\n"
            "• Answer questions: 'What is ISO 9001?'\n"
            "• Explain decisions: 'Why was supplier A ranked higher?'\n"
            "• Explore trade-offs: 'What if I prioritize price over speed?'\n\n"
            "What would you like to do?"
        )

    # ========================================================================
    # PRODUCT SEARCH (Enhanced with explanations)
    # ========================================================================

    session_state = get_session(sid)
    session_state.update(parsed_data)
    set_session(sid, session_state)

    # Make API call with explanations enabled
    try:
        # Enable explanations by default
        session_state['enable_explanations'] = True

        response = requests.post(API, json=session_state, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Store results in session for follow-up questions
        if 'results' in data:
            session_state['last_results'] = data['results']
            session_state['last_metadata'] = data.get('metadata', {})
            set_session(sid, session_state)

        return data

    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return "Sorry, the recommendation service is currently unavailable."


def handle_explanation_request(sid: str, text: str, parsed_data: dict):
    """
    Handle explanation-specific queries.

    Examples:
    - "Why was supplier A ranked #1?"
    - "Explain the decision for supplier B"
    - "Why did supplier X score higher than Y?"
    """
    session_state = get_session(sid)
    last_results = session_state.get('last_results', [])

    if not last_results:
        return (
            "I don't have any recent search results to explain. "
            "Please search for suppliers first, then ask me to explain the rankings."
        )

    # Try to extract supplier name from query
    text_lower = text.lower()

    # Check if asking about specific rank
    if "rank #" in text_lower or "#" in text_lower:
        try:
            # Extract rank number
            rank = None
            for word in text.split():
                if word.startswith('#'):
                    rank = int(word[1:])
                    break

            if rank and 1 <= rank <= len(last_results):
                result = last_results[rank - 1]
                return format_explanation(result)
        except:
            pass

    # Check if asking about top supplier
    if "top" in text_lower or "#1" in text_lower or "first" in text_lower:
        result = last_results[0]
        return format_explanation(result)

    # Check if asking about specific supplier name
    for result in last_results:
        supplier_name = result['supplier'].lower()
        if supplier_name in text_lower:
            return format_explanation(result)

    # General explanation
    return (
        "I can explain specific rankings! Try:\n"
        "• 'Why was supplier X ranked first?'\n"
        "• 'Explain rank #3'\n"
        "• 'Why did the top supplier score well?'"
    )


def handle_what_if_query(sid: str, text: str, parsed_data: dict):
    """
    Handle what-if scenario queries.

    Examples:
    - "What if I prioritize price over speed?"
    - "What if I increase my budget?"
    - "What if I relax the location requirement?"
    """
    session_state = get_session(sid)
    last_results = session_state.get('last_results', [])

    if not last_results:
        return (
            "I need search results first to run what-if scenarios. "
            "Please search for suppliers, then ask what-if questions."
        )

    text_lower = text.lower()

    # Detect scenario type
    scenario = None
    if "price" in text_lower and ("over" in text_lower or "more" in text_lower):
        scenario = "price_over_speed"
        scenario_name = "Price Focused"
    elif "speed" in text_lower or "fast" in text_lower:
        scenario = "speed_over_speed"
        scenario_name = "Speed Focused"
    elif "quality" in text_lower or "certification" in text_lower:
        scenario = "quality_over_cost"
        scenario_name = "Quality Focused"
    elif "local" in text_lower or "location" in text_lower:
        scenario = "local_over_cheap"
        scenario_name = "Location Focused"
    else:
        return (
            "I can simulate these what-if scenarios:\n"
            "• 'What if I prioritize price over speed?'\n"
            "• 'What if I prioritize speed over price?'\n"
            "• 'What if I prioritize quality over cost?'\n"
            "• 'What if I prioritize local suppliers?'"
        )

    # Call what-if API
    try:
        query = {
            "product": session_state.get('product', ''),
            "constraints": {
                "max_price": session_state.get('max_price'),
                "location": session_state.get('location'),
                "certification": session_state.get('certification')
            },
            "scenario": scenario
        }

        response = requests.post(WHAT_IF_API, json=query, timeout=30)
        response.raise_for_status()

        data = response.json()

        return format_what_if_result(data, scenario_name)

    except Exception as e:
        print(f"What-if simulation failed: {e}")
        return "Sorry, I couldn't run that scenario right now."


def format_explanation(result: dict) -> str:
    """Format a supplier explanation in human-readable text."""

    supplier = result['supplier']
    rank = result['rank']
    score = result['score']

    explanation = f"📊 **Explanation for {supplier} (Rank #{rank})**\n\n"
    explanation += f"**Overall Score:** {score:.3f}\n\n"

    # Decision trace
    if result.get('decision_trace'):
        trace = result['decision_trace']

        explanation += "**Why this ranking:**\n"
        for summary_point in trace.get('summary', []):
            explanation += f"• {summary_point}\n"

        explanation += "\n**Detailed Breakdown:**\n"

        # Top 3 contributions
        contributions = sorted(
            trace['contributions'].items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        )[:3]

        for factor, data in contributions:
            factor_name = factor.replace('_', ' ').title()
            contribution = data['contribution']
            explanation_text = data['explanation']

            explanation += f"• **{factor_name}** (+{contribution:.3f}): {explanation_text}\n"

        # Constraints
        if trace.get('constraints'):
            constraints = trace['constraints']
            if constraints.get('passed_all'):
                explanation += "\n✅ Passes all your constraints\n"

    else:
        # Fallback to basic reasons
        explanation += "**Key factors:**\n"
        for reason in result.get('reasons', [])[:3]:
            explanation += f"• {reason}\n"

    return explanation


def format_what_if_result(data: dict, scenario_name: str) -> str:
    """Format what-if scenario results."""

    result = f"🔮 **What-If Scenario: {scenario_name}**\n\n"

    if 'error' in data:
        return f"❌ {data['error']}"

    ranking_comparison = data.get('ranking_comparison', data)

    # Top changes
    if 'top_10_changes' in ranking_comparison:
        changes = ranking_comparison['top_10_changes'][:5]

        if changes:
            result += "**Biggest Ranking Changes:**\n"
            for change in changes:
                supplier = change['supplier']
                old_rank = change['original_rank']
                new_rank = change['new_rank']
                rank_change = change.get('rank_change', 0)

                if rank_change > 0:
                    emoji = "📈"
                    direction = "up"
                else:
                    emoji = "📉"
                    direction = "down"

                result += f"{emoji} {supplier}: #{old_rank} → #{new_rank} ({abs(rank_change)} positions {direction})\n"
        else:
            result += "No significant ranking changes in this scenario.\n"

    # New top supplier
    if 'new_top_supplier' in ranking_comparison:
        new_top = ranking_comparison['new_top_supplier']
        original_top = ranking_comparison.get('original_top_supplier')

        if new_top != original_top:
            result += f"\n🏆 **New Top Supplier:** {new_top}"
        else:
            result += f"\n🏆 **Top Supplier Unchanged:** {new_top}"

    return result