"""
Orchestrator Module
-------------------
Handles conversation flow and API calls with intent-aware routing.
"""

import requests
from sourcebot.memory.session import get_session, set_session
from sourcebot.nlu.parser import parse
from sourcebot.responses.info_responses import generate_info_response

API = "http://localhost:8000/recommend"


def handle(sid: str, text: str):
    """
    Handle user query with intent-aware routing.

    Args:
        sid: Session ID
        text: User query text

    Returns:
        API response, informational text, or follow-up question
    """
    # Parse user input (includes intent classification)
    parsed_data = parse(text)

    print(f"💬 Message: {text}")
    print(f"🎯 Intent: {parsed_data.get('intent', 'unknown')}")

    # Route based on intent
    intent = parsed_data.get('intent', 'product_search')

    # ========================================================================
    # INFORMATIONAL QUERIES - Return text answer
    # ========================================================================
    if intent == 'information':
        return generate_info_response(text)

    # ========================================================================
    # CONVERSATIONAL - Provide guidance
    # ========================================================================
    if intent == 'conversation':
        return (
            "👋 Hi! I'm SourceBot. I can help you:\n\n"
            "• Find suppliers: 'Find plastic containers from China'\n"
            "• Answer questions: 'What is ISO 9001?'\n"
            "• Get recommendations: 'Show me electronics manufacturers'\n\n"
            "What would you like to do?"
        )

    # ========================================================================
    # PRODUCT SEARCH - Continue with normal flow
    # ========================================================================

    # Get current session state
    session_state = get_session(sid)

    # Update session with parsed data
    session_state.update(parsed_data)
    set_session(sid, session_state)

    # Check if required fields are present
    # (You can make this optional if you want location-agnostic searches)
    # if not session_state.get("location"):
    #     return "Which location do you prefer for suppliers?"

    # Make API call to recommendation service
    try:
        response = requests.post(API, json=session_state, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return "Sorry, the recommendation service is currently unavailable."