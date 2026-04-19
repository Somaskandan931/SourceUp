"""
Chat API Endpoint (Enhanced with Groq LLM)
-------------------------------------------------
Conversational interface for supplier search with AI assistance.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import traceback
import os
from groq import Groq
from sourcebot.memory.session import get_session, set_session
from sourcebot.nlu.parser import parse
from backend.app.models.retriever import retrieve
from backend.app.models.ranker import score
from backend.app.services.explanation import explain

router = APIRouter()

# Groq client will be initialized lazily
groq_client = None

def get_groq_client():
    """Get or initialize Groq client."""
    global groq_client
    if groq_client is None:
        try:
            # Try to get API key from environment
            api_key = os.getenv("GROQ_API_KEY")

            # Debug output
            print(f"🔍 Attempting to initialize Groq client...")
            print(f"🔍 API Key found: {bool(api_key)}")
            if api_key:
                print(f"🔍 API Key starts with: {api_key[:10]}...")
                print(f"🔍 API Key length: {len(api_key)}")

            if not api_key or api_key.strip() == "":
                print("❌ GROQ_API_KEY is empty or not set")
                return None

            # Initialize with explicit api_key parameter
            groq_client = Groq(api_key=api_key.strip())
            print("✅ Groq client initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize groq: {e}")
            groq_client = None
    return groq_client


class ChatRequest(BaseModel):
    """Request model for chat"""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Response model for chat"""
    message: str
    suppliers: List[dict]
    session_id: str


async def get_llm_response(query: str, context: dict = None) -> str:
    """
    Get response from Groq LLM for general/information queries.

    Args:
        query: User's question
        context: Session context (location, product preferences, etc.)

    Returns:
        AI-generated response
    """
    try:
        client = get_groq_client()

        # If client initialization failed, return fallback message
        if client is None:
            print("⚠️ Groq client not available, using fallback response")
            return "I apologize, but I'm unable to process information queries at the moment. Please try asking about specific products or suppliers, or contact support for assistance."
        system_prompt = """You are a helpful AI assistant for SourceUP, a B2B supplier sourcing platform.

Your role is to help users with:
- ISO certification requirements (ISO 9001, ISO 14001, ISO 45001, etc.)
- Manufacturing compliance and quality standards
- Export/import regulations and customs requirements
- Product specifications and industry certifications (CE, UL, RoHS, etc.)
- General supplier and sourcing questions
- Industry best practices and standards

Guidelines:
- Be concise and professional (2-3 paragraphs max)
- Provide actionable information
- If asked about specific regulations, mention that users should verify with official sources
- Focus on practical advice relevant to B2B sourcing

Keep responses under 300 words."""

        # Add context if available
        user_message = query
        if context:
            context_parts = []
            if context.get('location'):
                context_parts.append(f"User is interested in suppliers from: {context['location']}")
            if context.get('product'):
                context_parts.append(f"Current product interest: {context['product']}")
            if context.get('certification'):
                context_parts.append(f"Certification requirement: {context['certification']}")

            if context_parts:
                user_message = f"Context:\n" + "\n".join(context_parts) + f"\n\nQuestion: {query}"

        print(f"🤖 Calling Groq LLM...")
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ FIXED: Updated from llama-3.1-70b-versatile
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )

        llm_response = response.choices[0].message.content
        print(f"✅ LLM response generated ({len(llm_response)} chars)")
        return llm_response

    except Exception as e:
        print(f"❌ Error calling Groq API: {e}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again or rephrase your question."


def generate_response_message(session_state: dict, suppliers: List[dict]) -> str:
    """Generate natural language response based on search context."""
    if not suppliers:
        return "I couldn't find any suppliers matching your criteria. Could you provide more details about what you're looking for?"

    product = session_state.get("product", "products")
    location = session_state.get("location", "")
    max_price = session_state.get("max_price")
    certification = session_state.get("certification", "")

    # Build response message
    parts = [f"I found {len(suppliers)} supplier{'s' if len(suppliers) != 1 else ''} for {product}"]

    filters = []
    if location:
        filters.append(f"from {location}")
    if max_price:
        filters.append(f"under ${max_price}")
    if certification:
        filters.append(f"with {certification.upper()} certification")

    if filters:
        parts.append(" " + " ".join(filters))

    parts.append(f". Here are the top {min(5, len(suppliers))} recommendations:")

    return "".join(parts)


@router.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint for conversational supplier search with AI assistance."""
    try:
        print(f"\n💬 Chat message from {request.session_id}: {request.message}")

        # Get current session state
        session_state = get_session(request.session_id)
        print(f"📦 Current session: {session_state}")

        # Parse user message
        try:
            parsed_data = parse(request.message)
            print(f"🧠 Parsed: {parsed_data}")
        except Exception as parse_error:
            print(f"⚠️  Parser error: {parse_error}")
            parsed_data = {"product": request.message, "intent": "unknown"}

        # Determine intent
        intent = parsed_data.get("intent", "product_search")
        print(f"🎯 Intent: {intent}")

        # Update session with new information (only non-None values)
        for key, value in parsed_data.items():
            if value is not None and key != "intent":
                session_state[key] = value

        set_session(request.session_id, session_state)
        print(f"💾 Updated session: {session_state}")

        # Handle different intents
        if intent == "information" or intent == "general":
            # Use Groq LLM for knowledge/information queries
            print(f"ℹ️  Processing as information query with LLM")

            llm_response = await get_llm_response(
                query=request.message,
                context=session_state
            )

            return ChatResponse(
                message=llm_response,
                suppliers=[],
                session_id=request.session_id
            )

        # Product search intent (existing logic)
        # Check if we have enough information to search
        if not session_state.get("product"):
            # Use LLM to generate a helpful prompt
            llm_response = await get_llm_response(
                query="User hasn't specified a product yet. Ask them what product or supplier they're looking for in a friendly way.",
                context=session_state
            )

            return ChatResponse(
                message=llm_response if llm_response and len(llm_response) < 200 else "What type of product or supplier are you looking for?",
                suppliers=[],
                session_id=request.session_id
            )

        # Perform search (don't require location for GlobalSources data)
        print(f"🔍 Searching for: {session_state['product']}")
        candidates = retrieve(session_state["product"], k=50)
        print(f"✅ Found {len(candidates)} candidates")

        if not candidates:
            return ChatResponse(
                message=f"I couldn't find any suppliers for {session_state['product']}. Try a different product name or be more specific.",
                suppliers=[],
                session_id=request.session_id
            )

        # Score and rank candidates
        results = []
        query_dict = {
            "product": session_state.get("product", ""),
            "max_price": session_state.get("max_price"),
            "location": session_state.get("location", "").lower() if session_state.get("location") else "",
            "certification": session_state.get("certification", "").lower() if session_state.get("certification") else ""
        }

        for supplier in candidates:
            try:
                supplier_score = score(supplier, query_dict)
                reasons = explain(supplier, query_dict)

                # Get supplier details
                supplier_name = (
                    supplier.get("supplier name") or
                    supplier.get("company name") or
                    "Unknown Supplier"
                )

                product_name = (
                    supplier.get("product name") or
                    session_state["product"]
                )

                price = supplier.get("price", "Contact for pricing")
                location = supplier.get("supplier location", "")

                results.append({
                    "supplier": str(supplier_name),
                    "product": str(product_name),
                    "score": float(supplier_score),
                    "reasons": reasons,
                    "price": str(price),
                    "location": str(location)
                })
            except Exception as e:
                print(f"⚠️  Error processing supplier: {e}")
                continue

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:5]

        # Generate response message
        response_message = generate_response_message(session_state, top_results)

        print(f"✅ Returning {len(top_results)} suppliers")

        return ChatResponse(
            message=response_message,
            suppliers=top_results,
            session_id=request.session_id
        )

    except Exception as e:
        error_msg = f"Chat processing failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/chat/test")
def test_chat():
    """Test endpoint"""
    try:
        from sourcebot.memory.session import get_session, set_session
        test_sid = "test_123"
        set_session(test_sid, {"test": "data"})
        result = get_session(test_sid)

        return {
            "status": "ok",
            "session_test": result,
            "message": "Chat endpoint is working",
            "groq_configured": bool(os.getenv("GROQ_API_KEY"))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))