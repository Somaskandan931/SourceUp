"""
Dynamic Informational Response Generator
-----------------------------------------
Uses Groq/Ollama/OpenAI to generate informational answers dynamically.
Create this as: sourcebot/responses/info_responses.py
"""

import os
from typing import Optional

# Try to import available LLM providers
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

INFO_AI_PROVIDER = os.getenv('INFO_AI_PROVIDER', 'groq')  # 'groq', 'ollama', or 'openai'


def get_info_llm():
    """Get LLM for generating informational responses."""

    if INFO_AI_PROVIDER == 'groq' and GROQ_AVAILABLE:
        return ChatGroq(
            model="llama-3.3-70b-versatile",  # Best for detailed explanations
            temperature=0.3,  # Slightly creative for better explanations
            groq_api_key=os.getenv('GROQ_API_KEY')
        )

    elif INFO_AI_PROVIDER == 'ollama' and OLLAMA_AVAILABLE:
        return ChatOllama(
            model="llama3.1:8b",
            temperature=0.3
        )

    elif INFO_AI_PROVIDER == 'openai' and OPENAI_AVAILABLE:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3
        )

    else:
        raise ValueError(
            f"Info AI provider '{INFO_AI_PROVIDER}' not available. "
            f"Install required package or change INFO_AI_PROVIDER."
        )


# Initialize LLM
try:
    info_llm = get_info_llm()
    print(f"✅ Using {INFO_AI_PROVIDER.upper()} for informational responses")
except Exception as e:
    print(f"⚠️ Failed to initialize info LLM: {e}")
    info_llm = None


# ============================================================================
# SYSTEM PROMPT - Guides the AI to give supplier-sourcing focused answers
# ============================================================================

SYSTEM_PROMPT = """You are SourceBot, an AI assistant specialized in helping businesses with supplier sourcing and procurement.

Your expertise includes:
- International trade and supplier sourcing (China, India, Vietnam, etc.)
- Product certifications (ISO 9001, ISO 14001, FDA, CE, RoHS, UL)
- Manufacturing and supply chain management
- Import/export processes and customs clearance
- Quality control and supplier evaluation criteria
- Pricing strategies, MOQ negotiations, and payment terms
- Logistics, shipping methods (sea freight, air freight, express)
- Business types (manufacturers vs trading companies)
- Compliance and legal requirements for international trade

When answering questions:
1. Be concise but comprehensive (aim for 200-400 words maximum)
2. Use bullet points and clear structure for easy scanning
3. Focus on practical, actionable information that helps decision-making
4. Include specific examples and real-world scenarios when relevant
5. If the question relates to finding suppliers, suggest doing a product search after answering
6. Use professional but friendly, conversational tone
7. Use emoji sparingly (1-2 per response maximum) for visual breaks
8. Avoid overly technical jargon unless necessary; explain terms simply
9. Prioritize answering the specific question asked before providing additional context

Format your responses with:
- **Bold headers** for main sections
- • Bullet points for lists and key information
- Clear, scannable structure with short paragraphs
- Numbers for step-by-step processes
- Keep paragraphs to 2-3 sentences maximum

Answer style guidelines:
- Start with a direct answer to the question in the first 1-2 sentences
- Then provide supporting details and context
- End with actionable next steps or recommendations when relevant
- If asked "what is X", give a brief definition first, then elaborate
- If asked "how to", provide a numbered step-by-step process
- If asked about comparisons, use clear comparison format (vs, differences)

Remember: You're helping businesses make informed sourcing decisions quickly and confidently!"""


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

info_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


# Create chain
if info_llm:
    info_chain = info_prompt | info_llm
else:
    info_chain = None


# ============================================================================
# MAIN FUNCTION - Generate Dynamic Responses
# ============================================================================

def generate_info_response(question: str) -> str:
    """
    Generate an informational response using AI.

    Args:
        question: User's question

    Returns:
        AI-generated response text
    """
    if not info_chain:
        return _fallback_response(question)

    try:
        # Invoke the AI
        response = info_chain.invoke({"question": question})

        # Extract content
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)

        # Add a helpful footer
        footer = _get_contextual_footer(question)

        return f"{answer}\n\n{footer}"

    except Exception as e:
        print(f"⚠️ Info response generation failed: {e}")
        return _fallback_response(question)


def _get_contextual_footer(question: str) -> str:
    """Add a contextual call-to-action based on the question."""
    question_lower = question.lower()

    if any(cert in question_lower for cert in ['iso', 'fda', 'ce', 'certification']):
        return "💡 **Want to find certified suppliers?** Try: 'Find ISO certified manufacturers in China'"

    elif 'supplier' in question_lower or 'choose' in question_lower or 'select' in question_lower:
        return "💡 **Ready to search for suppliers?** Try: 'Find [your product] from [location]'"

    elif 'shipping' in question_lower or 'lead time' in question_lower:
        return "💡 **Need suppliers with fast delivery?** Specify your location preferences in your search."

    elif 'moq' in question_lower or 'minimum order' in question_lower:
        return "💡 **Looking for low MOQ suppliers?** Include this in your search criteria."

    elif 'payment' in question_lower:
        return "💡 **Ready to start sourcing?** Search for verified suppliers with secure payment options."

    else:
        return "💡 **Have more questions or ready to search?** Just ask!"


def _fallback_response(question: str) -> str:
    """Fallback when AI is unavailable."""
    return """I apologize, but I'm having trouble generating a detailed response right now.

**Here's what I can help with:**
• Find suppliers for specific products
• Answer questions about certifications and compliance
• Explain sourcing processes and best practices
• Compare suppliers based on your criteria

**Try asking:**
• "Find biodegradable food containers from China"
• "What is ISO 9001 certification?" (I'll try again)
• "Show me electronics manufacturers with FDA approval"

Or rephrase your question and I'll try again!"""


# ============================================================================
# QUICK ANSWER MODE (Optional - For Very Common Questions)
# ============================================================================

QUICK_ANSWERS = {
    "what is iso": """**ISO (International Organization for Standardization)** develops international standards for quality, safety, and efficiency.

**Common ISO Standards:**
• **ISO 9001**: Quality Management Systems
• **ISO 14001**: Environmental Management
• **ISO 45001**: Occupational Health & Safety

**Why it matters for sourcing:**
ISO certification shows a supplier follows international best practices for quality control and process management.

💡 Search: 'Find ISO 9001 certified manufacturers'""",

    "what is fda": """**FDA (U.S. Food and Drug Administration)** regulates food, drugs, medical devices, and cosmetics in the United States.

**For Suppliers:**
• Products entering the US market must meet FDA regulations
• Facilities may need FDA registration
• Requires compliance certificates and testing

💡 Search: 'Find FDA approved food packaging suppliers'""",

    "what is moq": """**MOQ (Minimum Order Quantity)** is the smallest amount a supplier will produce or sell in one order.

**Why suppliers have MOQs:**
• Production setup costs
• Material bulk purchasing efficiency
• Quality control consistency

**Typical ranges:**
• Custom products: 1,000-10,000 units
• Generic products: Often negotiable
• Electronics: 100-1,000 units

💡 Tip: Ask suppliers if they can reduce MOQ for trial orders."""
}


def get_quick_answer(question: str) -> Optional[str]:
    """Check if there's a quick answer for common questions."""
    question_lower = question.lower().strip()

    # Remove question marks and extra words
    question_clean = question_lower.replace('?', '').strip()

    for key, answer in QUICK_ANSWERS.items():
        if key in question_clean:
            return answer

    return None


# ============================================================================
# ENHANCED MAIN FUNCTION (with quick answers)
# ============================================================================

def generate_info_response_enhanced(question: str) -> str:
    """
    Generate response with quick answer fallback.
    Use this instead of generate_info_response for better performance.
    """
    # Check for quick answer first (faster, saves API calls)
    quick = get_quick_answer(question)
    if quick:
        return quick

    # Otherwise use AI
    return generate_info_response(question)