"""
Dynamic Informational Response Generator - FIXED
-------------------------------------------------
CRITICAL FIX: Changed prompt variable from {question} to {text} to avoid collisions.
Added general knowledge mode for business concepts.
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

INFO_AI_PROVIDER = os.getenv('INFO_AI_PROVIDER', 'groq')


def get_info_llm():
    """Get LLM for generating informational responses."""

    if INFO_AI_PROVIDER == 'groq' and GROQ_AVAILABLE:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
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
# ENHANCED SYSTEM PROMPT - Specialized for Supplier Sourcing Domain
# ============================================================================

SYSTEM_PROMPT = """You are SourceBot, an expert AI assistant specializing in B2B supplier sourcing, procurement, and international trade.

CORE EXPERTISE AREAS:
══════════════════════
**Manufacturing & Sourcing:**
- Global supplier ecosystems (China, India, Vietnam, Taiwan, Thailand, Bangladesh)
- Product categories: packaging, electronics, textiles, machinery, chemicals, food products
- Manufacturing processes: injection molding, CNC machining, die casting, textile weaving
- MOQ (Minimum Order Quantity) negotiation strategies
- Lead times, production capacity planning

**Certifications & Compliance:**
- Quality: ISO 9001, ISO 13485, TS 16949, AS9100
- Environmental: ISO 14001, RoHS, REACH, Prop 65
- Safety: FDA, CE, UL, CSA, FCC
- Social: BSCI, SEDEX, SA8000, WRAP
- Industry-specific: GMP, HACCP, FSSC 22000

**Trade & Logistics:**
- Incoterms (FOB, CIF, DDP, EXW, FCA)
- Shipping methods (sea freight, air freight, express courier)
- Customs clearance, HS codes, import duties
- Payment terms (L/C, T/T, D/P, D/A, escrow)
- Supply chain risk management

**General Business Concepts:**
- Buyer and seller roles in B2B commerce
- Supply chain fundamentals
- Trade terminology and definitions
- Business relationship building

**Supplier Evaluation:**
- Factory audits and quality control
- Supplier verification methods
- Red flags and warning signs
- Negotiation tactics and best practices
- Building long-term supplier relationships

RESPONSE GUIDELINES:
══════════════════════
1. **Direct Answer First** (1-2 sentences)
   Start with the core answer immediately. No fluff.

2. **Structure for Scanning** (200-400 words max)
   - Use **bold headers** for sections
   - Use • bullet points for lists
   - Keep paragraphs to 2-3 sentences max
   - Numbers for step-by-step processes

3. **Practical & Actionable**
   - Focus on what the user can DO with this information
   - Include real-world examples when relevant
   - Mention typical costs/timelines/ranges where applicable
   
4. **Industry Context**
   - Reference specific countries/regions when relevant
   - Mention industry standards and best practices
   - Note common pitfalls or misconceptions

5. **Call to Action**
   - End with next steps or recommendations
   - If search-related, suggest a specific search query
   - Use 1-2 emojis maximum for visual breaks

TONE: Professional yet conversational, like an experienced sourcing manager mentoring a colleague.

Remember: You're helping businesses make faster, smarter sourcing decisions with confidence!"""


# ============================================================================
# FIX 4: RENAME PROMPT VARIABLE FROM {question} TO {text}
# ============================================================================

info_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{text}")  # ✅ CHANGED FROM {question} to {text}
])


# Create chain
if info_llm:
    info_chain = info_prompt | info_llm
else:
    info_chain = None


# ============================================================================
# FIX 5: GENERAL KNOWLEDGE MODE
# ============================================================================

GENERAL_KNOWLEDGE_KEYWORDS = [
    'buyer', 'seller', 'business',
    'customer', 'market', 'trade',
    'commerce', 'role', 'definition',
    'vendor', 'procurement', 'sourcing'
]


def generate_info_response(question: str) -> str:
    """
    Generate an informational response using AI.
    FIXED to use {text} variable instead of {question}.

    Args:
        question: User's question

    Returns:
        AI-generated response text
    """
    if not info_chain:
        return _fallback_response(question)

    try:
        # ============================================================================
        # FIX 4: INVOKE WITH {text} PARAMETER
        # ============================================================================
        response = info_chain.invoke({"text": question})  # ✅ CHANGED FROM {"question": question}

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
# QUICK ANSWERS (Unchanged but included for completeness)
# ============================================================================

QUICK_ANSWERS = {
    "what is iso": """**ISO (International Organization for Standardization)** develops international standards for quality, safety, and efficiency.

**Common ISO Standards:**
• **ISO 9001**: Quality Management Systems - ensures consistent product/service quality
• **ISO 14001**: Environmental Management - reduces environmental impact
• **ISO 13485**: Medical Devices - quality management for healthcare products
• **ISO 45001**: Occupational Health & Safety - workplace safety management

**Why it matters for sourcing:**
ISO certification shows a supplier follows international best practices for quality control, process management, and continuous improvement. It reduces your risk when sourcing from overseas suppliers.

**How to verify:**
Ask for certificate copies and verify with the issuing body (e.g., SGS, TÜV, BSI).

💡 **Ready to search?** Try: 'Find ISO 9001 certified manufacturers in China'""",

    "who is": """**Buyer and Seller Roles in B2B Commerce:**

**Buyer (Procurement Side):**
• A business or individual purchasing products/services from suppliers
• Responsibilities: sourcing suppliers, negotiating terms, quality control, managing inventory
• Goals: finding reliable suppliers, competitive pricing, ensuring quality standards
• Common titles: Procurement Manager, Sourcing Specialist, Purchasing Agent

**Seller (Supply Side):**
• A business or individual providing products/services to buyers
• Responsibilities: manufacturing/sourcing products, meeting quality standards, managing inventory, fulfilling orders
• Goals: attracting buyers, maintaining product quality, building long-term relationships
• Common titles: Sales Manager, Account Executive, Business Development Manager

**In SourceBot's context:**
• **Buyers** search for suppliers using our platform
• **Sellers (Suppliers/Manufacturers)** are listed in our database and can be discovered by buyers

**How to become a seller:**
1. Register your business on supplier platforms
2. Obtain relevant certifications (ISO, FDA, etc.)
3. Build a strong company profile with product catalogs
4. Get verified through third-party audits
5. Respond promptly to buyer inquiries

💡 **Ready to find suppliers as a buyer?** Try: 'Find [product] from [location]'"""
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
# FIX 5: ENHANCED MAIN FUNCTION (with general knowledge mode)
# ============================================================================

def generate_info_response_enhanced(question: str) -> str:
    """
    Generate response with quick answer fallback and general knowledge mode.
    FIXED to handle general business questions properly.

    Args:
        question: User's question (raw text string ONLY, not a dict)

    Returns:
        AI-generated response text
    """
    # ============================================================================
    # FIX 5: General knowledge mode for basic business concepts
    # ============================================================================
    if any(k in question.lower() for k in GENERAL_KNOWLEDGE_KEYWORDS):
        # Check quick answers first
        quick = get_quick_answer(question)
        if quick:
            return quick
        # Otherwise use AI for general knowledge
        return generate_info_response(question)

    # Check for quick answer first (faster, saves API calls)
    quick = get_quick_answer(question)
    if quick:
        return quick

    # Otherwise use AI
    return generate_info_response(question)