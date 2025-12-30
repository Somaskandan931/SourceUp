"""
GPT Fallback Parser with FREE AI Options
-----------------------------------------
Supports multiple free AI providers:
1. Groq (FREE, 30 req/min, very fast) - UPDATED FOR 2025
2. Ollama (FREE, runs locally)
3. Together AI (FREE tier available)
4. OpenAI (paid fallback)
"""

import json
import os
from typing import Dict, Optional
from pydantic import BaseModel, Field

# ============================================================================
# OPTION 1: GROQ (RECOMMENDED - FREE & FAST)
# ============================================================================
# Sign up: https://console.groq.com/
# Free tier: 30 requests/minute, unlimited tokens
# Models: llama-3.3-70b-versatile (recommended), llama-3.1-8b-instant

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed. Run: pip install langchain-groq")

# ============================================================================
# OPTION 2: OLLAMA (FREE LOCAL AI)
# ============================================================================
# Install Ollama: https://ollama.ai/
# Run locally: ollama run llama3.1:8b
# No API key needed, fully offline

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not installed. Run: pip install langchain-ollama")

# ============================================================================
# OPTION 3: OPENAI (PAID - YOUR CURRENT SETUP)
# ============================================================================
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Pydantic model for structured output
class FallbackIntent(BaseModel):
    product: str
    max_price: Optional[float] = None
    location: Optional[str] = None
    certification: Optional[str] = None


# ============================================================================
# CONFIGURATION - CHOOSE YOUR AI PROVIDER
# ============================================================================

AI_PROVIDER = os.getenv('AI_PROVIDER', 'groq')  # 'groq', 'ollama', or 'openai'

def get_llm():
    """Get the configured LLM based on AI_PROVIDER setting."""

    if AI_PROVIDER == 'groq' and GROQ_AVAILABLE:
        # GROQ - FREE & FAST (Recommended)
        # Get API key from: https://console.groq.com/keys
        # UPDATED: Using llama-3.3-70b-versatile (llama-3.1-70b-versatile deprecated Jan 2025)
        return ChatGroq(
            model="llama-3.3-70b-versatile",  # Updated model, better quality
            temperature=0,
            groq_api_key=os.getenv('GROQ_API_KEY')
        )

    elif AI_PROVIDER == 'ollama' and OLLAMA_AVAILABLE:
        # OLLAMA - LOCAL & FREE (No API key needed)
        # Make sure Ollama is running: ollama serve
        return ChatOllama(
            model="llama3.1:8b",  # or "mistral", "gemma2", etc.
            temperature=0
        )

    elif AI_PROVIDER == 'openai' and OPENAI_AVAILABLE:
        # OPENAI - PAID (Your current setup)
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"}
        )

    else:
        raise ValueError(
            f"AI provider '{AI_PROVIDER}' not available. "
            f"Install required package or change AI_PROVIDER environment variable."
        )


# Initialize LLM
try:
    llm = get_llm()
    print(f"✅ Using {AI_PROVIDER.upper()} for query parsing")
except Exception as e:
    print(f"⚠️ Failed to initialize {AI_PROVIDER}: {e}")
    llm = None


# Prompt template (works with all providers)
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a query parser for a supplier search platform. Extract structured data from user queries."),
    (
        "human",
        "Extract the following fields from this query:\n"
        "- product: what the user is searching for\n"
        "- max_price: maximum price if mentioned (as float)\n"
        "- location: country/city if mentioned\n"
        "- certification: ISO/FDA/CE/etc if mentioned\n\n"
        "Query: {text}\n\n"
        "Return ONLY valid JSON with these exact keys: product, max_price, location, certification.\n"
        "Use null for missing values."
    )
])

# Create chain only if LLM is available
if llm:
    chain = prompt | llm
else:
    chain = None


def gpt_parse(text: str) -> Dict:
    """
    Parse text using configured AI provider.
    Falls back to simple extraction if AI fails.

    Returns a dictionary with product, max_price, location, certification.
    """
    if not chain:
        print("⚠️ AI parsing unavailable, using fallback")
        return {
            "product": text,
            "max_price": None,
            "location": None,
            "certification": None
        }

    try:
        response = chain.invoke({"text": text})

        # Handle response content (different providers return differently)
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        # Extract JSON from response
        # Some models wrap JSON in markdown code blocks
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()

        data = json.loads(content)
        intent = FallbackIntent(**data)
        return intent.model_dump()

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed: {e}")
        print(f"   Raw response: {content if 'content' in locals() else 'N/A'}")
        return {
            "product": text,
            "max_price": None,
            "location": None,
            "certification": None
        }

    except Exception as e:
        print(f"⚠️ AI parsing failed: {e}")
        return {
            "product": text,
            "max_price": None,
            "location": None,
            "certification": None
        }


# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================
"""
GROQ SETUP (Recommended - FREE):
---------------------------------
1. Sign up at https://console.groq.com/
2. Get your API key from https://console.groq.com/keys
3. Install: pip install langchain-groq
4. Set environment variable:
   export GROQ_API_KEY='your_api_key_here'  # Linux/Mac
   set GROQ_API_KEY=your_api_key_here       # Windows CMD
   $env:GROQ_API_KEY='your_api_key_here'    # Windows PowerShell
5. Set AI_PROVIDER=groq (or leave as default)

⚠️ MODEL UPDATE (Jan 2025):
   - llama-3.1-70b-versatile → DEPRECATED ❌
   - llama-3.3-70b-versatile → Current model ✅
   - llama-3.1-8b-instant → Still available (faster, smaller)


OLLAMA SETUP (FREE LOCAL):
---------------------------
1. Download from https://ollama.ai/
2. Install and start: ollama serve
3. Pull a model: ollama pull llama3.1:8b
4. Install: pip install langchain-ollama
5. Set environment variable:
   export AI_PROVIDER=ollama  # Linux/Mac
   set AI_PROVIDER=ollama     # Windows CMD


OPENAI SETUP (Paid - Your Current):
------------------------------------
1. Already set up (you're using this now)
2. Set AI_PROVIDER=openai to use OpenAI
3. Add credits at https://platform.openai.com/account/billing


GROQ MODEL OPTIONS (2025):
---------------------------
- llama-3.3-70b-versatile (Recommended - best quality)
- llama-3.3-70b-specdec (Faster inference with speculative decoding)
- llama-3.1-8b-instant (Fastest, smaller model)
- mixtral-8x7b-32768 (Good for longer contexts)


TO SWITCH PROVIDERS:
--------------------
# Use Groq (free, fast)
export AI_PROVIDER=groq
export GROQ_API_KEY=your_groq_key

# Use Ollama (free, local)
export AI_PROVIDER=ollama

# Use OpenAI (paid)
export AI_PROVIDER=openai


TESTING:
--------
python -c "from sourcebot.nlu.gpt_fallback import gpt_parse; print(gpt_parse('Find plastic containers under $2 from China'))"
"""