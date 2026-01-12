"""
GPT Fallback Parser - FIXED
----------------------------
Enhanced with domain-specific prompt engineering for supplier sourcing.
ONLY triggers for product_search intent.
"""

import json
import os
from typing import Dict, Optional
from pydantic import BaseModel, Field

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed. Run: pip install langchain-groq")

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


class FallbackIntent(BaseModel):
    product: str
    max_price: Optional[float] = None
    location: Optional[str] = None
    certification: Optional[str] = None


AI_PROVIDER = os.getenv('AI_PROVIDER', 'groq')

def get_llm():
    """Get the configured LLM based on AI_PROVIDER setting."""
    if AI_PROVIDER == 'groq' and GROQ_AVAILABLE:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
    elif AI_PROVIDER == 'ollama' and OLLAMA_AVAILABLE:
        return ChatOllama(model="llama3.1:8b", temperature=0)
    elif AI_PROVIDER == 'openai' and OPENAI_AVAILABLE:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        raise ValueError(f"AI provider '{AI_PROVIDER}' not available.")


try:
    llm = get_llm()
    print(f"✅ Using {AI_PROVIDER.upper()} for query parsing")
except Exception as e:
    print(f"⚠️ Failed to initialize {AI_PROVIDER}: {e}")
    llm = None


from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# SPECIALIZED SYSTEM PROMPT - Domain-Specific for Supplier Sourcing
# ============================================================================

SYSTEM_PROMPT = """You are a specialized query parser for SourceBot, a B2B supplier sourcing platform.

Your task: Extract structured search parameters from natural language queries about finding suppliers, manufacturers, and products.

DOMAIN KNOWLEDGE:
- Common products: containers, packaging, electronics, machinery, textiles, plastics, metals
- Key locations: China, India, Vietnam, USA, UK, Germany, Taiwan, South Korea, Thailand
- Certifications: ISO 9001, ISO 14001, ISO 13485, FDA, CE, RoHS, UL, BSCI, SEDEX
- Price formats: "$X", "under $X", "below X dollars", "max $X", "X USD"

EXTRACTION RULES:
1. **product**: The item/service being sourced
   - Include material type (plastic, metal, fabric)
   - Include product category (containers, electronics, machinery)
   - Clean up filler words like "find", "search for", "looking for"
   
2. **max_price**: Maximum budget (as float)
   - Extract from: "under $50", "$100 max", "below 200 dollars"
   - Convert to numeric value only
   - Set to null if not mentioned
   
3. **location**: Preferred supplier country/region
   - Extract from: "from China", "in India", "Vietnam suppliers"
   - Normalize to country name (China, not "chinese")
   - Set to null if not mentioned
   
4. **certification**: Required certifications
   - Common: ISO, FDA, CE, RoHS, UL
   - Extract the abbreviation (e.g., "ISO" not "ISO certified")
   - Set to null if not mentioned

EXAMPLES:
Query: "Find biodegradable food containers under $2 from China with FDA approval"
→ {
  "product": "biodegradable food containers",
  "max_price": 2.0,
  "location": "China",
  "certification": "FDA"
}

Query: "Looking for electronics manufacturers in Vietnam"
→ {
  "product": "electronics",
  "max_price": null,
  "location": "Vietnam",
  "certification": null
}

Query: "ISO 9001 certified plastic suppliers"
→ {
  "product": "plastic",
  "max_price": null,
  "location": null,
  "certification": "ISO"
}

IMPORTANT:
- Return ONLY valid JSON
- Use null (not "null" string) for missing values
- Keep product names concise but descriptive
- Normalize certification names to uppercase abbreviations
- Extract numeric price values without currency symbols"""

# Updated prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{text}")
])

# Create chain
if llm:
    chain = prompt | llm
else:
    chain = None


def gpt_parse(text: str) -> Dict:
    """
    Parse text using specialized supplier sourcing prompt.
    ⚠️ SHOULD ONLY BE CALLED FOR PRODUCT_SEARCH INTENT

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

        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        # Extract JSON from response
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