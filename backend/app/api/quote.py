"""
Quote Generation API — SourceUp
--------------------------------
Generates professional RFQ (Request for Quotation) emails using Groq LLM.
Triggered after a supplier is selected from search results.

Endpoints:
    POST /quote/draft   — generate an RFQ draft
    POST /quote/refine  — refine an existing draft with user instructions
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from groq import Groq

router = APIRouter(prefix="/quote", tags=["quote"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QuoteDraftRequest(BaseModel):
    supplier_name: str
    product_name: str
    quantity: Optional[int] = None
    target_price: Optional[float] = None
    delivery_location: Optional[str] = None
    required_certification: Optional[str] = None
    lead_time_days: Optional[int] = None
    buyer_company: Optional[str] = "Our Company"
    buyer_name: Optional[str] = "Procurement Team"
    additional_notes: Optional[str] = None


class QuoteRefineRequest(BaseModel):
    original_draft: str
    refinement_instruction: str   # e.g. "Make it shorter", "Add ISO requirement"


class QuoteResponse(BaseModel):
    subject: str
    body: str
    tone: str


# ---------------------------------------------------------------------------
# LLM client (lazy init, same pattern as chat.py)
# ---------------------------------------------------------------------------

_groq_client = None


def _get_client() -> Optional[Groq]:
    global _groq_client
    if _groq_client is None:
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            return None
        try:
            _groq_client = Groq(api_key=key)
        except Exception as e:
            print(f"⚠️  Groq init failed: {e}")
    return _groq_client


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert B2B procurement specialist helping SMEs write
professional Request for Quotation (RFQ) emails to suppliers.

Your output must ALWAYS be valid JSON with exactly three keys:
  "subject"  — a concise email subject line
  "body"     — the full email body (plain text, professional tone)
  "tone"     — one of: "formal", "semi-formal", "direct"

Rules:
- Keep the body between 150 and 300 words.
- Be specific about quantities, prices, and requirements.
- End with a clear call-to-action asking for a quote within a timeframe.
- Use placeholders [PHONE] and [EMAIL] where contact details are needed.
- Do NOT use markdown in the body — plain text only.
- Output ONLY the JSON object, no extra text."""


def _build_user_prompt(req: QuoteDraftRequest) -> str:
    parts = [f"Supplier: {req.supplier_name}",
             f"Product: {req.product_name}"]
    if req.quantity:
        parts.append(f"Required quantity: {req.quantity} units")
    if req.target_price:
        parts.append(f"Target unit price: USD {req.target_price:.2f}")
    if req.delivery_location:
        parts.append(f"Delivery location: {req.delivery_location}")
    if req.required_certification:
        parts.append(f"Required certification: {req.required_certification}")
    if req.lead_time_days:
        parts.append(f"Maximum lead time: {req.lead_time_days} days")
    if req.buyer_company:
        parts.append(f"Buyer company: {req.buyer_company}")
    if req.buyer_name:
        parts.append(f"Buyer contact name: {req.buyer_name}")
    if req.additional_notes:
        parts.append(f"Additional notes: {req.additional_notes}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/draft", response_model=QuoteResponse)
async def draft_quote(req: QuoteDraftRequest):
    """
    Generate a professional RFQ email draft using Groq LLM.
    Falls back to a structured template if the LLM is unavailable.
    """
    client = _get_client()

    if client:
        try:
            import json
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(req)},
                ],
                temperature=0.4,
                max_tokens=600,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return QuoteResponse(**data)
        except Exception as e:
            print(f"⚠️  LLM quote draft failed: {e} — using fallback template")

    # ── Fallback template (no LLM required) ─────────────────────────
    qty_str   = f"{req.quantity} units of " if req.quantity else ""
    price_str = f" Our target price is USD {req.target_price:.2f} per unit." if req.target_price else ""
    cert_str  = f" The product must carry {req.required_certification} certification." if req.required_certification else ""
    lead_str  = f" We require delivery within {req.lead_time_days} days." if req.lead_time_days else ""
    loc_str   = f" Delivery to {req.delivery_location}." if req.delivery_location else ""

    subject = f"RFQ — {req.product_name} | {req.buyer_company}"
    body = (
        f"Dear {req.supplier_name} Sales Team,\n\n"
        f"My name is {req.buyer_name} from {req.buyer_company}. "
        f"We are interested in sourcing {qty_str}{req.product_name}.{price_str}"
        f"{cert_str}{lead_str}{loc_str}\n\n"
        f"Could you please provide:\n"
        f"  1. Unit price and MOQ\n"
        f"  2. Available certifications\n"
        f"  3. Lead time and payment terms\n"
        f"  4. Sample availability\n\n"
        f"We would appreciate your quotation within 3 business days.\n\n"
        f"Best regards,\n{req.buyer_name}\n{req.buyer_company}\n[EMAIL] | [PHONE]"
    )
    return QuoteResponse(subject=subject, body=body, tone="semi-formal")


@router.post("/refine", response_model=QuoteResponse)
async def refine_quote(req: QuoteRefineRequest):
    """
    Refine an existing quote draft based on user instructions.
    """
    client = _get_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Set GROQ_API_KEY in your .env file."
        )

    try:
        import json
        prompt = (
            f"Here is an existing RFQ email draft:\n\n"
            f"---\n{req.original_draft}\n---\n\n"
            f"Refine it according to this instruction: {req.refinement_instruction}\n\n"
            f"Return the refined version as JSON with 'subject', 'body', and 'tone' keys."
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return QuoteResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {e}")