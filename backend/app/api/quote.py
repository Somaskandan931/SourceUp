"""
Quote Generation API — SourceUp
---------------------------------
Generates professional RFQ (Request for Quotation) emails using Groq LLM.
Now also supports:
  - PDF export of the RFQ with a printable signature block
  - Digital signature embedding (typed name + timestamp, with optional drawn sig)

Endpoints:
    POST /quote/draft       — generate an RFQ draft
    POST /quote/refine      — refine an existing draft
    POST /quote/export-pdf  — export the RFQ as a signed PDF (download)
"""

import os
import io
import json
from datetime import datetime
from html import escape
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq

router = APIRouter(prefix="/quote", tags=["quote"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QuoteDraftRequest(BaseModel):
    supplier_name:          str
    product_name:           str
    quantity:               Optional[int]   = None
    target_price:           Optional[float] = None
    delivery_location:      Optional[str]   = None
    required_certification: Optional[str]   = None
    lead_time_days:         Optional[int]   = None
    buyer_company:          Optional[str]   = "Our Company"
    buyer_name:             Optional[str]   = "Procurement Team"
    additional_notes:       Optional[str]   = None


class QuoteRefineRequest(BaseModel):
    original_draft:          str
    refinement_instruction:  str


class QuoteResponse(BaseModel):
    subject: str
    body:    str
    tone:    str


class QuotePDFRequest(BaseModel):
    """Payload to export the RFQ as a signed PDF."""
    subject:              str
    body:                 str
    tone:                 str
    supplier_name:        Optional[str]   = ""
    product_name:         Optional[str]   = ""
    quantity:             Optional[int]   = None
    target_price:         Optional[float] = None
    delivery_location:    Optional[str]   = ""
    required_certification: Optional[str] = ""
    lead_time_days:       Optional[int]   = None
    buyer_name:           Optional[str]   = "Procurement Team"
    buyer_company:        Optional[str]   = "Our Company"
    buyer_title:          Optional[str]   = ""
    buyer_email:          Optional[str]   = ""
    buyer_phone:          Optional[str]   = ""
    # Digital signature
    digital_signature:    Optional[str]   = None   # typed name = "digital signature"
    signature_date:       Optional[str]   = None   # ISO date string, defaults to today


def _pdf_text(value) -> str:
    """Escape text for ReportLab Paragraph and normalize fragile punctuation."""
    text = "" if value is None else str(value)
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u20b9": "INR",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return escape(text, quote=False)


def _plain_text(value) -> str:
    return _pdf_text(value).replace("<br/>", "\n")


# ---------------------------------------------------------------------------
# LLM client
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
    parts = [f"Supplier: {req.supplier_name}", f"Product: {req.product_name}"]
    if req.quantity:               parts.append(f"Required quantity: {req.quantity} units")
    if req.target_price:           parts.append(f"Target unit price: USD {req.target_price:.2f}")
    if req.delivery_location:      parts.append(f"Delivery location: {req.delivery_location}")
    if req.required_certification: parts.append(f"Required certification: {req.required_certification}")
    if req.lead_time_days:         parts.append(f"Maximum lead time: {req.lead_time_days} days")
    if req.buyer_company:          parts.append(f"Buyer company: {req.buyer_company}")
    if req.buyer_name:             parts.append(f"Buyer contact name: {req.buyer_name}")
    if req.additional_notes:       parts.append(f"Additional notes: {req.additional_notes}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/draft", response_model=QuoteResponse)
async def draft_quote(req: QuoteDraftRequest):
    """Generate a professional RFQ email draft using Groq LLM."""
    client = _get_client()

    if client:
        try:
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
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return QuoteResponse(**data)
        except Exception as e:
            print(f"⚠️  LLM quote draft failed: {e} — using fallback template")

    # Fallback template
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
    """Refine an existing quote draft based on user instructions."""
    client = _get_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Set GROQ_API_KEY in your .env file."
        )

    try:
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


# ---------------------------------------------------------------------------
# PDF Export with Digital Signature
# ---------------------------------------------------------------------------

@router.post("/export-pdf-old")
async def export_pdf(req: QuotePDFRequest):
    """
    Export the RFQ as a professionally formatted PDF.

    Features:
      - Company header with document metadata
      - Full email body rendered in the PDF
      - Printable signature block with date/title/company
      - Digital signature: if `digital_signature` is provided, renders a
        styled "Digitally Signed By" block with the name and timestamp —
        this is a typed/acknowledged signature, not a cryptographic one.
        For legal e-signatures, integrate DocuSign / Aadhaar eSign separately.

    Returns: application/pdf binary stream for download.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="reportlab is not installed. Run: pip install reportlab"
        )

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    brand_color = colors.HexColor("#1a56db")   # SourceUp blue

    style_title = ParagraphStyle(
        "RFQTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=brand_color,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    style_meta = ParagraphStyle(
        "RFQMeta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=2,
    )
    style_section = ParagraphStyle(
        "RFQSection",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=brand_color,
        spaceBefore=10,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    style_body = ParagraphStyle(
        "RFQBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=16,
        spaceAfter=6,
    )
    style_sig_name = ParagraphStyle(
        "SigName",
        parent=styles["Normal"],
        fontSize=14,
        textColor=brand_color,
        fontName="Helvetica-BoldOblique",
        spaceAfter=2,
    )
    style_sig_label = ParagraphStyle(
        "SigLabel",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        spaceAfter=1,
    )
    style_footer = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )

    sig_date = req.signature_date or datetime.utcnow().strftime("%d %B %Y")
    doc_ref  = f"RFQ-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    story    = []

    # ── Header bar ──────────────────────────────────────────────────────────
    story.append(Paragraph("SourceUp", style_title))
    story.append(Paragraph("AI-Powered Procurement Platform", style_meta))
    story.append(HRFlowable(width="100%", thickness=2, color=brand_color, spaceAfter=8))

    # Document metadata table
    meta_data = [
        ["Document Type:", "Request for Quotation (RFQ)"],
        ["Reference No.:", doc_ref],
        ["Date Issued:",   sig_date],
        ["Issued By:",     f"{req.buyer_name or ''}, {req.buyer_company or ''}"],
    ]
    meta_table = Table(meta_data, colWidths=[4 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── Subject ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Subject", style_section))
    story.append(Paragraph(req.subject, ParagraphStyle(
        "SubjectText", parent=styles["Normal"],
        fontSize=11, fontName="Helvetica-Bold", spaceAfter=4
    )))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=8))

    # ── Email body ──────────────────────────────────────────────────────────
    story.append(Paragraph("RFQ Details", style_section))
    for line in req.body.split("\n"):
        text = line.strip()
        if text:
            story.append(Paragraph(text, style_body))
        else:
            story.append(Spacer(1, 0.2 * cm))

    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=10))

    # ── Signature block ──────────────────────────────────────────────────────
    story.append(Paragraph("Authorisation & Signature", style_section))
    story.append(Spacer(1, 0.2 * cm))

    if req.digital_signature:
        # ── Digital signature panel ─────────────────────────────────────────
        ds_bg = colors.HexColor("#eef3ff")
        ds_border = brand_color

        ds_inner = [
            [Paragraph("✦ Digitally Signed By", style_sig_label)],
            [Paragraph(req.digital_signature, style_sig_name)],
            [Paragraph(f"Date: {sig_date} (UTC)  |  Acknowledged via SourceUp RFQ Wizard", style_sig_label)],
            [Paragraph(f"Ref: {doc_ref}", style_sig_label)],
        ]
        ds_table = Table(ds_inner, colWidths=[16 * cm])
        ds_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), ds_bg),
            ("BOX",          (0, 0), (-1, -1), 1.5, ds_border),
            ("LEFTPADDING",  (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING",   (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ]))
        story.append(ds_table)
        story.append(Spacer(1, 0.5 * cm))

        # Note about legal standing
        story.append(Paragraph(
            "⚠ This document carries a typed digital acknowledgement. "
            "For legally binding e-signatures under IT Act 2000 (India) or eIDAS (EU), "
            "please use a certified e-signature provider (e.g., Aadhaar eSign, DocuSign).",
            ParagraphStyle("Note", parent=styles["Normal"], fontSize=8,
                           textColor=colors.HexColor("#888888"), leading=11)
        ))
    else:
        # ── Printable wet-ink signature block ────────────────────────────────
        sig_data = [
            ["Authorised Signatory", "", "Date"],
            ["", "", ""],
            ["", "", ""],
            [req.buyer_name or "________________________",
             "",
             sig_date],
            [req.buyer_title or "________________________",
             "",
             ""],
            [req.buyer_company or "________________________",
             "",
             ""],
        ]
        sig_table = Table(sig_data, colWidths=[8 * cm, 2 * cm, 6 * cm])
        sig_table.setStyle(TableStyle([
            ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",   (0, 3), (-1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("TEXTCOLOR",  (0, 0), (-1, 0),  colors.grey),
            ("LINEABOVE",  (0, 3), (0, 3),   0.5, colors.black),
            ("LINEABOVE",  (2, 3), (2, 3),   0.5, colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(sig_table)
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            "Print this document, sign in the space above, and return a scanned copy to the supplier.",
            ParagraphStyle("Note", parent=styles["Normal"], fontSize=8,
                           textColor=colors.grey, leading=11)
        ))

    story.append(Spacer(1, 1 * cm))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=4))
    story.append(Paragraph(
        f"Generated by SourceUp RFQ Wizard  •  {sig_date}  •  {doc_ref}  •  Confidential",
        style_footer
    ))

    doc.build(story)
    buffer.seek(0)

    filename = f"RFQ_{doc_ref}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/export-pdf")
async def export_pdf_formal(req: QuotePDFRequest):
    """Export a print-ready, formal RFQ PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    except ImportError:
        raise HTTPException(status_code=500, detail="reportlab is not installed. Run: pip install reportlab")

    buffer = io.BytesIO()
    doc_ref = f"RFQ-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    issued_date = req.signature_date or datetime.utcnow().strftime("%d %B %Y")

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=f"SourceUp {doc_ref}",
        author=req.buyer_company or "SourceUp",
    )

    styles = getSampleStyleSheet()
    blue = colors.HexColor("#1a56db")
    dark = colors.HexColor("#111827")
    muted = colors.HexColor("#6b7280")
    border = colors.HexColor("#d1d5db")
    soft = colors.HexColor("#f3f4f6")

    title_style = ParagraphStyle(
        "FormalTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=dark,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    brand_style = ParagraphStyle(
        "FormalBrand",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=blue,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "FormalSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=dark,
        spaceBefore=9,
        spaceAfter=5,
    )
    normal_style = ParagraphStyle(
        "FormalNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=dark,
    )
    small_style = ParagraphStyle(
        "FormalSmall",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=muted,
    )
    signature_style = ParagraphStyle(
        "FormalSignature",
        parent=styles["Normal"],
        fontName="Helvetica-BoldOblique",
        fontSize=13,
        leading=16,
        textColor=blue,
    )

    def p(value, style=normal_style):
        return Paragraph(_pdf_text(value), style)

    def label(value):
        return Paragraph(f"<b>{_pdf_text(value)}</b>", normal_style)

    table_style = TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, border),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("BACKGROUND", (0, 0), (0, -1), soft),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])

    story = [
        Paragraph("REQUEST FOR QUOTATION", title_style),
        Paragraph("Generated by SourceUp Procurement Platform", brand_style),
        HRFlowable(width="100%", thickness=1.2, color=blue, spaceAfter=8),
    ]

    meta_table = Table([
        [label("Reference No."), p(doc_ref)],
        [label("Date Issued"), p(issued_date)],
        [label("Buyer Company"), p(req.buyer_company or "Our Company")],
        [label("Buyer Contact"), p(req.buyer_name or "Procurement Team")],
        [label("Contact Email"), p(req.buyer_email or "To be provided")],
        [label("Contact Phone"), p(req.buyer_phone or "To be provided")],
    ], colWidths=[4 * cm, 12.4 * cm])
    meta_table.setStyle(table_style)
    story.extend([meta_table, Spacer(1, 0.3 * cm)])

    story.append(Paragraph("Subject", section_style))
    story.append(Paragraph(f"<b>{_pdf_text(req.subject)}</b>", normal_style))

    summary_table = Table([
        [label("Supplier"), p(req.supplier_name or "To be confirmed")],
        [label("Product / Item"), p(req.product_name or "As per RFQ details")],
        [label("Quantity"), p(req.quantity if req.quantity is not None else "To be confirmed")],
        [label("Target Price"), p(f"USD {req.target_price:.2f}" if req.target_price is not None else "To be quoted")],
        [label("Delivery Location"), p(req.delivery_location or "To be confirmed")],
        [label("Certification Required"), p(req.required_certification or "As applicable")],
        [label("Required Lead Time"), p(f"{req.lead_time_days} days" if req.lead_time_days else "To be quoted")],
    ], colWidths=[4 * cm, 12.4 * cm])
    summary_table.setStyle(table_style)
    story.extend([Paragraph("RFQ Summary", section_style), summary_table, Spacer(1, 0.25 * cm)])

    story.append(Paragraph("Formal RFQ Message", section_style))
    for line in (req.body or "").splitlines():
        if line.strip():
            story.append(p(line.strip()))
        else:
            story.append(Spacer(1, 0.12 * cm))

    response_rows = [
        [p("1."), p("Unit price, MOQ, applicable taxes, and quotation validity period.")],
        [p("2."), p("Lead time, shipping terms, delivery schedule, and packaging details.")],
        [p("3."), p("Available certifications, compliance documents, and sample policy.")],
        [p("4."), p("Payment terms, warranty terms, and after-sales support details.")],
    ]
    response_table = Table(response_rows, colWidths=[0.6 * cm, 15.8 * cm])
    response_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.extend([Paragraph("Requested Supplier Response", section_style), response_table])

    story.extend([Spacer(1, 0.35 * cm), HRFlowable(width="100%", thickness=0.5, color=border, spaceAfter=8)])
    story.append(Paragraph("Authorisation and Signature", section_style))

    if req.digital_signature:
        sig_table = Table([
            [Paragraph("Digitally Signed By", small_style)],
            [Paragraph(_pdf_text(req.digital_signature), signature_style)],
            [p(f"Date: {issued_date} | Reference: {doc_ref}", small_style)],
        ], colWidths=[16.4 * cm])
        sig_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3ff")),
            ("BOX", (0, 0), (-1, -1), 1, blue),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(sig_table)
    else:
        sig_table = Table([
            ["Authorised Signatory", "", "Date"],
            ["", "", ""],
            ["", "", ""],
            [_plain_text(req.buyer_name) or "________________________", "", _plain_text(issued_date)],
            [_plain_text(req.buyer_title) or "Designation", "", ""],
            [_plain_text(req.buyer_company) or "Company", "", ""],
        ], colWidths=[8 * cm, 2 * cm, 6.4 * cm])
        sig_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), muted),
            ("LINEABOVE", (0, 3), (0, 3), 0.7, colors.black),
            ("LINEABOVE", (2, 3), (2, 3), 0.7, colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(sig_table)
        story.append(Spacer(1, 0.2 * cm))
        story.append(p("Print this RFQ, sign in the space above, and share the signed copy with the supplier.", small_style))

    story.extend([
        Spacer(1, 0.5 * cm),
        HRFlowable(width="100%", thickness=0.5, color=border, spaceAfter=4),
        Paragraph(_pdf_text(f"Generated by SourceUp RFQ Wizard | {issued_date} | {doc_ref} | Confidential"), small_style),
    ])

    try:
        doc.build(story)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    buffer.seek(0)
    filename = f"SourceUp_{doc_ref}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
