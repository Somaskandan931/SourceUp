// components/QuoteModal.jsx
import React, { useState } from 'react';
import { C, btn, inputStyle, modalOverlay, modalBox, mono } from '../styles/tokens';
import { API, post } from '../utils/api';
import { resolveSupplierName, resolveProductName } from '../utils/supplier';
import Spinner from './Spinner';

export default function QuoteModal({ supplier, onClose }) {
  const [step, setStep]     = useState(1);
  const [qty,     setQty]   = useState('');
  const [price,   setPrice] = useState('');
  const [loc,     setLoc]   = useState('');
  const [cert,    setCert]  = useState('');
  const [lead,    setLead]  = useState('');
  const [notes,   setNotes] = useState('');
  const [tone,    setTone]  = useState('semi-formal');
  const [draft,   setDraft] = useState(null);
  const [refine,  setRefine]= useState('');
  const [loading, setLoad]  = useState(false);
  const [exporting, setExporting] = useState(false);
  const [pdfError, setPdfError] = useState('');
  const [copied,  setCopied]= useState(false);

  const supplierName = resolveSupplierName(supplier) || 'this supplier';
  const productName  = resolveProductName(supplier, '');

  const tones = [
    { id: 'formal',      label: '🎩 Formal',      desc: 'Corporate, strict' },
    { id: 'semi-formal', label: '💼 Semi-formal',  desc: 'Professional yet warm' },
    { id: 'direct',      label: '⚡ Direct',       desc: 'Concise & punchy' },
  ];

  const generate = async () => {
    setLoad(true);
    setPdfError('');
    const data = await post('/quote/draft', {
      supplier_name:          supplierName,
      product_name:           productName,
      quantity:               qty   ? parseInt(qty)     : null,
      target_price:           price ? parseFloat(price) : null,
      delivery_location:      loc   || null,
      required_certification: cert  || null,
      lead_time_days:         lead  ? parseInt(lead)    : null,
      buyer_company:          localStorage.getItem('su_company') || 'Our Company',
      buyer_name:             localStorage.getItem('su_email')   || 'Procurement Team',
      additional_notes:       (notes ? notes + '. ' : '') + `Preferred tone: ${tone}.`,
    });
    if (data.error) {
      setPdfError(data.detail || 'Unable to generate RFQ draft.');
      setLoad(false);
      return;
    }
    setDraft(data); setStep(2); setLoad(false);
  };

  const doRefine = async () => {
    if (!draft || !refine.trim()) return;
    setLoad(true);
    const data = await post('/quote/refine', {
      original_draft:         `${draft.subject}\n\n${draft.body}`,
      refinement_instruction: refine,
    });
    setDraft(data); setRefine(''); setLoad(false);
  };

  const copy = () => {
    navigator.clipboard.writeText(`Subject: ${draft.subject}\n\n${draft.body}`);
    setCopied(true); setTimeout(() => setCopied(false), 2000);
  };

  const openEmail = () => {
    const subject = encodeURIComponent(draft.subject);
    const body    = encodeURIComponent(draft.body);
    window.open(`mailto:?subject=${subject}&body=${body}`);
  };

  const buildPdfPayload = () => ({
    subject: draft.subject,
    body: draft.body,
    tone: draft.tone || tone,
    supplier_name: supplierName,
    product_name: productName,
    quantity: qty ? parseInt(qty) : null,
    target_price: price ? parseFloat(price) : null,
    delivery_location: loc || '',
    required_certification: cert || '',
    lead_time_days: lead ? parseInt(lead) : null,
    buyer_name: localStorage.getItem('su_email') || 'Procurement Team',
    buyer_company: localStorage.getItem('su_company') || 'Our Company',
    buyer_title: '',
    buyer_email: localStorage.getItem('su_email') || '',
    buyer_phone: '',
    digital_signature: null,
    signature_date: new Date().toISOString().split('T')[0],
  });

  const escapeHtml = (value) => String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');

  const openPrintableFallback = () => {
    if (!draft) return false;
    const today = new Date().toLocaleDateString();
    const printable = window.open('', '_blank');
    if (!printable) return false;

    printable.document.write(`
      <!doctype html>
      <html>
        <head>
          <title>SourceUp RFQ</title>
          <style>
            * { box-sizing: border-box; }
            body { font-family: Arial, sans-serif; color: #111827; margin: 32px; line-height: 1.5; }
            h1 { text-align: center; font-size: 22px; margin: 0; letter-spacing: 0.04em; }
            .brand { text-align: center; color: #1a56db; font-weight: 700; margin: 4px 0 18px; }
            .rule { border-top: 2px solid #1a56db; margin-bottom: 18px; }
            h2 { font-size: 13px; margin: 22px 0 8px; text-transform: uppercase; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 14px; }
            td { border: 1px solid #d1d5db; padding: 8px 10px; vertical-align: top; font-size: 13px; }
            td:first-child { width: 28%; background: #f3f4f6; font-weight: 700; }
            .subject { font-weight: 700; margin-bottom: 14px; }
            .body { white-space: pre-wrap; font-size: 13px; }
            .signature { display: grid; grid-template-columns: 1fr 180px; gap: 56px; margin-top: 48px; }
            .line { border-top: 1px solid #111827; padding-top: 8px; font-size: 13px; }
            .footer { border-top: 1px solid #d1d5db; margin-top: 36px; padding-top: 8px; color: #6b7280; font-size: 11px; }
            @media print { body { margin: 18mm; } button { display: none; } }
          </style>
        </head>
        <body>
          <h1>REQUEST FOR QUOTATION</h1>
          <div class="brand">Generated by SourceUp Procurement Platform</div>
          <div class="rule"></div>
          <table>
            <tr><td>Date Issued</td><td>${escapeHtml(today)}</td></tr>
            <tr><td>Buyer Company</td><td>${escapeHtml(localStorage.getItem('su_company') || 'Our Company')}</td></tr>
            <tr><td>Buyer Contact</td><td>${escapeHtml(localStorage.getItem('su_email') || 'Procurement Team')}</td></tr>
          </table>
          <h2>Subject</h2>
          <div class="subject">${escapeHtml(draft.subject)}</div>
          <h2>RFQ Summary</h2>
          <table>
            <tr><td>Supplier</td><td>${escapeHtml(supplierName)}</td></tr>
            <tr><td>Product / Item</td><td>${escapeHtml(productName || 'As per RFQ details')}</td></tr>
            <tr><td>Quantity</td><td>${escapeHtml(qty || 'To be confirmed')}</td></tr>
            <tr><td>Target Price</td><td>${escapeHtml(price ? `USD ${price}` : 'To be quoted')}</td></tr>
            <tr><td>Delivery Location</td><td>${escapeHtml(loc || 'To be confirmed')}</td></tr>
            <tr><td>Certification Required</td><td>${escapeHtml(cert || 'As applicable')}</td></tr>
            <tr><td>Required Lead Time</td><td>${escapeHtml(lead ? `${lead} days` : 'To be quoted')}</td></tr>
          </table>
          <h2>Formal RFQ Message</h2>
          <div class="body">${escapeHtml(draft.body)}</div>
          <h2>Authorisation and Signature</h2>
          <div class="signature">
            <div class="line">Authorised Signatory</div>
            <div class="line">Date</div>
          </div>
          <div class="footer">Generated by SourceUp RFQ Wizard | Confidential</div>
          <script>window.onload = () => { window.focus(); setTimeout(() => window.print(), 250); };</script>
        </body>
      </html>
    `);
    printable.document.close();
    return true;
  };

  const exportPdf = async ({ print = false } = {}) => {
    if (!draft || exporting) return;
    setPdfError('');
    setExporting(true);
    try {
      const res = await fetch(`${API}/quote/export-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPdfPayload()),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'PDF export failed');
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      if (print) {
        const pdfWindow = window.open(url, '_blank');
        if (pdfWindow) {
          setTimeout(() => {
            pdfWindow.focus();
            pdfWindow.print();
          }, 700);
        } else {
          throw new Error('Popup blocked. Allow popups for this site and try again.');
        }
      } else {
        const link = document.createElement('a');
        link.href = url;
        link.download = `SourceUp_RFQ_${Date.now()}.pdf`;
        document.body.appendChild(link);
        link.click();
        link.remove();
      }

      setTimeout(() => URL.revokeObjectURL(url), 30000);
    } catch (e) {
      const fallbackOpened = print && openPrintableFallback();
      setPdfError(
        fallbackOpened
          ? 'The PDF service was unavailable, so a print-ready RFQ page was opened instead. Choose "Save as PDF" in the print dialog.'
          : e.message || 'Unable to create PDF'
      );
    }
    setExporting(false);
  };

  const fi = { ...inputStyle, padding: '9px 12px', fontSize: 13 };

  return (
    <div style={modalOverlay}>
      <div style={{ ...modalBox, maxWidth: 660, padding: 0, overflow: 'hidden' }}>
        {/* Header */}
        <div style={{ padding: '20px 28px 16px', borderBottom: `1px solid ${C.border}`, background: C.surface }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
            <div>
              <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>📨 RFQ Wizard</h2>
              <div style={{ fontSize: 12, color: C.muted, marginTop: 2 }}>
                For <strong>{supplierName}</strong>{productName ? ` · ${productName}` : ''}
              </div>
            </div>
            <button style={{ border: 'none', background: 'none', fontSize: 20, color: C.muted, cursor: 'pointer', lineHeight: 1 }} onClick={onClose}>×</button>
          </div>

          {/* Step indicators */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {['Requirements', 'Preview & Edit', 'Refine'].map((label, i) => (
              <React.Fragment key={i}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <div style={{
                    width: 22, height: 22, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 11, fontWeight: 700, transition: 'all 0.3s',
                    background: step > i + 1 ? C.green : step === i + 1 ? C.blue : C.border,
                    color: step >= i + 1 ? '#fff' : C.muted,
                  }}>{step > i + 1 ? '✓' : i + 1}</div>
                  <span style={{ fontSize: 11, fontWeight: 600, color: step === i + 1 ? C.blue : C.muted }}>{label}</span>
                </div>
                {i < 2 && <div style={{ flex: 1, height: 2, background: step > i + 1 ? C.green : C.border, transition: 'background 0.3s' }} />}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Step 1 — Requirements */}
        {step === 1 && (
          <div style={{ padding: 28, animation: 'fadeIn 0.25s ease' }}>
            <div style={{ marginBottom: 20 }}>
              <label style={{ fontSize: 12, fontWeight: 700, color: C.muted, display: 'block', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 }}>Email Tone</label>
              <div style={{ display: 'flex', gap: 8 }}>
                {tones.map(t => (
                  <button key={t.id} onClick={() => setTone(t.id)} style={{
                    flex: 1, padding: '10px 8px', borderRadius: 10, cursor: 'pointer', transition: 'all 0.2s',
                    border: `2px solid ${tone === t.id ? C.blue : C.border}`,
                    background: tone === t.id ? C.blueLight : C.bg,
                    textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 16, marginBottom: 2 }}>{t.label}</div>
                    <div style={{ fontSize: 10, color: C.muted }}>{t.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
              {[
                ['📦 Quantity (units)', qty, setQty, 'number', 'e.g. 500'],
                ['💵 Target price (USD)', price, setPrice, 'number', 'e.g. 1.50'],
                ['📍 Delivery location', loc, setLoc, 'text', 'e.g. Mumbai, India'],
                ['🏅 Certification needed', cert, setCert, 'text', 'ISO, FDA, CE, BIS…'],
                ['⏱ Max lead time (days)', lead, setLead, 'number', 'e.g. 30'],
              ].map(([label, val, setter, type, ph]) => (
                <div key={label}>
                  <label style={{ fontSize: 11, fontWeight: 600, color: C.muted, display: 'block', marginBottom: 5 }}>{label}</label>
                  <input style={fi} type={type} placeholder={ph} value={val}
                    onChange={e => setter(e.target.value)}
                    onFocus={e => e.target.style.borderColor = C.blue}
                    onBlur={e => e.target.style.borderColor = C.border} />
                </div>
              ))}
            </div>

            <div style={{ marginBottom: 20 }}>
              <label style={{ fontSize: 11, fontWeight: 600, color: C.muted, display: 'block', marginBottom: 5 }}>📝 Additional requirements</label>
              <textarea style={{ ...fi, minHeight: 70, resize: 'vertical', width: '100%', lineHeight: 1.5 }}
                placeholder="Any special requirements, packaging specs, sample requests…"
                value={notes} onChange={e => setNotes(e.target.value)} />
            </div>

            <button style={{ ...btn('filled', 'lg'), width: '100%' }} onClick={generate} disabled={loading}>
              {loading ? <><Spinner /> Generating with AI…</> : '✨ Generate RFQ →'}
            </button>
          </div>
        )}

        {/* Step 2 — Preview */}
        {step === 2 && draft && (
          <div style={{ padding: 28, animation: 'fadeIn 0.25s ease' }}>
            <div style={{ marginBottom: 12 }}>
              <label style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: 0.5, display: 'block', marginBottom: 6 }}>Subject</label>
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: '10px 14px', fontSize: 14, fontWeight: 600 }}>
                {draft.subject}
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', padding: '3px 10px', borderRadius: 20, background: C.indigo + '15', color: C.indigo, border: `1px solid ${C.indigo}30`, fontSize: 11, fontWeight: 600 }}>Tone: {draft.tone || tone}</span>
              <span style={{ display: 'inline-flex', alignItems: 'center', padding: '3px 10px', borderRadius: 20, background: C.green + '15', color: C.green, border: `1px solid ${C.green}30`, fontSize: 11, fontWeight: 600 }}>AI Generated</span>
            </div>

            <div style={{ background: C.bg, border: `1.5px solid ${C.border}`, borderRadius: 10, padding: 20, marginBottom: 16, minHeight: 160, fontFamily: mono, fontSize: 13, lineHeight: 1.7, color: C.text, whiteSpace: 'pre-wrap', maxHeight: 280, overflowY: 'auto' }}>
              {draft.body}
            </div>

            <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
              <input style={{ ...fi, flex: 1 }}
                placeholder='✏️ Quick refine: "Make shorter" / "More formal" / "Add payment terms"'
                value={refine} onChange={e => setRefine(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !loading && refine.trim() && doRefine()}
                onFocus={e => e.target.style.borderColor = C.blue}
                onBlur={e => e.target.style.borderColor = C.border} />
              <button style={{ ...btn('outlined'), padding: '0 14px' }} onClick={doRefine} disabled={loading || !refine.trim()}>
                {loading ? <Spinner color={C.blue} size={14} /> : '⟳'}
              </button>
            </div>

            <div style={{ display: 'flex', gap: 10 }}>
              <button style={{ ...btn('ghost'), height: 38 }} onClick={() => setStep(1)}>← Edit</button>
              <button style={{ ...btn('success'), flex: 1, height: 38 }} onClick={copy}>
                {copied ? '✅ Copied!' : '📋 Copy Email'}
              </button>
              <button style={{ ...btn('filled', 'md'), height: 38 }} onClick={openEmail}>📧 Open in Mail</button>
            </div>

            <div style={{ display: 'flex', gap: 10, marginTop: 10 }}>
              <button style={{ ...btn('outlined'), flex: 1, height: 38 }} onClick={() => exportPdf()} disabled={exporting}>
                {exporting ? 'Creating PDF…' : '⬇ Download PDF'}
              </button>
              <button style={{ ...btn('outlined'), flex: 1, height: 38 }} onClick={() => exportPdf({ print: true })} disabled={exporting}>
                🖨 Print PDF
              </button>
            </div>

            {pdfError && (
              <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, padding: '10px 14px', fontSize: 12, color: C.red, marginTop: 10 }}>
                {pdfError}
              </div>
            )}

            {copied && (
              <div style={{ textAlign: 'center', marginTop: 10, fontSize: 12, color: C.green, animation: 'fadeIn 0.2s ease' }}>
                Email copied to clipboard! Paste it in your mail client.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
