/**
 * components/QuoteWizard.jsx — Enhanced RFQ Wizard with PDF Export & Digital Signature
 * Step 1: Fill in RFQ requirements
 * Step 2: AI-generated preview + refine
 * Step 3: Export options — Print PDF (blank sig) or Digital Signature PDF
 *
 * Usage: <QuoteWizard supplier={supplier} onClose={() => setShowQuote(false)} />
 */
import { useState } from 'react';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const BLUE  = '#1a56db';
const BLUE2 = '#eef3ff';
const GREY  = '#6b7280';

export default function QuoteWizard({ supplier, onClose }) {
  const [step, setStep] = useState(1);
  const [form, setForm] = useState({
    supplier_name:          supplier?.name || '',
    product_name:           supplier?.product || '',
    quantity:               '',
    target_price:           '',
    delivery_location:      '',
    required_certification: '',
    lead_time_days:         '',
    buyer_company:          '',
    buyer_name:             '',
    additional_notes:       '',
  });
  const [draft,    setDraft]    = useState(null);
  const [refineQ,  setRefineQ]  = useState('');
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState('');

  const [sigMode,   setSigMode]   = useState('print'); // "print" | "digital"
  const [sigName,   setSigName]   = useState('');
  const [sigTitle,  setSigTitle]  = useState('');
  const [sigEmail,  setSigEmail]  = useState('');
  const [sigPhone,  setSigPhone]  = useState('');
  const [exporting, setExporting] = useState(false);

  const handleGenerate = async () => {
    setError(''); setLoading(true);
    try {
      const body = {
        ...form,
        quantity:       form.quantity       ? parseInt(form.quantity)       : null,
        target_price:   form.target_price   ? parseFloat(form.target_price) : null,
        lead_time_days: form.lead_time_days ? parseInt(form.lead_time_days) : null,
      };
      const res  = await fetch(`${API}/quote/draft`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error((await res.json()).detail || 'Draft failed');
      setDraft(await res.json());
      setStep(2);
    } catch (e) { setError(e.message); }
    setLoading(false);
  };

  const handleRefine = async () => {
    if (!refineQ.trim() || !draft) return;
    setError(''); setLoading(true);
    try {
      const res  = await fetch(`${API}/quote/refine`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ original_draft: draft.body, refinement_instruction: refineQ }) });
      if (!res.ok) throw new Error((await res.json()).detail || 'Refine failed');
      setDraft(await res.json()); setRefineQ('');
    } catch (e) { setError(e.message); }
    setLoading(false);
  };

  const handleExportPDF = async () => {
    if (!draft) return;
    setExporting(true); setError('');
    try {
      const payload = {
        subject:           draft.subject,
        body:              draft.body,
        tone:              draft.tone,
        buyer_name:        sigName    || form.buyer_name    || 'Procurement Team',
        buyer_company:     form.buyer_company || 'Our Company',
        buyer_title:       sigTitle,
        buyer_email:       sigEmail,
        buyer_phone:       sigPhone,
        digital_signature: sigMode === 'digital' && sigName ? sigName : null,
        signature_date:    new Date().toISOString().split('T')[0],
      };
      const res = await fetch(`${API}/quote/export-pdf`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!res.ok) throw new Error(((await res.json().catch(() => ({ detail: 'Export failed' }))).detail));
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href = url; a.download = `RFQ_${Date.now()}.pdf`; a.click();
      URL.revokeObjectURL(url);
    } catch (e) { setError(`PDF export failed: ${e.message}`); }
    setExporting(false);
  };

  const field = (key, label, type = 'text', placeholder = '') => (
    <div style={{ marginBottom: 14 }}>
      <label style={ls.label}>{label}</label>
      <input style={ls.input} type={type} placeholder={placeholder} value={form[key]}
        onChange={(e) => setForm({ ...form, [key]: e.target.value })} />
    </div>
  );

  return (
    <div style={ls.overlay} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div style={ls.modal}>
        {/* Header */}
        <div style={ls.header}>
          <span style={ls.headerTitle}>📋 RFQ Wizard</span>
          <div style={ls.steps}>
            {['Requirements', 'Preview', 'Export'].map((s, i) => (
              <div key={i} style={{ ...ls.stepDot, background: step >= i + 1 ? BLUE : '#e5e7eb', color: step >= i + 1 ? '#fff' : '#6b7280', fontWeight: step === i + 1 ? 700 : 400 }}>
                {step > i + 1 ? '✓' : i + 1}
              </div>
            ))}
          </div>
          <button style={ls.closeBtn} onClick={onClose}>✕</button>
        </div>

        <div style={ls.body}>
          {error && <div style={ls.error}>⚠️ {error}</div>}

          {/* Step 1: Form */}
          {step === 1 && (
            <>
              <div style={ls.sectionTitle}>Procurement Details</div>
              <div style={ls.grid2}>
                {field('supplier_name', 'Supplier Name *', 'text', 'e.g. Tata Steel')}
                {field('product_name',  'Product / Item *', 'text', 'e.g. Cold-rolled steel sheet')}
              </div>
              <div style={ls.grid3}>
                {field('quantity',       'Quantity (units)',        'number', '500')}
                {field('target_price',   'Target Price (USD)',      'number', '12.50')}
                {field('lead_time_days', 'Max Lead Time (days)',    'number', '30')}
              </div>
              {field('delivery_location',      'Delivery Location',       'text', 'Chennai, India')}
              {field('required_certification', 'Required Certification',  'text', 'ISO 9001, BIS')}
              <div style={ls.grid2}>
                {field('buyer_company', 'Your Company', 'text', 'Acme Pvt Ltd')}
                {field('buyer_name',    'Your Name',    'text', 'Rajesh Kumar')}
              </div>
              <div style={{ marginBottom: 14 }}>
                <label style={ls.label}>Additional Notes</label>
                <textarea style={{ ...ls.input, height: 72, resize: 'vertical' }}
                  placeholder="Any special requirements, packaging instructions, etc."
                  value={form.additional_notes}
                  onChange={(e) => setForm({ ...form, additional_notes: e.target.value })} />
              </div>
              <button style={{ ...ls.btnPrimary, opacity: loading ? 0.7 : 1 }}
                onClick={handleGenerate} disabled={loading || !form.supplier_name || !form.product_name}>
                {loading ? 'Generating…' : '✨ Generate RFQ Draft'}
              </button>
            </>
          )}

          {/* Step 2: Preview */}
          {step === 2 && draft && (
            <>
              <div style={ls.sectionTitle}>AI-Generated Draft</div>
              <div style={ls.previewBox}>
                <div style={ls.previewSubject}>📧 {draft.subject}</div>
                <div style={ls.tone}>Tone: {draft.tone}</div>
                <pre style={ls.previewBody}>{draft.body}</pre>
              </div>
              <div style={ls.refineRow}>
                <input style={{ ...ls.input, flex: 1, marginBottom: 0 }}
                  placeholder="Ask AI to refine… e.g. 'Make it shorter' or 'Add urgency'"
                  value={refineQ} onChange={(e) => setRefineQ(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleRefine()} />
                <button style={ls.btnSecondary} onClick={handleRefine} disabled={loading}>{loading ? '…' : 'Refine'}</button>
              </div>
              <button style={{ ...ls.btnOutline, marginTop: 8 }}
                onClick={() => navigator.clipboard.writeText(`${draft.subject}\n\n${draft.body}`)}>
                📋 Copy to Clipboard
              </button>
              <div style={ls.navRow}>
                <button style={ls.btnGhost} onClick={() => setStep(1)}>← Back</button>
                <button style={ls.btnPrimary} onClick={() => setStep(3)}>Export PDF →</button>
              </div>
            </>
          )}

          {/* Step 3: Export */}
          {step === 3 && (
            <>
              <div style={ls.sectionTitle}>Export & Sign</div>
              <div style={ls.toggleRow}>
                <button style={{ ...ls.toggleBtn, ...(sigMode === 'print' ? ls.toggleActive : {}) }} onClick={() => setSigMode('print')}>🖨️ Print & Sign (wet ink)</button>
                <button style={{ ...ls.toggleBtn, ...(sigMode === 'digital' ? ls.toggleActive : {}) }} onClick={() => setSigMode('digital')}>✍️ Digital Signature</button>
              </div>
              {sigMode === 'print' && (
                <div style={ls.infoBox}><b>Print & Sign mode:</b> The PDF will include a blank signature block. Print it, sign with a pen, and send a scan to the supplier.</div>
              )}
              {sigMode === 'digital' && (
                <div style={ls.infoBox}><b>Digital Signature mode:</b> Your typed name is embedded in the PDF as an acknowledged digital signature with a timestamp. For legally binding e-signatures under IT Act 2000, use Aadhaar eSign or DocuSign separately.</div>
              )}
              <div style={{ marginBottom: 14 }}>
                <label style={ls.label}>{sigMode === 'digital' ? 'Full Name (appears as digital signature) *' : 'Authorised Signatory Name'}</label>
                <input style={ls.input} placeholder="Rajesh Kumar" value={sigName} onChange={(e) => setSigName(e.target.value)} />
              </div>
              <div style={ls.grid2}>
                <div>
                  <label style={ls.label}>Title / Designation</label>
                  <input style={ls.input} placeholder="Head of Procurement" value={sigTitle} onChange={(e) => setSigTitle(e.target.value)} />
                </div>
                <div>
                  <label style={ls.label}>Email</label>
                  <input style={ls.input} placeholder="rajesh@company.com" value={sigEmail} onChange={(e) => setSigEmail(e.target.value)} />
                </div>
              </div>
              <div style={{ marginBottom: 20 }}>
                <label style={ls.label}>Phone</label>
                <input style={ls.input} placeholder="+91 98765 43210" value={sigPhone} onChange={(e) => setSigPhone(e.target.value)} />
              </div>
              <button style={{ ...ls.btnPrimary, opacity: exporting ? 0.7 : 1 }}
                onClick={handleExportPDF} disabled={exporting || (sigMode === 'digital' && !sigName)}>
                {exporting ? 'Generating PDF…' : '⬇️ Download PDF'}
              </button>
              <div style={ls.navRow}>
                <button style={ls.btnGhost} onClick={() => setStep(2)}>← Back to Preview</button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

const ls = {
  overlay:        { position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999, padding: '16px' },
  modal:          { background: '#fff', borderRadius: 16, width: 640, maxWidth: '100%', maxHeight: '90vh', boxShadow: '0 24px 64px rgba(0,0,0,0.2)', display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  header:         { display: 'flex', alignItems: 'center', gap: 12, padding: '16px 24px', borderBottom: '1px solid #e5e7eb', background: '#f9fafb' },
  headerTitle:    { fontWeight: 700, fontSize: 16, color: '#111827', flex: 1 },
  steps:          { display: 'flex', gap: 8 },
  stepDot:        { width: 28, height: 28, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, transition: 'all .2s' },
  closeBtn:       { background: 'none', border: 'none', cursor: 'pointer', fontSize: 18, color: '#9ca3af', padding: '0 4px' },
  body:           { padding: '24px', overflowY: 'auto', flex: 1 },
  sectionTitle:   { fontSize: 15, fontWeight: 700, color: '#111827', marginBottom: 16 },
  label:          { display: 'block', fontSize: 12, fontWeight: 600, color: '#374151', marginBottom: 4 },
  input:          { width: '100%', padding: '9px 11px', border: '1.5px solid #e5e7eb', borderRadius: 8, fontSize: 13, boxSizing: 'border-box', outline: 'none' },
  grid2:          { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 },
  grid3:          { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 },
  btnPrimary:     { width: '100%', padding: '11px 0', background: BLUE, color: '#fff', border: 'none', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: 'pointer' },
  btnSecondary:   { padding: '9px 18px', background: BLUE, color: '#fff', border: 'none', borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap' },
  btnOutline:     { width: '100%', padding: '9px 0', background: 'none', color: BLUE, border: `1.5px solid ${BLUE}`, borderRadius: 8, fontSize: 13, cursor: 'pointer' },
  btnGhost:       { padding: '9px 18px', background: 'none', color: GREY, border: '1.5px solid #e5e7eb', borderRadius: 8, fontSize: 13, cursor: 'pointer' },
  previewBox:     { background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: 10, padding: '16px 20px', marginBottom: 12 },
  previewSubject: { fontWeight: 700, fontSize: 14, color: '#111827', marginBottom: 4 },
  tone:           { fontSize: 11, color: GREY, marginBottom: 10 },
  previewBody:    { fontFamily: 'inherit', fontSize: 13, color: '#374151', lineHeight: 1.6, whiteSpace: 'pre-wrap', margin: 0 },
  refineRow:      { display: 'flex', gap: 8, marginTop: 12, alignItems: 'flex-start' },
  navRow:         { display: 'flex', justifyContent: 'space-between', marginTop: 16 },
  error:          { background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, padding: '10px 14px', color: '#dc2626', fontSize: 13, marginBottom: 14 },
  toggleRow:      { display: 'flex', gap: 8, marginBottom: 16 },
  toggleBtn:      { flex: 1, padding: '10px 0', borderRadius: 8, border: '1.5px solid #e5e7eb', background: '#f9fafb', fontSize: 13, cursor: 'pointer', fontWeight: 500, color: GREY },
  toggleActive:   { background: BLUE2, borderColor: BLUE, color: BLUE, fontWeight: 700 },
  infoBox:        { background: BLUE2, border: '1px solid #c7d7ff', borderRadius: 8, padding: '12px 16px', fontSize: 12, color: '#374151', marginBottom: 16, lineHeight: 1.5 },
};
