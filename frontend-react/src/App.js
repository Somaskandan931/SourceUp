import React, { useState, useEffect, useRef, useCallback } from 'react';

const API = 'http://localhost:8000';

// ─── helpers ──────────────────────────────────────────────────────────────────
const authHeader = () => {
  const t = localStorage.getItem('su_token');
  return t ? { Authorization: `Bearer ${t}` } : {};
};
const post = (url, body) =>
  fetch(`${API}${url}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeader() },
    body: JSON.stringify(body),
  }).then(r => r.json());
const get = (url) =>
  fetch(`${API}${url}`, { headers: authHeader() }).then(r => r.json());

// Fix unknown supplier names — mirrors the backend get_supplier_name logic
const resolveSupplierName = (s) => {
  if (!s) return null;
  const raw =
    s.supplier ||
    s['supplier name'] || s['Supplier Name'] ||
    s['company name']  || s['Company Name'] ||
    s.company || s.name || s.brand || s.manufacturer;
  if (!raw || /^unknown/i.test(raw.trim())) return null;
  return raw.trim();
};

const resolveProductName = (s, fallback = '') => {
  if (!s) return fallback;
  return (
    s.product ||
    s['product name'] || s['Product Name'] ||
    s['Product'] ||
    s.item || s.description || fallback
  ).trim();
};

// ─── design tokens ────────────────────────────────────────────────────────────
const C = {
  blue:        '#1a73e8',
  blueDark:    '#1557b0',
  blueLight:   '#e8f0fe',
  green:       '#188038',
  greenLight:  '#e6f4ea',
  amber:       '#b06000',
  amberLight:  '#fef7e0',
  red:         '#c5221f',
  bg:          '#ffffff',
  surface:     '#f8f9fa',
  border:      '#dadce0',
  text:        '#202124',
  muted:       '#5f6368',
  chip:        '#f1f3f4',
};

const font = "'Google Sans', 'Segoe UI', sans-serif";

const css = {
  app:    { fontFamily: font, background: C.bg, minHeight: '100vh', color: C.text },
  nav:    {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '8px 24px', borderBottom: `1px solid ${C.border}`,
    position: 'sticky', top: 0, background: C.bg, zIndex: 100,
    height: 64,
  },
  brand:  { fontWeight: 700, fontSize: 22, letterSpacing: '-0.5px', cursor: 'pointer',
            color: C.text, userSelect: 'none' },
  xBlue:  { color: C.blue },
  btn:    (v = 'filled') => ({
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6,
    padding: v === 'filled' ? '0 24px' : '0 20px', height: 36,
    background: v === 'filled' ? C.blue : 'transparent',
    color:      v === 'filled' ? '#fff' : C.blue,
    border:     v === 'outlined' ? `1px solid ${C.border}` : 'none',
    borderRadius: 4, fontWeight: 500, fontSize: 14, cursor: 'pointer',
    fontFamily: font, transition: 'background .15s, box-shadow .15s',
    whiteSpace: 'nowrap',
  }),
  pill:   (color = C.blue) => ({
    display: 'inline-flex', alignItems: 'center', gap: 4,
    padding: '2px 10px', borderRadius: 12,
    background: color + '18', color, border: `1px solid ${color}30`,
    fontSize: 11, fontWeight: 600, whiteSpace: 'nowrap',
  }),
  input:  {
    border: `1px solid ${C.border}`, borderRadius: 24, padding: '12px 20px',
    fontSize: 16, outline: 'none', width: '100%', boxSizing: 'border-box',
    fontFamily: font, background: C.bg,
    boxShadow: '0 1px 3px #0001',
    transition: 'box-shadow .2s',
  },
  card:   {
    background: C.bg, border: `1px solid ${C.border}`, borderRadius: 8,
    padding: '16px 20px', marginBottom: 8,
    transition: 'box-shadow .15s',
  },
  modal:  { position: 'fixed', inset: 0, background: '#0006', display: 'flex',
            alignItems: 'center', justifyContent: 'center', zIndex: 300, padding: 16 },
  mbox:   { background: C.bg, borderRadius: 8, padding: 32, width: '100%',
            maxWidth: 520, maxHeight: '92vh', overflowY: 'auto',
            boxShadow: '0 8px 40px #0003' },
  chip:   { display: 'inline-flex', alignItems: 'center', padding: '6px 16px',
            borderRadius: 20, border: `1px solid ${C.border}`, fontSize: 13,
            cursor: 'pointer', background: C.bg, fontFamily: font,
            transition: 'background .12s', color: C.text },
};


// ════════════════════════════════════════════════════════════════════════════
// ONBOARDING WIZARD
// ════════════════════════════════════════════════════════════════════════════
function OnboardingWizard({ onComplete }) {
  const [step, setStep] = useState(0);
  const steps = [
    { icon: '🔍', title: 'Welcome to SourceUp-X', body: 'Find verified suppliers in seconds — with AI that understands your real business constraints.' },
    { icon: '💬', title: 'Chat or Search', body: 'Type naturally ("Find biodegradable containers from India under ₹2 with FDA certification") or use advanced filters.' },
    { icon: '📨', title: 'Get a Quote Draft', body: 'After finding a supplier, click "Draft RFQ" to get a professional email ready to send — in one click.' },
    { icon: '📊', title: 'Transparent Decisions', body: 'See exactly why each supplier was ranked where it was, with full decision trace breakdowns.' },
  ];
  const cur = steps[step];
  return (
    <div style={css.modal}>
      <div style={{ ...css.mbox, textAlign: 'center' }}>
        <div style={{ fontSize: 52, marginBottom: 12 }}>{cur.icon}</div>
        <h2 style={{ margin: '0 0 8px', fontWeight: 600, fontSize: 20 }}>{cur.title}</h2>
        <p style={{ color: C.muted, lineHeight: 1.65, margin: '0 0 28px', fontSize: 15 }}>{cur.body}</p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: 6, marginBottom: 24 }}>
          {steps.map((_, i) => (
            <div key={i} style={{ width: 8, height: 8, borderRadius: '50%', background: i === step ? C.blue : C.border, transition: 'background .2s' }} />
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
          {step > 0 && <button style={css.btn('outlined')} onClick={() => setStep(s => s - 1)}>Back</button>}
          {step < steps.length - 1
            ? <button style={css.btn()} onClick={() => setStep(s => s + 1)}>Next →</button>
            : <button style={css.btn()} onClick={onComplete}>Get Started 🚀</button>
          }
        </div>
        <button style={{ marginTop: 16, background: 'none', border: 'none', color: C.muted, cursor: 'pointer', fontSize: 13 }} onClick={onComplete}>Skip tour</button>
      </div>
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// AUTH MODAL
// ════════════════════════════════════════════════════════════════════════════
function AuthModal({ onClose, onSuccess }) {
  const [tab, setTab]         = useState('login');
  const [email, setEmail]     = useState('');
  const [password, setPass]   = useState('');
  const [company, setCompany] = useState('');
  const [err, setErr]         = useState('');
  const [loading, setLoad]    = useState(false);

  const submit = async () => {
    if (!email || !password) { setErr('Please fill all fields'); return; }
    setErr(''); setLoad(true);
    try {
      const url  = tab === 'login' ? '/auth/login' : '/auth/register';
      const body = tab === 'login' ? { email, password } : { email, password, company };
      const data = await post(url, body);
      if (data.access_token) {
        localStorage.setItem('su_token', data.access_token);
        localStorage.setItem('su_email', data.email);
        localStorage.setItem('su_plan',  data.plan);
        onSuccess(data);
      } else { setErr(data.detail || 'Something went wrong'); }
    } catch { setErr('Network error — is the backend running?'); }
    setLoad(false);
  };

  const inputStyle = { ...css.input, borderRadius: 4, padding: '10px 14px', fontSize: 14 };

  return (
    <div style={css.modal}>
      <div style={css.mbox}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <h2 style={{ margin: 0, fontWeight: 600 }}>{tab === 'login' ? 'Sign in' : 'Create account'}</h2>
          <button style={{ border: 'none', background: 'none', fontSize: 24, cursor: 'pointer', color: C.muted }} onClick={onClose}>×</button>
        </div>
        <div style={{ display: 'flex', borderBottom: `1px solid ${C.border}`, marginBottom: 20 }}>
          {['login', 'register'].map(t => (
            <button key={t} style={{ background: 'none', border: 'none', borderBottom: `2px solid ${tab === t ? C.blue : 'transparent'}`, color: tab === t ? C.blue : C.muted, padding: '8px 16px', cursor: 'pointer', fontFamily: font, fontWeight: 500, fontSize: 14 }} onClick={() => setTab(t)}>
              {t === 'login' ? 'Sign in' : 'Register'}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <input style={inputStyle} placeholder="Email" type="email" value={email} onChange={e => setEmail(e.target.value)} onKeyDown={e => e.key === 'Enter' && submit()} />
          <input style={inputStyle} placeholder="Password" type="password" value={password} onChange={e => setPass(e.target.value)} onKeyDown={e => e.key === 'Enter' && submit()} />
          {tab === 'register' && <input style={inputStyle} placeholder="Company name (optional)" value={company} onChange={e => setCompany(e.target.value)} />}
          {err && <div style={{ color: C.red, fontSize: 13 }}>{err}</div>}
          <button style={{ ...css.btn(), height: 40, marginTop: 4 }} onClick={submit} disabled={loading}>
            {loading ? 'Please wait…' : tab === 'login' ? 'Sign in' : 'Create account'}
          </button>
        </div>
      </div>
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// BILLING MODAL
// ════════════════════════════════════════════════════════════════════════════
function BillingModal({ onClose }) {
  const [plans, setPlans]   = useState([]);
  const [msg, setMsg]       = useState('');
  const [orderId, setOrderId] = useState('');
  const [utr, setUtr]       = useState('');
  const [pendingPlan, setPending] = useState('');

  useEffect(() => { get('/auth/billing/plans').then(setPlans).catch(() => {}); }, []);

  const createOrder = async (planId) => {
    setMsg('Creating order…');
    try {
      const order = await post('/auth/billing/order', { plan: planId });
      if (!order.order_id) { setMsg(order.detail || 'Order failed'); return; }
      setPending(planId);
      setOrderId(order.order_id);
      setMsg(`Pay ₹${order.amount} to UPI: ${order.upi_id}\nThen enter the UTR transaction ID below.`);
    } catch (e) { setMsg('Error: ' + e.message); }
  };

  const verifyPayment = async () => {
    if (!utr.trim()) { setMsg('Please enter the UPI transaction ID'); return; }
    setMsg('Verifying…');
    try {
      const res = await post('/auth/billing/verify', { order_id: orderId, upi_transaction_id: utr, plan: pendingPlan });
      if (res.success) {
        localStorage.setItem('su_plan', pendingPlan);
        localStorage.setItem('su_token', res.new_token);
        setMsg(`✅ Upgraded to ${pendingPlan}! Refresh the page to apply.`);
      } else { setMsg('❌ Verification failed. Please try again.'); }
    } catch (e) { setMsg('Error: ' + e.message); }
  };

  return (
    <div style={css.modal}>
      <div style={css.mbox}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ margin: 0, fontWeight: 600 }}>Upgrade Plan</h2>
          <button style={{ border: 'none', background: 'none', fontSize: 24, cursor: 'pointer', color: C.muted }} onClick={onClose}>×</button>
        </div>
        {msg && (
          <div style={{ padding: 12, borderRadius: 6, background: C.blueLight, marginBottom: 16, fontSize: 13, whiteSpace: 'pre-line', color: C.text }}>
            {msg}
            {orderId && (
              <div style={{ marginTop: 10, display: 'flex', gap: 6 }}>
                <input style={{ ...css.input, borderRadius: 4, padding: '8px 12px', fontSize: 13, flex: 1 }} placeholder="UTR / Transaction ID" value={utr} onChange={e => setUtr(e.target.value)} />
                <button style={{ ...css.btn(), height: 36, fontSize: 13 }} onClick={verifyPayment}>Verify</button>
              </div>
            )}
          </div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {plans.map(p => (
            <div key={p.id} style={{ ...css.card, border: `1.5px solid ${p.id === 'pro' ? C.blue : C.border}`, marginBottom: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 15 }}>{p.name}</div>
                  <div style={{ color: C.muted, fontSize: 13 }}>₹{p.price_inr}/month</div>
                </div>
                {p.id !== 'free'
                  ? <button style={css.btn()} onClick={() => createOrder(p.id)}>Upgrade</button>
                  : <span style={css.pill(C.green)}>Current</span>
                }
              </div>
              <ul style={{ margin: '8px 0 0 16px', padding: 0, color: C.muted, fontSize: 13, lineHeight: 1.9 }}>
                {p.features.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// QUOTE MODAL
// ════════════════════════════════════════════════════════════════════════════
function QuoteModal({ supplier, onClose }) {
  const [qty,     setQty]     = useState('');
  const [price,   setPrice]   = useState('');
  const [loc,     setLoc]     = useState('');
  const [cert,    setCert]    = useState('');
  const [lead,    setLead]    = useState('');
  const [notes,   setNotes]   = useState('');
  const [draft,   setDraft]   = useState(null);
  const [refine,  setRefine]  = useState('');
  const [loading, setLoad]    = useState(false);
  const [copied,  setCopied]  = useState(false);

  const supplierName = resolveSupplierName(supplier) || 'this supplier';
  const productName  = resolveProductName(supplier, '');

  const generate = async () => {
    setLoad(true);
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
      additional_notes:       notes || null,
    });
    setDraft(data); setLoad(false);
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
    setCopied(true); setTimeout(() => setCopied(false), 1800);
  };

  const fi = { ...css.input, borderRadius: 4, padding: '9px 12px', fontSize: 13 };

  return (
    <div style={css.modal}>
      <div style={{ ...css.mbox, maxWidth: 620 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
          <h2 style={{ margin: 0, fontWeight: 600, fontSize: 18 }}>Draft RFQ — {supplierName}</h2>
          <button style={{ border: 'none', background: 'none', fontSize: 24, cursor: 'pointer', color: C.muted }} onClick={onClose}>×</button>
        </div>
        {!draft ? (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 12 }}>
              {[
                ['Quantity (units)', qty, setQty, 'number', 'e.g. 500'],
                ['Target unit price (USD)', price, setPrice, 'number', 'e.g. 1.50'],
                ['Delivery location', loc, setLoc, 'text', 'e.g. Mumbai'],
                ['Certification required', cert, setCert, 'text', 'ISO, FDA, CE…'],
                ['Max lead time (days)', lead, setLead, 'number', 'e.g. 30'],
              ].map(([label, val, setter, type, ph]) => (
                <div key={label}>
                  <label style={{ fontSize: 11, color: C.muted, display: 'block', marginBottom: 4 }}>{label}</label>
                  <input style={fi} type={type} placeholder={ph} value={val} onChange={e => setter(e.target.value)} />
                </div>
              ))}
            </div>
            <div style={{ marginBottom: 14 }}>
              <label style={{ fontSize: 11, color: C.muted, display: 'block', marginBottom: 4 }}>Additional notes</label>
              <textarea style={{ ...fi, minHeight: 60, resize: 'vertical', width: '100%', boxSizing: 'border-box' }} placeholder="Any special requirements…" value={notes} onChange={e => setNotes(e.target.value)} />
            </div>
            <button style={{ ...css.btn(), width: '100%', height: 42 }} onClick={generate} disabled={loading}>
              {loading ? 'Generating…' : '✨ Generate RFQ with AI'}
            </button>
          </>
        ) : (
          <>
            <div style={{ background: C.blueLight, border: `1px solid ${C.blue}33`, borderRadius: 6, padding: 16, marginBottom: 12, fontSize: 13 }}>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>Subject: {draft.subject}</div>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontFamily: font, lineHeight: 1.65 }}>{draft.body}</pre>
            </div>
            <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
              <button style={{ ...css.btn(), flex: 1, height: 38 }} onClick={copy}>{copied ? '✅ Copied!' : '📋 Copy'}</button>
              <button style={{ ...css.btn('outlined'), height: 38 }} onClick={() => setDraft(null)}>Edit</button>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <input style={{ ...fi, flex: 1 }} placeholder='Refine: "Make it shorter" / "Add ISO requirement"' value={refine} onChange={e => setRefine(e.target.value)} />
              <button style={{ ...css.btn(), height: 38, padding: '0 16px' }} onClick={doRefine} disabled={loading || !refine.trim()}>
                {loading ? '…' : 'Refine'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// SUPPLIER CARD — with robust name resolution
// ════════════════════════════════════════════════════════════════════════════
function SupplierCard({ supplier, rank }) {
  const [expanded,  setExpanded]  = useState(false);
  const [showQuote, setShowQuote] = useState(false);
  const plan = localStorage.getItem('su_plan') || 'free';

  // Resolve name — never shows "Unknown Supplier"
  const name    = resolveSupplierName(supplier) || `Supplier #${rank}`;
  const product = resolveProductName(supplier, '—');

  const initials = name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
  const avatarBg = ['#4285f4','#34a853','#fbbc04','#ea4335','#9c27b0','#00acc1'][rank % 6];

  const scoreVal = Math.round((supplier.score || 0) * 100);

  return (
    <>
      <div
        style={{ ...css.card, marginBottom: 8 }}
        onMouseEnter={e => e.currentTarget.style.boxShadow = '0 1px 8px #0002'}
        onMouseLeave={e => e.currentTarget.style.boxShadow = 'none'}
      >
        {/* Row 1: avatar + name + badges */}
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
          {/* Avatar */}
          <div style={{ width: 40, height: 40, borderRadius: '50%', background: avatarBg, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 14, flexShrink: 0 }}>
            {initials}
          </div>

          {/* Info */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 2 }}>
              <span style={{ fontWeight: 600, fontSize: 15 }}>{name}</span>
              {supplier.location && <span style={css.pill(C.muted)}>📍 {supplier.location}</span>}
              {supplier.moq      && <span style={css.pill(C.amber)}>MOQ {supplier.moq}</span>}
            </div>
            <div style={{ color: C.muted, fontSize: 13, marginBottom: 8, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{product}</div>

            {/* Tags row */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
              {supplier.price && supplier.price !== 'Contact for pricing' && (
                <span style={css.pill(C.green)}>💰 {supplier.price}</span>
              )}
              {supplier.lead_time && <span style={css.pill(C.blue)}>⏱ {supplier.lead_time}</span>}
              {supplier.reasons?.slice(0, 3).map((r, i) => (
                <span key={i} style={{ background: '#fef9c3', color: '#854d0e', border: '1px solid #fde047', borderRadius: 4, padding: '1px 8px', fontSize: 11 }}>{r}</span>
              ))}
            </div>
          </div>

          {/* Score + actions */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6, flexShrink: 0 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontWeight: 700, fontSize: 20, color: scoreVal >= 70 ? C.green : scoreVal >= 40 ? C.amber : C.muted }}>
                {scoreVal}
              </div>
              <div style={{ fontSize: 10, color: C.muted }}>/ 100</div>
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              {supplier.url && (
                <a href={supplier.url} target="_blank" rel="noreferrer"
                   style={{ ...css.btn('outlined'), textDecoration: 'none', height: 32, fontSize: 12, padding: '0 12px' }}>
                  Visit ↗
                </a>
              )}
              <button style={{ ...css.btn(), height: 32, fontSize: 12, padding: '0 12px' }}
                onClick={() => {
                  if (plan === 'free') {
                    alert('RFQ drafting requires SourceUp Pro. Click the Demo button or Upgrade in the nav.');
                    return;
                  }
                  setShowQuote(true);
                }}>
                📨 RFQ
              </button>
              {supplier.decision_trace && (
                <button style={{ ...css.btn('outlined'), height: 32, fontSize: 12, padding: '0 10px' }}
                  onClick={() => setExpanded(e => !e)}>
                  {expanded ? 'Hide ▴' : 'Explain ▾'}
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Decision trace */}
        {expanded && supplier.decision_trace && (
          <div style={{ marginTop: 14, background: C.surface, borderRadius: 6, padding: 14, fontSize: 12 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, color: C.text }}>📊 Decision Trace</div>
            {supplier.decision_trace.summary?.map((line, i) => (
              <div key={i} style={{ marginBottom: 4, paddingLeft: 10, borderLeft: `3px solid ${C.blue}`, color: C.muted }}>{line}</div>
            ))}
            {supplier.decision_trace.contributions && (
              <div style={{ marginTop: 10 }}>
                {Object.entries(supplier.decision_trace.contributions).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
                    <div style={{ width: 110, color: C.muted, fontSize: 11 }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ flex: 1, height: 5, background: C.border, borderRadius: 3 }}>
                      <div style={{ width: `${Math.min(100, (v.contribution || 0) * 200)}%`, height: '100%', background: C.blue, borderRadius: 3, transition: 'width .4s' }} />
                    </div>
                    <div style={{ width: 32, textAlign: 'right', fontWeight: 600, fontSize: 11 }}>
                      {((v.contribution || 0) * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      {showQuote && <QuoteModal supplier={supplier} onClose={() => setShowQuote(false)} />}
    </>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// HOMEPAGE (Google-style landing)
// ════════════════════════════════════════════════════════════════════════════
function HomePage({ onSearch }) {
  const [query, setQuery]   = useState('');
  const [focused, setFocus] = useState(false);
  const inputRef = useRef(null);

  const suggestions = [
    'LED bulbs in Vietnam',
    'ISO certified textiles',
    'Electronics in China',
    'Biodegradable packaging India',
    'Solar panels Taiwan',
  ];

  const doSearch = (q) => {
    const term = (q || query).trim();
    if (term) onSearch(term);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 'calc(100vh - 64px)', padding: '0 16px 80px' }}>
      {/* Logo */}
      <div style={{ fontFamily: "'Google Sans', sans-serif", fontSize: 72, fontWeight: 700, letterSpacing: -2, marginBottom: 28, userSelect: 'none' }}>
        <span style={{ color: '#4285f4' }}>S</span>
        <span style={{ color: '#ea4335' }}>o</span>
        <span style={{ color: '#fbbc04' }}>u</span>
        <span style={{ color: '#4285f4' }}>r</span>
        <span style={{ color: '#34a853' }}>c</span>
        <span style={{ color: '#ea4335' }}>e</span>
        <span style={{ color: C.text }}>UP</span>
        <span style={{ color: '#4285f4', fontSize: 52, verticalAlign: 'super' }}>-X</span>
      </div>
      <p style={{ color: C.muted, fontSize: 16, marginBottom: 28, fontWeight: 400 }}>Explainable Procurement Intelligence for SMEs</p>

      {/* Search bar */}
      <div style={{ width: '100%', maxWidth: 584, marginBottom: 28 }}>
        <div style={{
          display: 'flex', alignItems: 'center',
          border: `1px solid ${focused ? 'transparent' : C.border}`,
          borderRadius: 24, padding: '0 16px',
          boxShadow: focused ? '0 1px 6px #20212447' : '0 1px 4px #0001',
          background: C.bg, transition: 'box-shadow .2s',
        }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0, color: C.muted }}>
            <circle cx="11" cy="11" r="7" stroke={C.muted} strokeWidth="2"/>
            <path d="M20 20l-3-3" stroke={C.muted} strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <input
            ref={inputRef}
            style={{ flex: 1, border: 'none', outline: 'none', fontSize: 16, padding: '14px 12px', fontFamily: font, background: 'transparent', color: C.text }}
            placeholder="Search suppliers…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onFocus={() => setFocus(true)}
            onBlur={() => setFocus(false)}
            onKeyDown={e => e.key === 'Enter' && doSearch()}
          />
          {query && (
            <button style={{ border: 'none', background: 'none', cursor: 'pointer', color: C.muted, fontSize: 18, padding: '0 4px' }} onClick={() => { setQuery(''); inputRef.current?.focus(); }}>×</button>
          )}
        </div>
      </div>

      {/* Buttons */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 32 }}>
        <button style={{ ...css.btn(), height: 40, background: C.surface, color: C.text, border: `1px solid ${C.border}` }} onClick={() => doSearch()}>
          🔍 Search Suppliers
        </button>
        <button style={{ ...css.btn(), height: 40, background: C.surface, color: C.text, border: `1px solid ${C.border}` }} onClick={() => doSearch(suggestions[Math.floor(Math.random() * suggestions.length)])}>
          I'm Feeling Lucky
        </button>
      </div>

      {/* Chips */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
        <span style={{ color: C.muted, fontSize: 13, alignSelf: 'center' }}>Try searching for:</span>
        {suggestions.map(s => (
          <button key={s} style={css.chip} onClick={() => doSearch(s)}
            onMouseEnter={e => e.currentTarget.style.background = C.chip}
            onMouseLeave={e => e.currentTarget.style.background = C.bg}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// SEARCH RESULTS VIEW
// ════════════════════════════════════════════════════════════════════════════
function SearchResultsView({ initialQuery, onBack }) {
  const [query,    setQuery]    = useState(initialQuery);
  const [results,  setResults]  = useState([]);
  const [meta,     setMeta]     = useState(null);
  const [loading,  setLoad]     = useState(false);
  const [searched, setSearched] = useState(false);
  const [adv,      setAdv]      = useState(false);

  // Filters
  const [maxPrice,  setMaxP]   = useState('');
  const [moqBudget, setMoqB]   = useState('');
  const [location,  setLoc]    = useState('');
  const [locMand,   setLocM]   = useState(false);
  const [cert,      setCert]   = useState('');
  const [leadTime,  setLead]   = useState('');
  const [minYears,  setYears]  = useState('');
  const [explain,   setXpl]    = useState(true);
  const [whatIf,    setWI]     = useState(false);

  const doSearch = useCallback(async (q) => {
    const term = (q || query).trim();
    if (!term) return;
    setLoad(true); setSearched(true);
    const data = await post('/recommend', {
      product:              term,
      max_price:            maxPrice  ? parseFloat(maxPrice)  : null,
      moq_budget:           moqBudget ? parseFloat(moqBudget) : null,
      location:             location  || null,
      location_mandatory:   locMand,
      certification:        cert      || null,
      max_lead_time:        leadTime  ? parseInt(leadTime)    : null,
      min_years_experience: minYears  ? parseInt(minYears)   : null,
      enable_explanations:  explain,
      enable_what_if:       whatIf,
    });
    // Filter out suppliers that resolved to nothing
    const raw = Array.isArray(data) ? data : (data.results || []);
    const cleaned = raw.filter(s => resolveSupplierName(s) || resolveProductName(s, ''));
    setResults(cleaned);
    setMeta(Array.isArray(data) ? null : data.metadata);
    setLoad(false);
  }, [query, maxPrice, moqBudget, location, locMand, cert, leadTime, minYears, explain, whatIf]);

  useEffect(() => { doSearch(initialQuery); }, []); // eslint-disable-line

  const fi = { ...css.input, borderRadius: 4, padding: '8px 12px', fontSize: 13 };

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: '16px 16px 40px' }}>
      {/* Compact search bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
        <button style={{ background: 'none', border: 'none', cursor: 'pointer', color: C.muted, fontSize: 22, padding: '0 4px' }} onClick={onBack} title="Back to home">←</button>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', border: `1px solid ${C.border}`, borderRadius: 24, padding: '0 14px', boxShadow: '0 1px 4px #0001' }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
            <circle cx="11" cy="11" r="7" stroke={C.muted} strokeWidth="2"/>
            <path d="M20 20l-3-3" stroke={C.muted} strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <input
            style={{ flex: 1, border: 'none', outline: 'none', fontSize: 15, padding: '10px 10px', fontFamily: font, background: 'transparent' }}
            value={query} onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && doSearch()}
          />
        </div>
        <button style={{ ...css.btn(), height: 38, padding: '0 20px' }} onClick={() => doSearch()}>Search</button>
      </div>

      {/* Advanced toggle */}
      <button style={{ ...css.btn('outlined'), height: 30, fontSize: 12, padding: '0 14px', marginBottom: adv ? 0 : 16, color: C.muted, border: 'none' }} onClick={() => setAdv(a => !a)}>
        {adv ? '▲ Hide filters' : '▼ Advanced filters'}
      </button>

      {adv && (
        <div style={{ ...css.card, marginBottom: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
            {[
              ['Max unit price (USD)', maxPrice, setMaxP, 'number', 'e.g. 2.50'],
              ['MOQ budget (USD)',     moqBudget, setMoqB, 'number', 'e.g. 500'],
              ['Location',            location,  setLoc,  'text',   'e.g. India'],
              ['Certification',       cert,      setCert, 'text',   'ISO, FDA, CE…'],
              ['Max lead time (days)',leadTime,  setLead, 'number', 'e.g. 30'],
              ['Min platform years',  minYears,  setYears,'number', 'e.g. 3'],
            ].map(([label, val, setter, type, ph]) => (
              <div key={label}>
                <label style={{ fontSize: 11, color: C.muted, display: 'block', marginBottom: 3 }}>{label}</label>
                <input style={fi} type={type} placeholder={ph} value={val} onChange={e => setter(e.target.value)} />
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 20, marginTop: 12, flexWrap: 'wrap' }}>
            {[
              ['Location mandatory', locMand, setLocM],
              ['Show explanations',  explain, setXpl],
              ['What-if scenarios',  whatIf,  setWI],
            ].map(([label, val, setter]) => (
              <label key={label} style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', userSelect: 'none' }}>
                <input type="checkbox" checked={val} onChange={e => setter(e.target.checked)} />
                {label}
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Metadata */}
      {meta && !loading && (
        <div style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>
          About {meta.total_candidates?.toLocaleString()} suppliers found
          {meta.after_constraints != null && ` — ${meta.after_constraints} matched constraints`}
          {meta.latency_ms && ` · ${meta.latency_ms}ms`}
          {meta.ranking_method && ` · Ranked by ${meta.ranking_method}`}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[1,2,3].map(i => (
            <div key={i} style={{ ...css.card, height: 90, background: C.surface, animation: 'pulse 1.4s ease-in-out infinite', opacity: 0.7 }} />
          ))}
        </div>
      )}

      {/* Results */}
      {!loading && searched && results.length === 0 && (
        <div style={{ textAlign: 'center', padding: 60, color: C.muted }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>🔍</div>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>No suppliers matched</div>
          <div style={{ fontSize: 13 }}>Try relaxing your filters or using a different search term.</div>
        </div>
      )}

      {!loading && results.map((r, i) => <SupplierCard key={i} supplier={r} rank={i + 1} />)}
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// CHAT TAB
// ════════════════════════════════════════════════════════════════════════════
function ChatTab() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: "Hi! I'm SourceBot 🤖 Tell me what you're sourcing — e.g. \"Find biodegradable food containers from India under $2 with FDA certification\"." }
  ]);
  const [input,   setInput]   = useState('');
  const [loading, setLoad]    = useState(false);
  const [sid]                 = useState(`user_${Date.now()}`);
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    setMessages(m => [...m, { role: 'user', text }]);
    setLoad(true);
    try {
      const data = await post('/chat', { session_id: sid, message: text });
      const suppliers = (data.suppliers || []).filter(s => resolveSupplierName(s) || resolveProductName(s, ''));
      setMessages(m => [...m, { role: 'bot', text: data.message || 'No response', suppliers }]);
    } catch {
      setMessages(m => [...m, { role: 'bot', text: 'Error connecting to backend. Please check the server is running.' }]);
    }
    setLoad(false);
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '16px 16px 0', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 80px)' }}>
      <div style={{ flex: 1, overflowY: 'auto', paddingBottom: 8 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: 12 }}>
            {m.role === 'bot' && (
              <div style={{ width: 32, height: 32, borderRadius: '50%', background: C.blue, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, flexShrink: 0, marginRight: 8, alignSelf: 'flex-end' }}>S</div>
            )}
            <div style={{
              maxWidth: '78%',
              background: m.role === 'user' ? C.blue : C.surface,
              color: m.role === 'user' ? '#fff' : C.text,
              border: m.role === 'user' ? 'none' : `1px solid ${C.border}`,
              borderRadius: m.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
              padding: '10px 14px', fontSize: 14, lineHeight: 1.6,
            }}>
              {m.text}
              {m.suppliers?.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  {m.suppliers.slice(0, 3).map((s, j) => {
                    const name    = resolveSupplierName(s) || `Supplier #${j+1}`;
                    const product = resolveProductName(s, '');
                    return (
                      <div key={j} style={{ background: C.bg, borderRadius: 8, padding: '8px 12px', marginBottom: 6, fontSize: 12, border: `1px solid ${C.border}` }}>
                        <div style={{ fontWeight: 600 }}>#{j+1} {name}</div>
                        <div style={{ color: C.muted, marginTop: 2 }}>{product}{s.price ? ` · ${s.price}` : ''}{s.location ? ` · ${s.location}` : ''}</div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}>
            <div style={{ width: 32, height: 32, borderRadius: '50%', background: C.blue, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, flexShrink: 0, marginRight: 8 }}>S</div>
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: '18px 18px 18px 4px', padding: '12px 16px', fontSize: 14, color: C.muted }}>
              <span style={{ animation: 'blink 1s step-end infinite' }}>●</span>&nbsp;
              <span style={{ animationDelay: '.2s', animation: 'blink 1s step-end infinite' }}>●</span>&nbsp;
              <span style={{ animationDelay: '.4s', animation: 'blink 1s step-end infinite' }}>●</span>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>
      <div style={{ display: 'flex', gap: 8, padding: '12px 0 16px', borderTop: `1px solid ${C.border}`, background: C.bg }}>
        <input
          style={{ ...css.input, flex: 1, fontSize: 14, padding: '10px 16px' }}
          placeholder="Ask SourceBot…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
        />
        <button style={{ ...css.btn(), height: 42, padding: '0 20px' }} onClick={send} disabled={loading || !input.trim()}>Send</button>
      </div>
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// ROOT APP
// ════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [view,       setView]    = useState('home');  // home | search | chat
  const [searchQ,    setSearchQ] = useState('');
  const [showOnboard,setOnboard] = useState(!localStorage.getItem('su_onboarded'));
  const [showAuth,   setAuth]    = useState(false);
  const [showBilling,setBilling] = useState(false);
  const [user,       setUser]    = useState(
    localStorage.getItem('su_email')
      ? { email: localStorage.getItem('su_email'), plan: localStorage.getItem('su_plan') }
      : null
  );

  const handleAuthSuccess = (data) => {
    setUser({ email: data.email, plan: data.plan });
    setAuth(false);
  };

  const logout = () => {
    ['su_token','su_email','su_plan'].forEach(k => localStorage.removeItem(k));
    setUser(null);
  };

  const startSearch = (q) => {
    setSearchQ(q);
    setView('search');
  };

  // Demo login function
  const handleDemoLogin = () => {
    // Create a demo user session (Pro plan)
    const demoToken = 'demo_token_' + Date.now();
    localStorage.setItem('su_token', demoToken);
    localStorage.setItem('su_email', 'demo@sourceup.com');
    localStorage.setItem('su_plan', 'pro');
    setUser({ email: 'demo@sourceup.com', plan: 'pro' });
  };

  return (
    <div style={css.app}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
        *, *::before, *::after { box-sizing: border-box; }
        @keyframes pulse { 0%,100%{opacity:.7} 50%{opacity:.4} }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
        body { margin: 0; }
        button { font-family: inherit; }
      `}</style>

      {showOnboard && (
        <OnboardingWizard onComplete={() => {
          localStorage.setItem('su_onboarded', '1');
          setOnboard(false);
        }} />
      )}
      {showAuth    && <AuthModal onClose={() => setAuth(false)} onSuccess={handleAuthSuccess} />}
      {showBilling && <BillingModal onClose={() => setBilling(false)} />}

      {/* Navigation */}
      <nav style={css.nav}>
        <div style={css.brand} onClick={() => setView('home')}>
          SourceUP<span style={css.xBlue}>-X</span>
        </div>

        {/* Tab switcher */}
        {view !== 'home' && (
          <div style={{ display: 'flex', gap: 2 }}>
            {[['search','🔍 Search'], ['chat','💬 Chat']].map(([t, label]) => (
              <button key={t} style={{
                background: 'none', border: 'none',
                borderBottom: `2px solid ${view === t ? C.blue : 'transparent'}`,
                color: view === t ? C.blue : C.muted,
                padding: '4px 16px', cursor: 'pointer', fontFamily: font, fontWeight: 500,
                fontSize: 14, height: 48,
              }} onClick={() => setView(t)}>
                {label}
              </button>
            ))}
          </div>
        )}

        {view === 'home' && (
          <div style={{ display: 'flex', gap: 2 }}>
            <button style={{ background: 'none', border: 'none', color: C.muted, fontFamily: font, fontSize: 14, padding: '4px 16px', cursor: 'pointer', height: 48 }} onClick={() => setView('chat')}>💬 Chat</button>
          </div>
        )}

        {/* Auth area WITH DEMO BUTTON */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {user ? (
            <>
              {user.plan !== 'free' && <span style={css.pill(C.blue)}>⭐ Pro</span>}
              {user.plan === 'free' && (
                <button style={{ ...css.btn('outlined'), height: 34, fontSize: 13 }} onClick={() => setBilling(true)}>Upgrade</button>
              )}
              <span style={{ fontSize: 13, color: C.muted, maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{user.email}</span>
              <button style={{ ...css.btn('outlined'), height: 34, fontSize: 13 }} onClick={logout}>Sign out</button>
            </>
          ) : (
            <div style={{ display: 'flex', gap: 8 }}>
              <button style={{ ...css.btn(), height: 36 }} onClick={() => setAuth(true)}>
                Sign in
              </button>
              {/* DEMO BUTTON - One-click Pro access */}
              <button
                style={{ ...css.btn('outlined'), height: 36, background: '#e8f0fe', borderColor: C.blue }}
                onClick={handleDemoLogin}
              >
                🎬 Demo (Pro)
              </button>
            </div>
          )}
        </div>
      </nav>

      {/* Views */}
      {view === 'home'   && <HomePage onSearch={startSearch} />}
      {view === 'search' && <SearchResultsView key={searchQ} initialQuery={searchQ} onBack={() => setView('home')} />}
      {view === 'chat'   && <ChatTab />}
    </div>
  );
}