import React, { useState, useEffect, useRef } from 'react';

const API = 'http://localhost:8000';

// ─── tiny helpers ────────────────────────────────────────────────────────────
const authHeader = () => {
  const t = localStorage.getItem('su_token');
  return t ? { Authorization: `Bearer ${t}` } : {};
};
const post = (url, body, extra = {}) =>
  fetch(`${API}${url}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeader(), ...extra },
    body: JSON.stringify(body),
  }).then(r => r.json());
const get = (url) =>
  fetch(`${API}${url}`, { headers: authHeader() }).then(r => r.json());

// ─── colour palette ───────────────────────────────────────────────────────────
const C = {
  primary: '#2563eb', primaryDark: '#1d4ed8', primaryLight: '#eff6ff',
  success: '#16a34a', warn: '#d97706', danger: '#dc2626',
  bg: '#f8fafc', card: '#ffffff', border: '#e2e8f0',
  text: '#0f172a', muted: '#64748b', highlight: '#fef9c3',
};
const s = {
  app:      { fontFamily: 'Inter,system-ui,sans-serif', background: C.bg, minHeight: '100vh', color: C.text },
  nav:      { background: C.card, borderBottom: `1px solid ${C.border}`, padding: '0 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 56, position: 'sticky', top: 0, zIndex: 100 },
  brand:    { fontWeight: 700, fontSize: 18, color: C.primary, letterSpacing: '-0.5px' },
  badge:    (color) => ({ background: color + '22', color, border: `1px solid ${color}44`, borderRadius: 20, padding: '2px 10px', fontSize: 11, fontWeight: 600 }),
  btn:      (variant = 'primary') => ({
    background:   variant === 'primary' ? C.primary : variant === 'outline' ? 'transparent' : C.card,
    color:        variant === 'primary' ? '#fff' : C.primary,
    border:       `1.5px solid ${variant === 'ghost' ? 'transparent' : C.primary}`,
    borderRadius: 8, padding: '8px 18px', fontWeight: 600, fontSize: 13,
    cursor: 'pointer', transition: 'all .15s',
  }),
  input:  { border: `1.5px solid ${C.border}`, borderRadius: 8, padding: '9px 13px', fontSize: 14, width: '100%', boxSizing: 'border-box', outline: 'none' },
  card:   { background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, marginBottom: 14, boxShadow: '0 1px 4px #0001' },
  modal:  { position: 'fixed', inset: 0, background: '#0007', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 200, padding: 20 },
  mbox:   { background: C.card, borderRadius: 16, padding: 32, width: '100%', maxWidth: 520, maxHeight: '90vh', overflowY: 'auto', boxShadow: '0 20px 60px #0003' },
};


// ════════════════════════════════════════════════════════════════════════════
// 1. ONBOARDING WIZARD (shown on first visit)
// ════════════════════════════════════════════════════════════════════════════
function OnboardingWizard({ onComplete }) {
  const [step, setStep] = useState(0);
  const steps = [
    {
      title: '👋 Welcome to SourceUp',
      body:  'Find the right suppliers in seconds — with AI that understands your real business constraints.',
      icon:  '🔍',
    },
    {
      title: '💬 Chat or Search',
      body:  'Type naturally in the Chat tab ("Find biodegradable containers from India under ₹2") or use the Search tab with advanced filters.',
      icon:  '🤖',
    },
    {
      title: '📋 Get a Quote Draft',
      body:  'After finding a supplier, click "Draft RFQ" to get a professional email ready to send — in one click.',
      icon:  '📨',
    },
    {
      title: '🔬 Understand Every Decision',
      body:  'Enable "Show Explanations" to see exactly why each supplier was ranked where it was.',
      icon:  '📊',
    },
  ];
  const current = steps[step];
  return (
    <div style={s.modal}>
      <div style={{ ...s.mbox, textAlign: 'center' }}>
        <div style={{ fontSize: 56, marginBottom: 12 }}>{current.icon}</div>
        <h2 style={{ margin: '0 0 10px', fontSize: 20 }}>{current.title}</h2>
        <p style={{ color: C.muted, lineHeight: 1.6, margin: '0 0 28px' }}>{current.body}</p>
        {/* progress dots */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: 6, marginBottom: 24 }}>
          {steps.map((_, i) => (
            <div key={i} style={{ width: 8, height: 8, borderRadius: '50%', background: i === step ? C.primary : C.border }} />
          ))}
        </div>
        <div style={{ display: 'flex', gap: 10, justifyContent: 'center' }}>
          {step > 0 && (
            <button style={s.btn('outline')} onClick={() => setStep(step - 1)}>Back</button>
          )}
          {step < steps.length - 1 ? (
            <button style={s.btn()} onClick={() => setStep(step + 1)}>Next →</button>
          ) : (
            <button style={s.btn()} onClick={onComplete}>Get Started 🚀</button>
          )}
        </div>
        <button style={{ marginTop: 16, background: 'none', border: 'none', color: C.muted, cursor: 'pointer', fontSize: 13 }} onClick={onComplete}>
          Skip tour
        </button>
      </div>
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// 2. AUTH MODAL (register / login)
// ════════════════════════════════════════════════════════════════════════════
function AuthModal({ onClose, onSuccess }) {
  const [tab, setTab]         = useState('login');   // login | register
  const [email, setEmail]     = useState('');
  const [password, setPass]   = useState('');
  const [company, setCompany] = useState('');
  const [err, setErr]         = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setErr(''); setLoading(true);
    try {
      const url  = tab === 'login' ? '/auth/login' : '/auth/register';
      const body = tab === 'login' ? { email, password } : { email, password, company };
      const data = await post(url, body);
      if (data.access_token) {
        localStorage.setItem('su_token', data.access_token);
        localStorage.setItem('su_email', data.email);
        localStorage.setItem('su_plan',  data.plan);
        onSuccess(data);
      } else {
        setErr(data.detail || 'Something went wrong');
      }
    } catch (e) {
      setErr('Network error — is the backend running?');
    }
    setLoading(false);
  };

  return (
    <div style={s.modal}>
      <div style={s.mbox}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <h2 style={{ margin: 0 }}>{tab === 'login' ? 'Sign In' : 'Create Account'}</h2>
          <button style={{ border: 'none', background: 'none', fontSize: 22, cursor: 'pointer' }} onClick={onClose}>×</button>
        </div>
        {/* tab switcher */}
        <div style={{ display: 'flex', borderBottom: `1px solid ${C.border}`, marginBottom: 20 }}>
          {['login', 'register'].map(t => (
            <button key={t} style={{ ...s.btn('ghost'), borderRadius: 0, borderBottom: `2px solid ${tab === t ? C.primary : 'transparent'}`, color: tab === t ? C.primary : C.muted, padding: '8px 16px' }}
              onClick={() => setTab(t)}>
              {t === 'login' ? 'Sign In' : 'Register'}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <input style={s.input} placeholder="Email" type="email" value={email} onChange={e => setEmail(e.target.value)} />
          <input style={s.input} placeholder="Password" type="password" value={password} onChange={e => setPass(e.target.value)} />
          {tab === 'register' && (
            <input style={s.input} placeholder="Company name (optional)" value={company} onChange={e => setCompany(e.target.value)} />
          )}
          {err && <div style={{ color: C.danger, fontSize: 13 }}>{err}</div>}
          <button style={s.btn()} onClick={submit} disabled={loading}>
            {loading ? '...' : tab === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </div>
      </div>
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// 3. BILLING MODAL (plan upgrade via Razorpay)
// ════════════════════════════════════════════════════════════════════════════
function BillingModal({ onClose }) {
  const [plans, setPlans]   = useState([]);
  const [msg, setMsg]       = useState('');

  useEffect(() => {
    get('/auth/billing/plans').then(setPlans).catch(() => {});
  }, []);

  const upgrade = async (planId) => {
    setMsg('Creating order...');
    try {
      const order = await post('/auth/billing/order', { plan: planId });
      if (!order.order_id) { setMsg(order.detail || 'Order failed'); return; }

      // Check if Razorpay.js is loaded
      if (!window.Razorpay) {
        setMsg('⚠️  Add <script src="https://checkout.razorpay.com/v1/checkout.js"> to your index.html');
        return;
      }
      const rzp = new window.Razorpay({
        key:      order.razorpay_key_id,
        amount:   order.amount,
        currency: order.currency,
        name:     'SourceUp',
        order_id: order.order_id,
        handler: async (resp) => {
          const verify = await post('/auth/billing/verify', { ...resp, plan: planId });
          if (verify.success) {
            localStorage.setItem('su_plan',  planId);
            localStorage.setItem('su_token', verify.new_token);
            setMsg(`✅ Upgraded to ${planId}! Refresh to apply.`);
          } else {
            setMsg('❌ Payment verification failed');
          }
        },
        prefill: { email: localStorage.getItem('su_email') || '' },
      });
      rzp.open();
    } catch (e) {
      setMsg('Error: ' + e.message);
    }
  };

  return (
    <div style={s.modal}>
      <div style={s.mbox}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <h2 style={{ margin: 0 }}>Upgrade Plan</h2>
          <button style={{ border: 'none', background: 'none', fontSize: 22, cursor: 'pointer' }} onClick={onClose}>×</button>
        </div>
        {msg && <div style={{ padding: 10, borderRadius: 8, background: C.primaryLight, marginBottom: 16, fontSize: 13 }}>{msg}</div>}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {plans.map(p => (
            <div key={p.id} style={{ ...s.card, margin: 0, border: `1.5px solid ${p.id === 'pro' ? C.primary : C.border}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 15 }}>{p.name}</div>
                  <div style={{ color: C.muted, fontSize: 13 }}>₹{p.price_inr}/month</div>
                </div>
                {p.id !== 'free' && (
                  <button style={s.btn()} onClick={() => upgrade(p.id)}>Upgrade</button>
                )}
                {p.id === 'free' && <span style={s.badge(C.success)}>Current</span>}
              </div>
              <ul style={{ margin: '10px 0 0 16px', padding: 0, color: C.muted, fontSize: 13, lineHeight: 1.8 }}>
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
// 4. QUOTE MODAL — LLM-powered RFQ drafting
// ════════════════════════════════════════════════════════════════════════════
function QuoteModal({ supplier, onClose }) {
  const [qty,      setQty]      = useState('');
  const [price,    setPrice]    = useState('');
  const [location, setLoc]      = useState('');
  const [cert,     setCert]     = useState('');
  const [lead,     setLead]     = useState('');
  const [notes,    setNotes]    = useState('');
  const [draft,    setDraft]    = useState(null);
  const [refine,   setRefine]   = useState('');
  const [loading,  setLoading]  = useState(false);
  const [copied,   setCopied]   = useState(false);

  const generate = async () => {
    setLoading(true);
    const data = await post('/quote/draft', {
      supplier_name:          supplier.supplier || 'Supplier',
      product_name:           supplier.product  || 'Product',
      quantity:               qty   ? parseInt(qty)     : null,
      target_price:           price ? parseFloat(price) : null,
      delivery_location:      location || null,
      required_certification: cert     || null,
      lead_time_days:         lead     ? parseInt(lead) : null,
      buyer_company:          localStorage.getItem('su_company') || 'Our Company',
      buyer_name:             localStorage.getItem('su_email')   || 'Procurement Team',
      additional_notes:       notes || null,
    });
    setDraft(data);
    setLoading(false);
  };

  const doRefine = async () => {
    if (!draft || !refine.trim()) return;
    setLoading(true);
    const data = await post('/quote/refine', {
      original_draft:          draft.subject + '\n\n' + draft.body,
      refinement_instruction:  refine,
    });
    setDraft(data);
    setRefine('');
    setLoading(false);
  };

  const copy = () => {
    navigator.clipboard.writeText(`Subject: ${draft.subject}\n\n${draft.body}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div style={s.modal}>
      <div style={{ ...s.mbox, maxWidth: 620 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ margin: 0 }}>📨 Draft RFQ — {supplier.supplier}</h2>
          <button style={{ border: 'none', background: 'none', fontSize: 22, cursor: 'pointer' }} onClick={onClose}>×</button>
        </div>

        {!draft ? (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 14 }}>
              <div><label style={{ fontSize: 12, color: C.muted }}>Quantity (units)</label>
                <input style={s.input} placeholder="e.g. 500" value={qty} onChange={e => setQty(e.target.value)} /></div>
              <div><label style={{ fontSize: 12, color: C.muted }}>Target unit price (USD)</label>
                <input style={s.input} placeholder="e.g. 1.50" value={price} onChange={e => setPrice(e.target.value)} /></div>
              <div><label style={{ fontSize: 12, color: C.muted }}>Delivery location</label>
                <input style={s.input} placeholder="e.g. Mumbai" value={location} onChange={e => setLoc(e.target.value)} /></div>
              <div><label style={{ fontSize: 12, color: C.muted }}>Certification required</label>
                <input style={s.input} placeholder="e.g. ISO, FDA" value={cert} onChange={e => setCert(e.target.value)} /></div>
              <div><label style={{ fontSize: 12, color: C.muted }}>Max lead time (days)</label>
                <input style={s.input} placeholder="e.g. 30" value={lead} onChange={e => setLead(e.target.value)} /></div>
            </div>
            <div style={{ marginBottom: 14 }}>
              <label style={{ fontSize: 12, color: C.muted }}>Additional notes</label>
              <textarea style={{ ...s.input, minHeight: 60, resize: 'vertical' }} placeholder="Any special requirements…" value={notes} onChange={e => setNotes(e.target.value)} />
            </div>
            <button style={{ ...s.btn(), width: '100%' }} onClick={generate} disabled={loading}>
              {loading ? 'Generating…' : '✨ Generate RFQ with AI'}
            </button>
          </>
        ) : (
          <>
            <div style={{ background: C.primaryLight, border: `1px solid ${C.primary}33`, borderRadius: 8, padding: 16, marginBottom: 14, fontSize: 13 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Subject: {draft.subject}</div>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontFamily: 'inherit', lineHeight: 1.6 }}>{draft.body}</pre>
            </div>
            <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
              <button style={{ ...s.btn(), flex: 1 }} onClick={copy}>{copied ? '✅ Copied!' : '📋 Copy to clipboard'}</button>
              <button style={s.btn('outline')} onClick={() => setDraft(null)}>Edit inputs</button>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <input style={{ ...s.input, flex: 1 }} placeholder='Refine: "Make it shorter" / "Add ISO requirement"' value={refine} onChange={e => setRefine(e.target.value)} />
              <button style={s.btn()} onClick={doRefine} disabled={loading || !refine.trim()}>
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
// 5. SUPPLIER CARD
// ════════════════════════════════════════════════════════════════════════════
function SupplierCard({ supplier, rank }) {
  const [expanded,  setExpanded]  = useState(false);
  const [showQuote, setShowQuote] = useState(false);
  const plan = localStorage.getItem('su_plan') || 'free';

  return (
    <>
      <div style={s.card}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ fontWeight: 700, fontSize: 14, color: C.muted }}>#{rank}</span>
              <span style={{ fontWeight: 700, fontSize: 15 }}>{supplier.supplier || 'Supplier'}</span>
              {supplier.location && <span style={s.badge(C.muted)}>📍 {supplier.location}</span>}
            </div>
            <div style={{ color: C.muted, fontSize: 13, marginBottom: 6 }}>{supplier.product}</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {supplier.price    && <span style={s.badge(C.success)}>💰 {supplier.price}</span>}
              {supplier.moq      && <span style={s.badge(C.warn)}>MOQ: {supplier.moq}</span>}
              {supplier.lead_time && <span style={s.badge(C.primary)}>⏱ {supplier.lead_time}</span>}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
            <div style={{ fontWeight: 800, fontSize: 18, color: C.primary }}>
              {(supplier.score * 100).toFixed(0)}<span style={{ fontSize: 11, fontWeight: 400 }}>/100</span>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {supplier.url && (
                <a href={supplier.url} target="_blank" rel="noreferrer" style={{ ...s.btn('outline'), textDecoration: 'none', fontSize: 12, padding: '5px 10px' }}>
                  Visit →
                </a>
              )}
              <button style={{ ...s.btn(), fontSize: 12, padding: '5px 10px' }}
                onClick={() => {
                  if (plan === 'free') { alert('Quote drafting requires SourceUp Pro. Click the Upgrade button.'); return; }
                  setShowQuote(true);
                }}>
                📨 Draft RFQ
              </button>
              {supplier.decision_trace && (
                <button style={{ ...s.btn('outline'), fontSize: 12, padding: '5px 10px' }}
                  onClick={() => setExpanded(!expanded)}>
                  {expanded ? 'Hide' : 'Explain'} ▾
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Reasons */}
        {supplier.reasons?.length > 0 && (
          <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {supplier.reasons.map((r, i) => (
              <span key={i} style={{ background: C.highlight, border: `1px solid ${C.warn}44`, borderRadius: 6, padding: '2px 8px', fontSize: 11 }}>
                {r}
              </span>
            ))}
          </div>
        )}

        {/* Decision trace */}
        {expanded && supplier.decision_trace && (
          <div style={{ marginTop: 14, background: '#f8fafc', borderRadius: 8, padding: 14, fontSize: 12 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>📊 Decision Trace</div>
            {supplier.decision_trace.summary?.map((line, i) => (
              <div key={i} style={{ marginBottom: 4, paddingLeft: 12, borderLeft: `3px solid ${C.primary}` }}>
                {line}
              </div>
            ))}
            {supplier.decision_trace.contributions && (
              <div style={{ marginTop: 10 }}>
                {Object.entries(supplier.decision_trace.contributions).slice(0, 5).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <div style={{ width: 90, color: C.muted }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ flex: 1, height: 6, background: C.border, borderRadius: 3 }}>
                      <div style={{ width: `${Math.min(100, (v.contribution || 0) * 100)}%`, height: '100%', background: C.primary, borderRadius: 3 }} />
                    </div>
                    <div style={{ width: 36, textAlign: 'right', fontWeight: 600 }}>
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
// 6. SEARCH TAB
// ════════════════════════════════════════════════════════════════════════════
function SearchTab() {
  const [query,    setQuery]    = useState('');
  const [results,  setResults]  = useState([]);
  const [meta,     setMeta]     = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [searched, setSearched] = useState(false);
  // Advanced filters
  const [adv,       setAdv]     = useState(false);
  const [maxPrice,  setMaxP]    = useState('');
  const [moqBudget, setMoqB]    = useState('');
  const [location,  setLoc]     = useState('');
  const [locMand,   setLocM]    = useState(false);
  const [cert,      setCert]    = useState('');
  const [leadTime,  setLead]    = useState('');
  const [minYears,  setYears]   = useState('');
  const [explain,   setExplain] = useState(true);
  const [whatIf,    setWhatIf]  = useState(false);

  const search = async (e) => {
    e?.preventDefault();
    if (!query.trim()) return;
    setLoading(true); setSearched(true);
    const data = await post('/recommend', {
      product:              query,
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
    setResults(Array.isArray(data) ? data : (data.results || []));
    setMeta(Array.isArray(data) ? null : data.metadata);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '24px 16px' }}>
      {/* Search box */}
      <form onSubmit={search} style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
        <input
          style={{ ...s.input, fontSize: 15 }}
          placeholder="Search suppliers: plastic containers, cotton fabric, solar panels…"
          value={query} onChange={e => setQuery(e.target.value)}
        />
        <button style={{ ...s.btn(), whiteSpace: 'nowrap', padding: '9px 22px' }} type="submit" disabled={loading}>
          {loading ? '…' : 'Search'}
        </button>
      </form>

      {/* Advanced toggle */}
      <button style={{ ...s.btn('ghost'), marginBottom: 14, fontSize: 12, color: C.muted }}
        onClick={() => setAdv(!adv)}>
        {adv ? '▲ Hide' : '▼ Show'} Advanced Filters
      </button>

      {adv && (
        <div style={{ ...s.card, marginBottom: 18 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {[
              ['Max unit price (USD)', maxPrice, setMaxP, 'number', 'e.g. 2.50'],
              ['MOQ budget (USD)',     moqBudget, setMoqB,'number','e.g. 500'],
              ['Location',            location,  setLoc, 'text',  'e.g. India'],
              ['Certification',       cert,      setCert,'text',  'ISO, FDA, CE…'],
              ['Max lead time (days)',leadTime,  setLead,'number','e.g. 30'],
              ['Min. platform years', minYears,  setYears,'number','e.g. 3'],
            ].map(([label, val, setter, type, ph]) => (
              <div key={label}>
                <label style={{ fontSize: 11, color: C.muted, display: 'block', marginBottom: 4 }}>{label}</label>
                <input style={s.input} type={type} placeholder={ph} value={val} onChange={e => setter(e.target.value)} />
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 20, marginTop: 12 }}>
            <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
              <input type="checkbox" checked={locMand} onChange={e => setLocM(e.target.checked)} />
              Location mandatory
            </label>
            <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
              <input type="checkbox" checked={explain} onChange={e => setExplain(e.target.checked)} />
              Show explanations
            </label>
            <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
              <input type="checkbox" checked={whatIf} onChange={e => setWhatIf(e.target.checked)} />
              What-if scenarios
            </label>
          </div>
        </div>
      )}

      {/* Metadata */}
      {meta && (
        <div style={{ fontSize: 12, color: C.muted, marginBottom: 14 }}>
          {meta.total_candidates} found → {meta.after_constraints} passed constraints → top {results.length} shown
          {meta.ranking_method && ` · Ranked by ${meta.ranking_method}`}
        </div>
      )}

      {/* Results */}
      {searched && !loading && results.length === 0 && (
        <div style={{ textAlign: 'center', padding: 40, color: C.muted }}>
          No suppliers matched. Try relaxing your filters.
        </div>
      )}
      {results.map((r, i) => <SupplierCard key={i} supplier={r} rank={i + 1} />)}
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// 7. CHAT TAB
// ════════════════════════════════════════════════════════════════════════════
function ChatTab() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: "Hi! I'm SourceBot. Tell me what you're sourcing — e.g. 'Find biodegradable food containers from India under $2 with FDA certification'." }
  ]);
  const [input,    setInput]    = useState('');
  const [loading,  setLoading]  = useState(false);
  const [sid]                   = useState(`user_${Date.now()}`);
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const send = async () => {
    if (!input.trim() || loading) return;
    const text = input.trim();
    setInput('');
    setMessages(m => [...m, { role: 'user', text }]);
    setLoading(true);
    const data = await post('/chat', { session_id: sid, message: text });
    const reply = data.message || 'No response';
    setMessages(m => [...m, { role: 'assistant', text: reply, suppliers: data.suppliers || [] }]);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '24px 16px', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)' }}>
      <div style={{ flex: 1, overflowY: 'auto', marginBottom: 12 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: 12 }}>
            <div style={{
              maxWidth: '80%', borderRadius: m.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
              background: m.role === 'user' ? C.primary : C.card,
              color: m.role === 'user' ? '#fff' : C.text,
              border: m.role === 'user' ? 'none' : `1px solid ${C.border}`,
              padding: '10px 14px', fontSize: 14, lineHeight: 1.55,
            }}>
              {m.text}
              {m.suppliers?.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  {m.suppliers.slice(0, 3).map((s, j) => (
                    <div key={j} style={{ background: C.bg, borderRadius: 8, padding: '8px 12px', marginBottom: 6, fontSize: 12 }}>
                      <div style={{ fontWeight: 700 }}>#{j+1} {s.supplier}</div>
                      <div style={{ color: C.muted }}>{s.product} · {s.price} · {s.location}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}>
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: '16px 16px 16px 4px', padding: '10px 14px', fontSize: 14, color: C.muted }}>
              SourceBot is thinking…
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Input */}
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          style={{ ...s.input, flex: 1 }}
          placeholder="Ask SourceBot anything…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
        />
        <button style={s.btn()} onClick={send} disabled={loading || !input.trim()}>Send</button>
      </div>
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
// 8. ROOT APP
// ════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [tab,          setTab]     = useState('search');
  const [showOnboard,  setOnboard] = useState(!localStorage.getItem('su_onboarded'));
  const [showAuth,     setAuth]    = useState(false);
  const [showBilling,  setBilling] = useState(false);
  const [user,         setUser]    = useState(
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

  return (
    <div style={s.app}>
      {/* Onboarding wizard */}
      {showOnboard && (
        <OnboardingWizard onComplete={() => {
          localStorage.setItem('su_onboarded', '1');
          setOnboard(false);
        }} />
      )}

      {/* Auth modal */}
      {showAuth && <AuthModal onClose={() => setAuth(false)} onSuccess={handleAuthSuccess} />}

      {/* Billing modal */}
      {showBilling && <BillingModal onClose={() => setBilling(false)} />}

      {/* Navigation */}
      <nav style={s.nav}>
        <div style={s.brand}>SourceUp ✦</div>

        {/* Tab switcher */}
        <div style={{ display: 'flex', gap: 4 }}>
          {['search', 'chat'].map(t => (
            <button key={t} style={{
              ...s.btn('ghost'),
              color:      tab === t ? C.primary : C.muted,
              borderBottom: `2px solid ${tab === t ? C.primary : 'transparent'}`,
              borderRadius: 0, padding: '4px 14px', textTransform: 'capitalize',
            }} onClick={() => setTab(t)}>
              {t === 'search' ? '🔍 Search' : '💬 Chat'}
            </button>
          ))}
        </div>

        {/* Auth / billing */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {user ? (
            <>
              <span style={s.badge(user.plan === 'pro' ? C.primary : C.muted)}>
                {user.plan === 'pro' ? '⭐ Pro' : 'Free'}
              </span>
              {user.plan === 'free' && (
                <button style={{ ...s.btn(), fontSize: 12, padding: '5px 12px' }} onClick={() => setBilling(true)}>
                  Upgrade
                </button>
              )}
              <span style={{ fontSize: 13, color: C.muted }}>{user.email}</span>
              <button style={{ ...s.btn('outline'), fontSize: 12, padding: '5px 10px' }} onClick={logout}>Sign out</button>
            </>
          ) : (
            <button style={s.btn()} onClick={() => setAuth(true)}>Sign in</button>
          )}
        </div>
      </nav>

      {/* Tab content */}
      {tab === 'search' ? <SearchTab /> : <ChatTab />}
    </div>
  );
}