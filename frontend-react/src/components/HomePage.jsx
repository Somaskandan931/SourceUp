// components/HomePage.jsx
import { useState, useRef } from 'react';
import { C, btn, card, font } from '../styles/tokens';

// ─── Services Section ─────────────────────────────────────────────────────────
export function ServicesSection({ onSearch, onOpenChat, onOpenBilling, onOpenAuth, user }) {
  const services = [
    {
      icon: '🔍', title: 'Smart Search', color: C.blue,
      desc: 'FAISS-powered semantic search across thousands of verified global suppliers with constraint filtering.',
      action: () => onSearch('eco-friendly packaging India'), cta: 'Try Search',
    },
    {
      icon: '💬', title: 'AI Chat (SourceBot)', color: C.indigo,
      desc: 'Conversational procurement assistant powered by Groq LLM — describe what you need in plain language.',
      action: onOpenChat, cta: 'Open Chat',
    },
    {
      icon: '📨', title: 'RFQ Wizard', color: C.green,
      desc: 'AI-powered multi-step RFQ email builder with tone selection, instant preview, and one-click refinement.',
      action: user ? () => onSearch('solar panels Taiwan') : onOpenAuth,
      cta: user ? 'Search & Draft RFQ' : 'Sign In to Use',
    },
    {
      icon: '📊', title: 'Decision Traces', color: C.amber,
      desc: 'Full transparency into why each supplier was ranked — see score contributions for price, lead time, and more.',
      action: () => onSearch('ISO certified textiles'), cta: 'See Traces',
    },
    {
      icon: '💳', title: 'Flexible Plans', color: '#7c3aed',
      desc: 'Free tier for exploration. Pro at ₹999/month unlocks unlimited RFQs and AI features via UPI payment.',
      action: onOpenBilling, cta: 'View Plans',
    },
    {
      icon: '🔬', title: 'What-If Scenarios', color: '#0891b2',
      desc: 'Simulate how changing price targets, lead times, or certifications affects supplier rankings and scores.',
      action: () => onSearch('electronics components China'), cta: 'Try It',
    },
  ];

  return (
    <div style={{ maxWidth: 1000, margin: '0 auto', padding: '48px 24px 64px' }}>
      <div style={{ textAlign: 'center', marginBottom: 40 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.blue, textTransform: 'uppercase', letterSpacing: 1.5, marginBottom: 10 }}>Everything You Need</div>
        <h2 style={{ fontFamily: "'DM Serif Display', serif", fontSize: 32, fontWeight: 400, margin: '0 0 12px' }}>Procurement Intelligence Suite</h2>
        <p style={{ color: C.muted, fontSize: 15, maxWidth: 500, margin: '0 auto' }}>One platform for AI-powered supplier discovery, smart RFQ drafting, and transparent decision-making.</p>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        {services.map((s, i) => (
          <div key={i} style={{ ...card, padding: '24px', cursor: 'pointer', borderTop: `3px solid ${s.color}`, transition: 'all 0.25s', animationDelay: `${i * 0.05}s` }}
            onClick={s.action}
            onMouseEnter={e => { e.currentTarget.style.boxShadow = `0 8px 30px ${s.color}20`; e.currentTarget.style.transform = 'translateY(-3px)'; }}
            onMouseLeave={e => { e.currentTarget.style.boxShadow = 'none'; e.currentTarget.style.transform = 'none'; }}>
            <div style={{ fontSize: 28, marginBottom: 12 }}>{s.icon}</div>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 6 }}>{s.title}</div>
            <div style={{ color: C.muted, fontSize: 13, lineHeight: 1.6, marginBottom: 16 }}>{s.desc}</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: s.color, display: 'flex', alignItems: 'center', gap: 4 }}>{s.cta} →</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Home Page ────────────────────────────────────────────────────────────────
export default function HomePage({ onSearch, onOpenChat, onOpenBilling, onOpenAuth, user }) {
  const [query, setQuery]   = useState('');
  const [focused, setFocus] = useState(false);
  const inputRef = useRef(null);

  const suggestions = ['LED bulbs Vietnam', 'ISO certified textiles', 'Electronics China', 'Biodegradable packaging India', 'Solar panels Taiwan'];
  const doSearch = (q) => { const term = (q || query).trim(); if (term) onSearch(term); };

  return (
    <>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 'calc(100vh - 64px)', padding: '0 16px 40px', background: `radial-gradient(ellipse 70% 50% at 50% -10%, ${C.blueLight}, transparent)` }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.blue, textTransform: 'uppercase', letterSpacing: 2, marginBottom: 16, opacity: 0.8 }}>Explainable Procurement Intelligence</div>

        {/* Logo */}
        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 68, fontWeight: 400, letterSpacing: -2, marginBottom: 8, userSelect: 'none', lineHeight: 1 }}>
          <span style={{ color: '#2563eb' }}>S</span>
          <span style={{ color: '#dc2626' }}>o</span>
          <span style={{ color: '#d97706' }}>u</span>
          <span style={{ color: '#2563eb' }}>r</span>
          <span style={{ color: '#059669' }}>c</span>
          <span style={{ color: '#dc2626' }}>e</span>
          <span style={{ color: C.text }}>UP</span>
          <span style={{ color: '#2563eb', fontSize: 44, verticalAlign: 'super' }}>-X</span>
        </div>
        <p style={{ color: C.muted, fontSize: 15, marginBottom: 36, fontWeight: 400, textAlign: 'center', maxWidth: 440 }}>
          Find verified global suppliers with AI that understands your real business constraints
        </p>

        {/* Search bar */}
        <div style={{ width: '100%', maxWidth: 600, marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', border: `1.5px solid ${focused ? C.blue : C.border}`, borderRadius: 50, padding: '0 8px 0 20px', boxShadow: focused ? `0 0 0 4px ${C.blue}15, 0 4px 20px rgba(0,0,0,0.08)` : '0 2px 8px rgba(0,0,0,0.06)', background: C.bg, transition: 'all 0.2s' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
              <circle cx="11" cy="11" r="7" stroke={focused ? C.blue : C.muted} strokeWidth="2" style={{ transition: 'stroke 0.2s' }}/>
              <path d="M20 20l-3-3" stroke={focused ? C.blue : C.muted} strokeWidth="2" strokeLinecap="round" style={{ transition: 'stroke 0.2s' }}/>
            </svg>
            <input
              ref={inputRef}
              style={{ flex: 1, border: 'none', outline: 'none', fontSize: 16, padding: '15px 12px', fontFamily: font, background: 'transparent', color: C.text }}
              placeholder="Search suppliers by product, location, certification…"
              value={query} onChange={e => setQuery(e.target.value)}
              onFocus={() => setFocus(true)} onBlur={() => setFocus(false)}
              onKeyDown={e => e.key === 'Enter' && doSearch()}
            />
            {query && (
              <button style={{ border: 'none', background: 'none', cursor: 'pointer', color: C.muted, fontSize: 20, padding: '0 8px' }}
                onClick={() => { setQuery(''); inputRef.current?.focus(); }}>×</button>
            )}
            <button style={{ ...btn('filled'), borderRadius: 40, height: 38, padding: '0 22px', marginLeft: 4 }} onClick={() => doSearch()}>Search</button>
          </div>
        </div>

        {/* Suggestion chips */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center', marginBottom: 12 }}>
          {suggestions.map(s => (
            <button key={s} style={{ padding: '6px 16px', borderRadius: 20, border: `1px solid ${C.border}`, background: C.bg, fontSize: 13, cursor: 'pointer', color: C.text, fontFamily: font, transition: 'all 0.15s' }}
              onClick={() => doSearch(s)}
              onMouseEnter={e => { e.currentTarget.style.background = C.blueLight; e.currentTarget.style.borderColor = C.blue; e.currentTarget.style.color = C.blue; }}
              onMouseLeave={e => { e.currentTarget.style.background = C.bg; e.currentTarget.style.borderColor = C.border; e.currentTarget.style.color = C.text; }}>
              {s}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
          <button style={{ ...btn('ghost', 'sm'), height: 34, fontSize: 13 }} onClick={onOpenChat}>💬 Ask SourceBot</button>
          <button style={{ ...btn('ghost', 'sm'), height: 34, fontSize: 13 }} onClick={() => doSearch(suggestions[Math.floor(Math.random() * suggestions.length)])}>🎲 I'm Feeling Lucky</button>
        </div>
      </div>

      <ServicesSection onSearch={onSearch} onOpenChat={onOpenChat} onOpenBilling={onOpenBilling} onOpenAuth={onOpenAuth} user={user} />
    </>
  );
}
