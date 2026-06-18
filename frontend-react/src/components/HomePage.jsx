// components/HomePage.jsx
import { useState, useRef } from 'react';
import { C, btn, font } from '../styles/tokens';

// ─── Stat ticker ──────────────────────────────────────────────────────────────
const stats = [
  { value: '50K+', label: 'Verified suppliers' },
  { value: '<250ms', label: 'Median search time' },
  { value: '12+', label: 'Countries covered' },
  { value: 'SHAP', label: 'Explainable AI' },
];

// ─── Category tags ────────────────────────────────────────────────────────────
const categories = [
  { icon: '📦', label: 'Packaging' },
  { icon: '💡', label: 'Electronics' },
  { icon: '🧵', label: 'Textiles' },
  { icon: '🌱', label: 'Eco & Green' },
  { icon: '⚙️', label: 'Industrial' },
  { icon: '☀️', label: 'Renewable Energy' },
];

const suggestions = [
  'Biodegradable packaging India',
  'ISO certified textiles',
  'LED bulbs Vietnam',
  'Electronics China',
  'Solar panels Taiwan',
];

export default function HomePage({ onSearch, onOpenChat, onOpenBilling, onOpenAuth, user }) {
  const [query,   setQuery]  = useState('');
  const [focused, setFocus]  = useState(false);
  const inputRef = useRef(null);

  const doSearch = (q) => { const t = (q || query).trim(); if (t) onSearch(t); };

  return (
    <div style={{ minHeight: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column' }}>

      {/* ── Hero ── */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 0,
        maxWidth: 1100,
        margin: '0 auto',
        padding: '64px 32px 40px',
        alignItems: 'center',
        width: '100%',
      }}>

        {/* Left column — copy + search */}
        <div style={{ paddingRight: 48 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 20, padding: '4px 14px', marginBottom: 24 }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: C.blue, display: 'inline-block' }} />
            <span style={{ fontSize: 11, fontWeight: 700, color: C.blue, letterSpacing: 1, textTransform: 'uppercase' }}>Explainable Procurement AI</span>
          </div>

          <h1 style={{
            fontFamily: "'DM Serif Display', serif",
            fontSize: 46,
            fontWeight: 400,
            lineHeight: 1.12,
            margin: '0 0 16px',
            color: C.text,
            letterSpacing: -1,
          }}>
            Source smarter.<br />
            <span style={{ color: C.blue }}>Know why.</span>
          </h1>

          <p style={{ fontSize: 15, color: C.muted, lineHeight: 1.7, margin: '0 0 32px', maxWidth: 400 }}>
            FAISS-powered semantic search across verified global suppliers — with constraint filtering, LightGBM ranking, and full SHAP decision traces.
          </p>

          {/* Search bar */}
          <div style={{ marginBottom: 16 }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              border: `2px solid ${focused ? C.blue : C.border}`,
              borderRadius: 12,
              padding: '0 6px 0 16px',
              boxShadow: focused ? `0 0 0 4px ${C.blue}18` : '0 2px 10px rgba(0,0,0,0.06)',
              background: '#fff',
              transition: 'all 0.2s',
            }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
                <circle cx="11" cy="11" r="7" stroke={focused ? C.blue : C.muted} strokeWidth="2" style={{ transition: 'stroke 0.2s' }} />
                <path d="M20 20l-3-3" stroke={focused ? C.blue : C.muted} strokeWidth="2" strokeLinecap="round" style={{ transition: 'stroke 0.2s' }} />
              </svg>
              <input
                ref={inputRef}
                style={{ flex: 1, border: 'none', outline: 'none', fontSize: 15, padding: '14px 12px', fontFamily: font, background: 'transparent', color: C.text }}
                placeholder="e.g. biodegradable packaging India..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                onFocus={() => setFocus(true)}
                onBlur={() => setFocus(false)}
                onKeyDown={e => e.key === 'Enter' && doSearch()}
              />
              {query && (
                <button style={{ border: 'none', background: 'none', cursor: 'pointer', color: C.muted, fontSize: 18, padding: '0 6px' }}
                  onClick={() => { setQuery(''); inputRef.current?.focus(); }}>×</button>
              )}
              <button
                style={{ ...btn('filled'), borderRadius: 8, height: 40, padding: '0 20px', margin: '4px 0' }}
                onClick={() => doSearch()}
              >Search</button>
            </div>
          </div>

          {/* Suggestions */}
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 20 }}>
            {suggestions.slice(0, 4).map(s => (
              <button key={s}
                style={{ padding: '5px 12px', borderRadius: 6, border: `1px solid ${C.border}`, background: C.surface, fontSize: 12, cursor: 'pointer', color: C.muted, fontFamily: font, transition: 'all 0.15s' }}
                onClick={() => doSearch(s)}
                onMouseEnter={e => { e.currentTarget.style.background = C.blueLight; e.currentTarget.style.borderColor = C.blue; e.currentTarget.style.color = C.blue; }}
                onMouseLeave={e => { e.currentTarget.style.background = C.surface; e.currentTarget.style.borderColor = C.border; e.currentTarget.style.color = C.muted; }}
              >{s}</button>
            ))}
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <button style={{ ...btn('ghost', 'sm'), fontSize: 12 }} onClick={onOpenChat}>💬 Ask SourceBot</button>
            <button style={{ ...btn('ghost', 'sm'), fontSize: 12 }} onClick={() => doSearch(suggestions[Math.floor(Math.random() * suggestions.length)])}>🎲 Feeling lucky</button>
          </div>
        </div>

        {/* Right column — visual panel */}
        <div style={{ position: 'relative' }}>
          {/* Mock result cards — decorative */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { name: 'EcoWrap India Pvt Ltd', loc: 'Mumbai, IN', score: 94, color: C.green, tag: 'ISO 14001' },
              { name: 'GreenPack Solutions', loc: 'Chennai, IN', score: 81, color: '#0891b2', tag: 'FSC Certified' },
              { name: 'BioBox Manufacturing', loc: 'Delhi, IN', score: 72, color: C.amber, tag: 'MOQ 500' },
            ].map((c, i) => (
              <div key={i} style={{
                background: '#fff',
                border: `1px solid ${C.border}`,
                borderRadius: 12,
                padding: '12px 16px',
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
                transform: `translateX(${i * 8}px)`,
                opacity: 1 - i * 0.08,
                transition: 'all 0.2s',
              }}>
                <div style={{ width: 36, height: 36, borderRadius: 9, background: c.color + '20', border: `1.5px solid ${c.color}40`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 13, color: c.color, flexShrink: 0 }}>
                  {c.name.split(' ').map(w => w[0]).join('').slice(0, 2)}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: C.text, marginBottom: 2 }}>{c.name}</div>
                  <div style={{ fontSize: 11, color: C.muted }}>{c.loc} · <span style={{ color: c.color, fontWeight: 600 }}>{c.tag}</span></div>
                </div>
                <div style={{ textAlign: 'center', background: `${c.color}12`, borderRadius: 8, padding: '5px 10px', border: `1px solid ${c.color}25` }}>
                  <div style={{ fontWeight: 800, fontSize: 18, color: c.color, lineHeight: 1 }}>{c.score}%</div>
                  <div style={{ fontSize: 9, color: c.color, fontWeight: 600 }}>match</div>
                </div>
              </div>
            ))}
          </div>

          {/* Floating SHAP label */}
          <div style={{
            position: 'absolute', bottom: -20, right: 0,
            background: '#fff', border: `1px solid ${C.border}`,
            borderRadius: 10, padding: '8px 14px', boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
            display: 'flex', alignItems: 'center', gap: 8, fontSize: 12,
          }}>
            <span style={{ fontSize: 16 }}>📊</span>
            <div>
              <div style={{ fontWeight: 700, fontSize: 12, color: C.text }}>SHAP Explainability</div>
              <div style={{ fontSize: 11, color: C.muted }}>See exactly why each result was ranked</div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Stats strip ── */}
      <div style={{ borderTop: `1px solid ${C.border}`, borderBottom: `1px solid ${C.border}`, background: C.surface }}>
        <div style={{ maxWidth: 1100, margin: '0 auto', padding: '16px 32px', display: 'flex', justifyContent: 'space-around', gap: 16, flexWrap: 'wrap' }}>
          {stats.map(s => (
            <div key={s.label} style={{ textAlign: 'center' }}>
              <div style={{ fontWeight: 800, fontSize: 20, color: C.blue, letterSpacing: -0.5 }}>{s.value}</div>
              <div style={{ fontSize: 11, color: C.muted, fontWeight: 500 }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Browse by category ── */}
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '40px 32px 60px', width: '100%' }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: C.blue, textTransform: 'uppercase', letterSpacing: 1.5, marginBottom: 8 }}>Browse by category</div>
        <h2 style={{ fontFamily: "'DM Serif Display', serif", fontSize: 26, fontWeight: 400, margin: '0 0 24px', color: C.text }}>What are you sourcing today?</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 10 }}>
          {categories.map((cat) => (
            <button key={cat.label}
              style={{
                background: '#fff', border: `1px solid ${C.border}`, borderRadius: 12,
                padding: '18px 12px', cursor: 'pointer', fontFamily: font, transition: 'all 0.15s',
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
              }}
              onClick={() => doSearch(cat.label)}
              onMouseEnter={e => { e.currentTarget.style.borderColor = C.blue; e.currentTarget.style.boxShadow = `0 4px 16px ${C.blue}18`; e.currentTarget.style.transform = 'translateY(-2px)'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.boxShadow = 'none'; e.currentTarget.style.transform = 'none'; }}
            >
              <span style={{ fontSize: 24 }}>{cat.icon}</span>
              <span style={{ fontSize: 12, fontWeight: 600, color: C.muted }}>{cat.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}