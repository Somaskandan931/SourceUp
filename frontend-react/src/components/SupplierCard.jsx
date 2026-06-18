// components/SupplierCard.jsx
import { useState } from 'react';
import { C, btn, card, pill } from '../styles/tokens';
import { resolveSupplierName, resolveProductName } from '../utils/supplier';
import QuoteModal from './QuoteModal';

// Score is sometimes 0–1 (float) and sometimes 0–100 (int). Normalise to 0–100.
const normaliseScore = (raw) => {
  if (raw == null) return null;
  if (raw > 1) return Math.round(raw);        // already 0–100
  return Math.round(raw * 100);               // 0–1 → 0–100
};

const scoreLabel = (v) => {
  if (v >= 80) return { label: 'Excellent', color: C.green };
  if (v >= 60) return { label: 'Good',      color: '#0891b2' };
  if (v >= 40) return { label: 'Fair',      color: C.amber };
  return              { label: 'Low',       color: C.muted };
};

export default function SupplierCard({ supplier, rank }) {
  const [expanded,  setExpanded]  = useState(false);
  const [showQuote, setShowQuote] = useState(false);
  const plan = localStorage.getItem('su_plan') || 'free';

  const name    = resolveSupplierName(supplier) || `Supplier #${rank}`;
  const product = resolveProductName(supplier, '—');

  const initials = name.split(' ').filter(Boolean).map(w => w[0]).join('').slice(0, 2).toUpperCase();
  const avatarBg = ['#2563eb','#059669','#d97706','#7c3aed','#0891b2','#dc2626'][rank % 6];
  const scoreVal = normaliseScore(supplier.score);
  const { label: sLabel, color: sColor } = scoreVal != null ? scoreLabel(scoreVal) : { label: '', color: C.muted };

  // Only show top 2 reasons — filter out verbose internal strings like "(No price constraint specified)"
  const reasons = (supplier.reasons || [])
    .filter(r => r && !r.includes('(No ') && !r.startsWith('✓') && r.length < 80)
    .slice(0, 2);

  return (
    <>
      <div
        style={{ ...card, marginBottom: 10, transition: 'box-shadow 0.2s, transform 0.2s', padding: '14px 18px' }}
        onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 6px 24px rgba(0,0,0,0.07)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
        onMouseLeave={e => { e.currentTarget.style.boxShadow = 'none'; e.currentTarget.style.transform = 'none'; }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>

          {/* Rank + Avatar */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, flexShrink: 0 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: C.subtle, lineHeight: 1 }}>#{rank}</div>
            <div style={{ width: 40, height: 40, borderRadius: 10, background: avatarBg, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 14, letterSpacing: -0.5 }}>
              {initials}
            </div>
          </div>

          {/* Main info */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2, flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 700, fontSize: 14, color: C.text }}>{name}</span>
              {supplier.location && (
                <span style={{ fontSize: 11, color: C.muted }}>📍 {supplier.location}</span>
              )}
            </div>

            <div style={{ fontSize: 12, color: C.muted, marginBottom: 8, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '95%' }}>
              {product}
            </div>

            {/* Key stats row — compact */}
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
              {supplier.price && supplier.price !== 'Contact for pricing' && (
                <span style={pill(C.green)}>💰 {supplier.price}</span>
              )}
              {supplier.moq && (
                <span style={pill(C.amber)}>MOQ {supplier.moq}</span>
              )}
              {supplier.lead_time && (
                <span style={pill(C.blue)}>⏱ {supplier.lead_time}</span>
              )}
              {reasons.map((r, i) => (
                <span key={i} style={{ background: '#fef9c3', color: '#854d0e', border: '1px solid #fde047', borderRadius: 6, padding: '2px 8px', fontSize: 11, fontWeight: 500 }}>
                  {r}
                </span>
              ))}
            </div>
          </div>

          {/* Score badge + actions */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 8, flexShrink: 0 }}>
            {scoreVal != null && (
              <div style={{ textAlign: 'center', background: `${sColor}12`, borderRadius: 10, padding: '6px 12px', border: `1px solid ${sColor}30` }}>
                <div style={{ fontWeight: 800, fontSize: 20, color: sColor, lineHeight: 1 }}>{scoreVal}%</div>
                <div style={{ fontSize: 10, color: sColor, marginTop: 2, fontWeight: 600 }}>{sLabel}</div>
              </div>
            )}
            <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              {supplier.url && (
                <a href={supplier.url} target="_blank" rel="noreferrer" style={{ ...btn('ghost', 'sm'), textDecoration: 'none', fontSize: 11 }}>Visit ↗</a>
              )}
              <button style={{ ...btn('filled', 'sm'), fontSize: 11 }} onClick={() => {
                if (plan === 'free') { alert('RFQ drafting requires SourceUp Pro.'); return; }
                setShowQuote(true);
              }}>📨 RFQ</button>
              {supplier.decision_trace && (
                <button style={{ ...btn('ghost', 'sm'), fontSize: 11 }} onClick={() => setExpanded(e => !e)}>
                  {expanded ? 'Hide ▴' : 'Trace ▾'}
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Decision trace panel */}
        {expanded && supplier.decision_trace && (
          <div style={{ marginTop: 12, background: C.surface, borderRadius: 8, padding: '12px 14px', fontSize: 12, animation: 'fadeIn 0.2s ease', borderLeft: `3px solid ${C.blue}` }}>
            <div style={{ fontWeight: 700, marginBottom: 8, color: C.text, fontSize: 12 }}>📊 Score breakdown</div>
            {supplier.decision_trace.summary?.map((line, i) => (
              <div key={i} style={{ marginBottom: 4, color: C.muted, lineHeight: 1.5 }}>{line}</div>
            ))}
            {supplier.decision_trace.contributions && (
              <div style={{ marginTop: 10 }}>
                {Object.entries(supplier.decision_trace.contributions).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
                    <div style={{ width: 90, color: C.muted, fontSize: 11, textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ flex: 1, height: 5, background: C.border, borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${Math.min(100, (v.contribution || 0) * 200)}%`, height: '100%', background: `linear-gradient(90deg, ${C.blue}, ${C.indigo})`, borderRadius: 3, transition: 'width 0.5s cubic-bezier(.16,1,.3,1)' }} />
                    </div>
                    <div style={{ width: 30, textAlign: 'right', fontWeight: 700, fontSize: 11, color: C.blue }}>
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