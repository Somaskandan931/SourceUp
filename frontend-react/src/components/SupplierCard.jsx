// components/SupplierCard.jsx
import { useState } from 'react';
import { C, btn, card, pill } from '../styles/tokens';
import { resolveSupplierName, resolveProductName } from '../utils/supplier';
import QuoteModal from './QuoteModal';

export default function SupplierCard({ supplier, rank }) {
  const [expanded,  setExpanded]  = useState(false);
  const [showQuote, setShowQuote] = useState(false);
  const plan = localStorage.getItem('su_plan') || 'free';

  const name    = resolveSupplierName(supplier) || `Supplier #${rank}`;
  const product = resolveProductName(supplier, '—');

  const initials  = name.split(' ').filter(Boolean).map(w => w[0]).join('').slice(0, 2).toUpperCase();
  const avatarBg  = ['#2563eb','#059669','#d97706','#dc2626','#7c3aed','#0891b2'][rank % 6];
  const scoreVal  = Math.round((supplier.score || 0) * 100);
  const scoreColor = scoreVal >= 70 ? C.green : scoreVal >= 40 ? C.amber : C.muted;

  return (
    <>
      <div style={{ ...card, marginBottom: 10, transition: 'box-shadow 0.2s, transform 0.2s' }}
        onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.08)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
        onMouseLeave={e => { e.currentTarget.style.boxShadow = 'none'; e.currentTarget.style.transform = 'none'; }}>

        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
          {/* Avatar */}
          <div style={{ width: 44, height: 44, borderRadius: 12, background: avatarBg, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 15, flexShrink: 0, letterSpacing: -0.5 }}>
            {initials}
          </div>

          {/* Info */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 3 }}>
              <span style={{ fontWeight: 700, fontSize: 15 }}>{name}</span>
              {supplier.location && <span style={pill(C.muted)}>📍 {supplier.location}</span>}
              {supplier.moq      && <span style={pill(C.amber)}>MOQ {supplier.moq}</span>}
            </div>
            <div style={{ color: C.muted, fontSize: 13, marginBottom: 8, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{product}</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {supplier.price && supplier.price !== 'Contact for pricing' && (
                <span style={pill(C.green)}>💰 {supplier.price}</span>
              )}
              {supplier.lead_time && <span style={pill(C.blue)}>⏱ {supplier.lead_time}</span>}
              {supplier.reasons?.slice(0, 3).map((r, i) => (
                <span key={i} style={{ background: '#fef9c3', color: '#854d0e', border: '1px solid #fde047', borderRadius: 6, padding: '2px 8px', fontSize: 11, fontWeight: 500 }}>{r}</span>
              ))}
            </div>
          </div>

          {/* Score + actions */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 8, flexShrink: 0 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontWeight: 800, fontSize: 22, color: scoreColor, lineHeight: 1 }}>{scoreVal}</div>
              <div style={{ fontSize: 10, color: C.subtle, marginTop: 1 }}>/ 100</div>
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              {supplier.url && (
                <a href={supplier.url} target="_blank" rel="noreferrer" style={{ ...btn('ghost', 'sm'), textDecoration: 'none' }}>Visit ↗</a>
              )}
              <button style={btn('filled', 'sm')} onClick={() => {
                if (plan === 'free') { alert('RFQ drafting requires SourceUp Pro. Click the Demo button in the nav or Upgrade.'); return; }
                setShowQuote(true);
              }}>📨 RFQ</button>
              {supplier.decision_trace && (
                <button style={btn('ghost', 'sm')} onClick={() => setExpanded(e => !e)}>
                  {expanded ? 'Hide ▴' : 'Trace ▾'}
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Decision trace */}
        {expanded && supplier.decision_trace && (
          <div style={{ marginTop: 16, background: C.surface, borderRadius: 8, padding: 16, fontSize: 12, animation: 'fadeIn 0.2s ease' }}>
            <div style={{ fontWeight: 700, marginBottom: 10, color: C.text, fontSize: 13 }}>📊 Decision Trace</div>
            {supplier.decision_trace.summary?.map((line, i) => (
              <div key={i} style={{ marginBottom: 5, paddingLeft: 12, borderLeft: `3px solid ${C.blue}`, color: C.muted, lineHeight: 1.5 }}>{line}</div>
            ))}
            {supplier.decision_trace.contributions && (
              <div style={{ marginTop: 12 }}>
                {Object.entries(supplier.decision_trace.contributions).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                    <div style={{ width: 100, color: C.muted, fontSize: 11, textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ flex: 1, height: 6, background: C.border, borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${Math.min(100, (v.contribution || 0) * 200)}%`, height: '100%', background: `linear-gradient(90deg, ${C.blue}, ${C.indigo})`, borderRadius: 3, transition: 'width 0.5s cubic-bezier(.16,1,.3,1)' }} />
                    </div>
                    <div style={{ width: 32, textAlign: 'right', fontWeight: 700, fontSize: 11, color: C.blue }}>
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
