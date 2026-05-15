// components/SearchResultsView.jsx
import { useState, useEffect, useCallback } from 'react';
import { C, btn, card, inputStyle } from '../styles/tokens';
import { post } from '../utils/api';
import { resolveSupplierName, resolveProductName } from '../utils/supplier';
import SupplierCard from './SupplierCard';

export default function SearchResultsView({ initialQuery, onBack }) {
  const [query,    setQuery]    = useState(initialQuery);
  const [results,  setResults]  = useState([]);
  const [meta,     setMeta]     = useState(null);
  const [loading,  setLoad]     = useState(false);
  const [error,    setError]    = useState('');
  const [searched, setSearched] = useState(false);
  const [adv,      setAdv]      = useState(false);

  const [maxPrice,  setMaxP]  = useState('');
  const [moqBudget, setMoqB]  = useState('');
  const [location,  setLoc]   = useState('');
  const [locMand,   setLocM]  = useState(false);
  const [cert,      setCert]  = useState('');
  const [leadTime,  setLead]  = useState('');
  const [minYears,  setYears] = useState('');
  const [explain,   setXpl]   = useState(true);
  const [whatIf,    setWI]    = useState(false);

  const doSearch = useCallback(async (q) => {
    const term = (q || query).trim();
    if (!term) return;
    setError('');
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
    if (data.error) {
      setResults([]);
      setMeta(null);
      setError(data.detail || 'Search failed. Please try again.');
      setLoad(false);
      return;
    }
    const raw     = Array.isArray(data) ? data : (data.results || []);
    const cleaned = raw.filter(s => resolveSupplierName(s) || resolveProductName(s, ''));
    setResults(cleaned);
    setMeta(Array.isArray(data) ? null : data.metadata);
    setLoad(false);
  }, [query, maxPrice, moqBudget, location, locMand, cert, leadTime, minYears, explain, whatIf]);

  useEffect(() => { doSearch(initialQuery); }, []); // eslint-disable-line

  const fi = { ...inputStyle, padding: '8px 12px', fontSize: 13 };

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: '16px 16px 60px' }}>
      {/* Search bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <button style={{ background: 'none', border: 'none', cursor: 'pointer', color: C.muted, fontSize: 22, lineHeight: 1 }} onClick={onBack}>←</button>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', border: `1.5px solid ${C.border}`, borderRadius: 40, padding: '0 16px', boxShadow: '0 1px 6px rgba(0,0,0,0.06)', background: C.bg }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
            <circle cx="11" cy="11" r="7" stroke={C.muted} strokeWidth="2"/>
            <path d="M20 20l-3-3" stroke={C.muted} strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <input style={{ flex: 1, border: 'none', outline: 'none', fontSize: 15, padding: '10px 10px', fontFamily: 'inherit', background: 'transparent', color: C.text }}
            value={query} onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && doSearch()} />
        </div>
        <button style={btn('filled')} onClick={() => doSearch()}>Search</button>
      </div>

      {/* Advanced filters toggle */}
      <button style={{ ...btn('ghost', 'sm'), marginBottom: adv ? 0 : 14, fontSize: 12 }} onClick={() => setAdv(a => !a)}>
        {adv ? '▲ Hide filters' : '⚙️ Advanced filters'}
      </button>

      {adv && (
        <div style={{ ...card, marginBottom: 14, padding: 20, animation: 'fadeIn 0.2s ease' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 12 }}>
            {[
              ['Max unit price (USD)', maxPrice,  setMaxP,  'number', '2.50'],
              ['MOQ budget (USD)',     moqBudget, setMoqB,  'number', '500'],
              ['Location',            location,  setLoc,   'text',   'India'],
              ['Certification',       cert,      setCert,  'text',   'ISO, FDA, CE…'],
              ['Max lead time (days)',leadTime,  setLead,  'number', '30'],
              ['Min platform years',  minYears,  setYears, 'number', '3'],
            ].map(([label, val, setter, type, ph]) => (
              <div key={label}>
                <label style={{ fontSize: 11, fontWeight: 600, color: C.muted, display: 'block', marginBottom: 4 }}>{label}</label>
                <input style={fi} type={type} placeholder={ph} value={val}
                  onChange={e => setter(e.target.value)}
                  onFocus={e => e.target.style.borderColor = C.blue}
                  onBlur={e => e.target.style.borderColor = C.border} />
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            {[['📍 Location mandatory', locMand, setLocM], ['🔍 Show explanations', explain, setXpl], ['🔀 What-if scenarios', whatIf, setWI]].map(([label, val, setter]) => (
              <label key={label} style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 7, cursor: 'pointer', userSelect: 'none', color: C.muted }}>
                <input type="checkbox" checked={val} onChange={e => setter(e.target.checked)} style={{ accentColor: C.blue, width: 14, height: 14 }} />
                <span>{label}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Metadata */}
      {meta && !loading && (
        <div style={{ fontSize: 12, color: C.muted, marginBottom: 12, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <span>📊 {meta.total_candidates?.toLocaleString()} suppliers found</span>
          {meta.after_constraints != null && <span>✓ {meta.after_constraints} matched constraints</span>}
          {meta.latency_ms && <span>⚡ {meta.latency_ms}ms</span>}
          {meta.ranking_method && <span>🏆 Ranked by {meta.ranking_method}</span>}
        </div>
      )}

      {error && !loading && (
        <div style={{ ...card, borderColor: '#fecaca', background: '#fef2f2', color: C.red, marginBottom: 14 }}>
          {error}
        </div>
      )}

      {/* Loading skeletons */}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[1,2,3].map(i => (
            <div key={i} style={{ ...card, height: 100, background: `linear-gradient(90deg, ${C.surface} 25%, ${C.surface2} 50%, ${C.surface} 75%)`, backgroundSize: '200% 100%', animation: 'shimmer 1.4s infinite', border: 'none' }} />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && searched && results.length === 0 && (
        <div style={{ textAlign: 'center', padding: 80, color: C.muted }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>🔍</div>
          <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 6, color: C.text }}>No suppliers matched</div>
          <div style={{ fontSize: 14 }}>Try relaxing your filters or using a different search term.</div>
        </div>
      )}

      {!loading && results.map((r, i) => <SupplierCard key={i} supplier={r} rank={i + 1} />)}
    </div>
  );
}
