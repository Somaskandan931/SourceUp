// components/ChatTab.jsx
import { useState, useRef, useEffect } from 'react';
import { C, btn, inputStyle, font } from '../styles/tokens';
import { post } from '../utils/api';
import { resolveSupplierName, resolveProductName } from '../utils/supplier';
import Spinner from './Spinner';

const SAMPLES = [
  { icon: '📋', text: 'What do I need to become a SourceUp supplier?' },
  { icon: '✅', text: 'How do I get my company verified as a supplier?' },
  { icon: '🧾', text: 'What are the requirements for listing my products on SourceUp?' },
  { icon: '🏗️', text: 'How can I create a supplier profile on SourceUp?' },
  { icon: '🤝', text: "What's the difference between a buyer and supplier account?" },
  { icon: '🎓', text: 'Is it hard to get ISO certification as a new supplier?' },
];

const normScore = (raw) => {
  if (raw == null) return null;
  return raw > 1 ? Math.round(raw) : Math.round(raw * 100);
};

function MiniSupplierCard({ s, rank }) {
  const name    = resolveSupplierName(s) || ('Supplier #' + rank);
  const product = resolveProductName(s, '');
  const colors  = [C.blue, C.green, '#7c3aed', C.amber, '#0891b2', C.red];
  const avatarC = colors[rank % colors.length];
  const score   = normScore(s.score);
  const sColor  = score >= 70 ? C.green : score >= 50 ? '#0891b2' : C.amber;

  return (
    <div style={{ background: '#fff', borderRadius: 10, padding: '10px 14px', border: '1px solid ' + C.border, fontSize: 13, display: 'flex', alignItems: 'center', gap: 10 }}>
      <div style={{ width: 32, height: 32, borderRadius: 8, background: avatarC, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 12, flexShrink: 0 }}>
        {name.slice(0, 2).toUpperCase()}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 700, color: C.text }}>#{rank} {name}</div>
        <div style={{ color: C.muted, marginTop: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {product}{s.price ? ' · ' + s.price : ''}{s.location ? ' · 📍 ' + s.location : ''}
        </div>
      </div>
      {score != null && (
        <div style={{ fontWeight: 800, fontSize: 13, color: sColor, background: sColor + '12', borderRadius: 6, padding: '3px 8px' }}>
          {score}%
        </div>
      )}
    </div>
  );
}

function WelcomeScreen({ onSelect }) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px 24px' }}>
      <div style={{ fontSize: 36, marginBottom: 12 }}>👋</div>
      <h2 style={{ fontFamily: "'DM Serif Display', serif", fontSize: 28, fontWeight: 400, margin: '0 0 8px', color: C.text, letterSpacing: -0.5 }}>
        Welcome to <span style={{ color: C.blue }}>SourceBot-X</span>
      </h2>
      <p style={{ fontSize: 14, color: C.muted, margin: '0 0 32px', textAlign: 'center', maxWidth: 380, lineHeight: 1.6 }}>
        Ask me about becoming a supplier, certifications, or how SourceUp works — or search for suppliers directly with price, location, and MOQ in one message.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: '100%', maxWidth: 520 }}>
        {SAMPLES.map(function(item) {
          return (
            <button key={item.text}
              onClick={function() { onSelect(item.text); }}
              style={{ display: 'flex', alignItems: 'center', gap: 12, background: '#fff', border: '1.5px solid ' + C.border, borderRadius: 12, padding: '12px 18px', fontSize: 14, cursor: 'pointer', color: C.text, fontFamily: font, textAlign: 'left', transition: 'all 0.15s', boxShadow: '0 1px 4px rgba(0,0,0,0.04)' }}
              onMouseEnter={function(e) { e.currentTarget.style.borderColor = C.blue; e.currentTarget.style.background = C.blueLight; e.currentTarget.style.color = C.blue; }}
              onMouseLeave={function(e) { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.background = '#fff'; e.currentTarget.style.color = C.text; }}
            >
              <span style={{ fontSize: 18, flexShrink: 0 }}>{item.icon}</span>
              <span style={{ flex: 1 }}>{item.text}</span>
              <span style={{ fontSize: 16, color: C.subtle, flexShrink: 0 }}>→</span>
            </button>
          );
        })}
      </div>
      <div style={{ marginTop: 24, fontSize: 12, color: C.subtle, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span>💡</span>
        <span>Tip: ask anything platform-related — <em>"What do I need to become a supplier?"</em></span>
      </div>
    </div>
  );
}

export default function ChatTab() {
  const [messages, setMessages] = useState([]);
  const [input,    setInput]    = useState('');
  const [loading,  setLoad]     = useState(false);
  const [sid]                   = useState('user_' + Date.now());
  const endRef   = useRef(null);
  const inputRef = useRef(null);

  useEffect(function() { if (endRef.current) endRef.current.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const send = async function(text) {
    var msg = (text !== undefined ? text : input).trim();
    if (!msg || loading) return;
    setInput('');
    setMessages(function(m) { return [...m, { role: 'user', text: msg }]; });
    setLoad(true);
    try {
      var data = await post('/chat', { session_id: sid, message: msg });
      var suppliers = (data.suppliers || []).filter(function(s) { return resolveSupplierName(s) || resolveProductName(s, ''); });
      setMessages(function(m) { return [...m, { role: 'bot', text: data.message || 'No response', suppliers: suppliers }]; });
    } catch(err) {
      setMessages(function(m) { return [...m, { role: 'bot', text: '⚠️ Error connecting to backend. Please check the server is running.' }]; });
    }
    setLoad(false);
    setTimeout(function() { if (inputRef.current) inputRef.current.focus(); }, 50);
  };

  var botAvatar = { width: 34, height: 34, borderRadius: 9, flexShrink: 0, alignSelf: 'flex-start', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(135deg, ' + C.blue + ', #4f46e5)', color: '#fff', fontSize: 16, marginTop: 2 };
  var userAvatar = { width: 34, height: 34, borderRadius: 9, flexShrink: 0, alignSelf: 'flex-end', display: 'flex', alignItems: 'center', justifyContent: 'center', background: C.surface2, border: '1px solid ' + C.border, color: C.muted, fontSize: 14, marginBottom: 2 };

  return (
    <div style={{ maxWidth: 820, margin: '0 auto', padding: '0 16px', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)' }}>

      <div style={{ padding: '12px 0 10px', borderBottom: '1px solid ' + C.border, display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{ width: 36, height: 36, borderRadius: 10, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(135deg, ' + C.blue + ', #4f46e5)', color: '#fff', fontSize: 18 }}>🤖</div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15 }}>SourceBot</div>
          <div style={{ fontSize: 12, color: C.green, display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: C.green }} />
            Online · AI-powered by Groq
          </div>
        </div>
      </div>

      {messages.length === 0 && !loading ? (
        <WelcomeScreen onSelect={send} />
      ) : (
        <div style={{ flex: 1, overflowY: 'auto', paddingTop: 14, paddingBottom: 8 }}>
          {messages.map(function(m, i) {
            return (
              <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: 14, gap: 8, animation: 'fadeIn 0.25s ease' }}>
                {m.role === 'bot' && <div style={botAvatar}>🤖</div>}
                <div style={{ maxWidth: '76%', background: m.role === 'user' ? 'linear-gradient(135deg, ' + C.blue + ', ' + C.blueDark + ')' : C.surface, color: m.role === 'user' ? '#fff' : C.text, border: m.role === 'user' ? 'none' : '1px solid ' + C.border, borderRadius: m.role === 'user' ? '18px 18px 4px 18px' : '4px 18px 18px 18px', padding: '12px 16px', fontSize: 14, lineHeight: 1.65, boxShadow: '0 1px 4px rgba(0,0,0,0.05)', whiteSpace: 'pre-wrap' }}>
                  {m.text}
                  {m.suppliers && m.suppliers.length > 0 && (
                    <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {m.suppliers.slice(0, 5).map(function(s, j) {
                        return <MiniSupplierCard key={j} s={s} rank={j + 1} />;
                      })}
                    </div>
                  )}
                </div>
                {m.role === 'user' && <div style={userAvatar}>👤</div>}
              </div>
            );
          })}
          {loading && (
            <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
              <div style={botAvatar}>🤖</div>
              <div style={{ background: C.surface, border: '1px solid ' + C.border, borderRadius: '4px 18px 18px 18px', padding: '14px 18px', display: 'flex', gap: 5, alignItems: 'center' }}>
                {[0,1,2].map(function(i) {
                  return <div key={i} className="typing-dot" style={{ width: 7, height: 7, borderRadius: '50%', background: C.blue, animationDelay: (i * 0.15) + 's' }} />;
                })}
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>
      )}

      <div style={{ padding: '10px 0 16px', borderTop: '1px solid ' + C.border, display: 'flex', gap: 8 }}>
        <input
          ref={inputRef}
          style={{ ...inputStyle, flex: 1, borderRadius: 40, padding: '11px 20px', fontSize: 14, boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}
          placeholder="Ask me about suppliers…"
          value={input}
          onChange={function(e) { setInput(e.target.value); }}
          onKeyDown={function(e) { if (e.key === 'Enter' && !e.shiftKey) send(); }}
          onFocus={function(e) { e.target.style.borderColor = C.blue; }}
          onBlur={function(e) { e.target.style.borderColor = C.border; }}
        />
        <button
          style={{ ...btn('filled'), borderRadius: 40, height: 44, width: 44, padding: 0 }}
          onClick={function() { send(); }}
          disabled={loading || !input.trim()}
        >
          {loading ? <Spinner size={16} /> : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M22 2L11 13" stroke="white" strokeWidth="2" strokeLinecap="round"/>
              <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
