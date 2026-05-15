// components/ChatTab.jsx
import { useState, useRef, useEffect } from 'react';
import { C, btn, inputStyle, font } from '../styles/tokens';
import { post } from '../utils/api';
import { resolveSupplierName, resolveProductName } from '../utils/supplier';
import Spinner from './Spinner';

export default function ChatTab() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: "Hi! I'm SourceBot 🤖 I can help you find the perfect suppliers. Try something like:\n\n\"Find biodegradable food containers from India under $2 with FDA certification\"" }
  ]);
  const [input,   setInput]  = useState('');
  const [loading, setLoad]   = useState(false);
  const [sid]                = useState(`user_${Date.now()}`);
  const endRef   = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const suggestions = ['Eco-friendly packaging India', 'Electronics in China', 'ISO certified textiles', 'Solar panels under $0.5/W'];

  const send = async (text = input) => {
    const msg = text.trim();
    if (!msg || loading) return;
    setInput('');
    setMessages(m => [...m, { role: 'user', text: msg }]);
    setLoad(true);
    try {
      const data = await post('/chat', { session_id: sid, message: msg });
      const suppliers = (data.suppliers || []).filter(s => resolveSupplierName(s) || resolveProductName(s, ''));
      setMessages(m => [...m, { role: 'bot', text: data.message || 'No response', suppliers }]);
    } catch {
      setMessages(m => [...m, { role: 'bot', text: '⚠️ Error connecting to backend. Please check the server is running.' }]);
    }
    setLoad(false);
    setTimeout(() => inputRef.current?.focus(), 50);
  };

  const avatarBg = {
    width: 36, height: 36, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 16, flexShrink: 0, alignSelf: 'flex-end', marginBottom: 2,
  };

  return (
    <div style={{ maxWidth: 820, margin: '0 auto', padding: '0 16px', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)' }}>
      {/* Header */}
      <div style={{ padding: '14px 0 10px', borderBottom: `1px solid ${C.border}`, display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{ ...avatarBg, background: `linear-gradient(135deg, ${C.blue}, ${C.indigo})`, color: '#fff', marginBottom: 0 }}>🤖</div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15 }}>SourceBot</div>
          <div style={{ fontSize: 12, color: C.green, display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: C.green }} />
            Online · AI-powered by Groq
          </div>
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', paddingTop: 16, paddingBottom: 8 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: 14, gap: 8, animation: 'fadeIn 0.25s ease' }}>
            {m.role === 'bot' && (
              <div style={{ ...avatarBg, background: `linear-gradient(135deg, ${C.blue}, ${C.indigo})`, color: '#fff' }}>S</div>
            )}
            <div style={{
              maxWidth: '76%',
              background:   m.role === 'user' ? `linear-gradient(135deg, ${C.blue}, ${C.blueDark})` : C.surface,
              color:        m.role === 'user' ? '#fff' : C.text,
              border:       m.role === 'user' ? 'none' : `1px solid ${C.border}`,
              borderRadius: m.role === 'user' ? '18px 18px 4px 18px' : '4px 18px 18px 18px',
              padding: '12px 16px', fontSize: 14, lineHeight: 1.65,
              boxShadow: '0 1px 4px rgba(0,0,0,0.05)', whiteSpace: 'pre-wrap',
            }}>
              {m.text}
              {m.suppliers?.length > 0 && (
                <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {m.suppliers.slice(0, 4).map((s, j) => {
                    const name    = resolveSupplierName(s) || `Supplier #${j+1}`;
                    const product = resolveProductName(s, '');
                    const avatarC = ['#2563eb','#059669','#d97706','#dc2626'][j % 4];
                    return (
                      <div key={j} style={{ background: C.bg, borderRadius: 10, padding: '10px 14px', border: `1px solid ${C.border}`, fontSize: 13, display: 'flex', alignItems: 'center', gap: 10 }}>
                        <div style={{ width: 32, height: 32, borderRadius: 8, background: avatarC, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 12, flexShrink: 0 }}>
                          {name.slice(0, 2).toUpperCase()}
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontWeight: 700, color: C.text }}>#{j+1} {name}</div>
                          <div style={{ color: C.muted, marginTop: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {product}{s.price ? ` · ${s.price}` : ''}{s.location ? ` · 📍 ${s.location}` : ''}
                          </div>
                        </div>
                        {s.score && <div style={{ fontWeight: 700, fontSize: 12, color: C.blue }}>{Math.round(s.score * 100)}</div>}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            {m.role === 'user' && (
              <div style={{ ...avatarBg, background: C.surface, border: `1px solid ${C.border}`, fontSize: 14, color: C.muted, marginBottom: 0 }}>👤</div>
            )}
          </div>
        ))}

        {/* Typing indicator */}
        {loading && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 14, animation: 'fadeIn 0.25s ease' }}>
            <div style={{ ...avatarBg, background: `linear-gradient(135deg, ${C.blue}, ${C.indigo})`, color: '#fff' }}>S</div>
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: '4px 18px 18px 18px', padding: '14px 18px', display: 'flex', gap: 5, alignItems: 'center' }}>
              {[0,1,2].map(i => (
                <div key={i} className="typing-dot" style={{ width: 7, height: 7, borderRadius: '50%', background: C.blue, animationDelay: `${i*0.15}s` }} />
              ))}
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Suggestion chips (only when fresh) */}
      {messages.length <= 1 && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', paddingBottom: 10 }}>
          {suggestions.map(s => (
            <button key={s} onClick={() => send(s)} style={{ ...btn('ghost', 'sm'), fontSize: 12, borderRadius: 20, padding: '0 12px' }}>{s}</button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={{ display: 'flex', gap: 8, padding: '10px 0 16px', borderTop: `1px solid ${C.border}` }}>
        <input
          ref={inputRef}
          style={{ ...inputStyle, flex: 1, borderRadius: 40, padding: '11px 20px', fontSize: 14, boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}
          placeholder="Message SourceBot…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          onFocus={e => e.target.style.borderColor = C.blue}
          onBlur={e => e.target.style.borderColor = C.border}
        />
        <button style={{ ...btn('filled'), borderRadius: 40, height: 44, width: 44, padding: 0 }}
          onClick={() => send()} disabled={loading || !input.trim()}>
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
