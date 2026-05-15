// components/AuthModal.jsx
import { useState } from 'react';
import { C, btn, inputStyle, modalOverlay, modalBox } from '../styles/tokens';
import { API, post } from '../utils/api';
import Spinner from './Spinner';

export default function AuthModal({ onClose, onSuccess }) {
  const [tab, setTab]         = useState('login');
  const [email, setEmail]     = useState('');
  const [password, setPass]   = useState('');
  const [company, setCompany] = useState('');
  const [err, setErr]         = useState('');
  const [loading, setLoad]    = useState(false);
  const [pwShow, setPwShow]   = useState(false);

  const handleGoogleLogin = () => {
    window.location.href = `${API}/auth/google/login`;
  };

  const submit = async () => {
    if (!email || !password) { setErr('Please fill all required fields'); return; }
    if (tab === 'register' && password.length < 8) { setErr('Password must be at least 8 characters'); return; }
    setErr(''); setLoad(true);
    try {
      const url  = tab === 'login' ? '/auth/login' : '/auth/register';
      const body = tab === 'login' ? { email, password } : { email, password, company };
      const data = await post(url, body);
      if (data.access_token) {
        localStorage.setItem('su_token', data.access_token);
        localStorage.setItem('su_email', data.email);
        localStorage.setItem('su_plan', data.plan);
        if (company) localStorage.setItem('su_company', company);
        onSuccess(data);
      } else {
        setErr(data.detail || 'Something went wrong');
      }
    } catch {
      setErr('Network error — is the backend running?');
    }
    setLoad(false);
  };

  return (
    <div style={modalOverlay}>
      <div style={{ ...modalBox, padding: 32 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <h2 style={{ fontFamily: "'DM Serif Display', serif", fontSize: 22, fontWeight: 400 }}>
            {tab === 'login' ? 'Welcome back' : 'Create your account'}
          </h2>
          <button style={{ border: 'none', background: 'none', fontSize: 22, color: C.muted, cursor: 'pointer', lineHeight: 1 }} onClick={onClose}>×</button>
        </div>

        <div style={{ display: 'flex', borderBottom: `1.5px solid ${C.border}`, marginBottom: 24 }}>
          {['login', 'register'].map(t => (
            <button key={t} style={{
              background: 'none', border: 'none', fontSize: 14, fontWeight: 600, padding: '8px 16px',
              borderBottom: `2.5px solid ${tab === t ? C.blue : 'transparent'}`,
              color: tab === t ? C.blue : C.muted, cursor: 'pointer', transition: 'all 0.15s',
              marginBottom: -2,
            }} onClick={() => { setTab(t); setErr(''); }}>
              {t === 'login' ? '🔑 Sign In' : '✨ Register'}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: C.muted, display: 'block', marginBottom: 5 }}>Email address</label>
            <input style={inputStyle} placeholder="you@company.com" type="email" value={email}
              onChange={e => setEmail(e.target.value)} onKeyDown={e => e.key === 'Enter' && submit()}
              onFocus={e => e.target.style.borderColor = C.blue}
              onBlur={e => e.target.style.borderColor = C.border} />
          </div>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: C.muted, display: 'block', marginBottom: 5 }}>Password</label>
            <div style={{ position: 'relative' }}>
              <input style={{ ...inputStyle, paddingRight: 44 }} placeholder={tab === 'register' ? 'Min 8 characters' : '••••••••'}
                type={pwShow ? 'text' : 'password'} value={password}
                onChange={e => setPass(e.target.value)} onKeyDown={e => e.key === 'Enter' && submit()}
                onFocus={e => e.target.style.borderColor = C.blue}
                onBlur={e => e.target.style.borderColor = C.border} />
              <button onClick={() => setPwShow(!pwShow)} style={{
                position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)',
                background: 'none', border: 'none', color: C.muted, fontSize: 12, cursor: 'pointer'
              }}>{pwShow ? '🙈' : '👁'}</button>
            </div>
          </div>
          {tab === 'register' && (
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, color: C.muted, display: 'block', marginBottom: 5 }}>Company name <span style={{ fontWeight: 400 }}>(optional)</span></label>
              <input style={inputStyle} placeholder="Acme Procurement Ltd." value={company}
                onChange={e => setCompany(e.target.value)}
                onFocus={e => e.target.style.borderColor = C.blue}
                onBlur={e => e.target.style.borderColor = C.border} />
            </div>
          )}

          {err && (
            <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, padding: '10px 14px', fontSize: 13, color: C.red }}>
              ⚠️ {err}
            </div>
          )}

          <button style={{ ...btn('filled', 'lg'), width: '100%', marginTop: 4 }} onClick={submit} disabled={loading}>
            {loading ? <><Spinner /> Processing…</> : tab === 'login' ? 'Sign In →' : 'Create Account →'}
          </button>

          {tab === 'login' && (
            <>
              <div style={{ position: 'relative', margin: '4px 0' }}>
                <div style={{ position: 'absolute', inset: '50% 0 auto', borderTop: `1px solid ${C.border}` }} />
                <div style={{ position: 'relative', display: 'flex', justifyContent: 'center' }}>
                  <span style={{ background: C.surface, color: C.subtle, fontSize: 12, padding: '0 10px' }}>OR</span>
                </div>
              </div>

              <button
                type="button"
                onClick={handleGoogleLogin}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                  border: `1px solid ${C.border}`,
                  borderRadius: 10,
                  padding: '10px 16px',
                  background: '#fff',
                  color: C.text,
                  fontSize: 14,
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'background 0.15s',
                }}
                onMouseEnter={e => e.currentTarget.style.background = '#f9fafb'}
                onMouseLeave={e => e.currentTarget.style.background = '#fff'}
              >
                <img
                  src="https://developers.google.com/identity/images/g-logo.png"
                  alt="Google"
                  style={{ width: 20, height: 20 }}
                />
                Continue with Google
              </button>
            </>
          )}

          {tab === 'register' && (
            <p style={{ fontSize: 12, color: C.subtle, textAlign: 'center', lineHeight: 1.6 }}>
              Your credentials are securely stored in MongoDB with bcrypt hashing. By registering, you agree to our Terms of Service.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
