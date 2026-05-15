// components/Navbar.jsx
import { C, btn, pill, font } from '../styles/tokens';

export default function Navbar({ view, setView, user, onOpenAuth, onOpenBilling, onDemoLogin, onLogout }) {
  return (
    <nav style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 24px', borderBottom: `1px solid ${C.border}`,
      position: 'sticky', top: 0, background: 'rgba(255,255,255,0.95)',
      backdropFilter: 'blur(8px)', zIndex: 200, height: 64,
    }}>
      {/* Brand */}
      <div style={{ fontFamily: "'DM Serif Display', serif", fontWeight: 400, fontSize: 22, cursor: 'pointer', userSelect: 'none', letterSpacing: -0.5 }}
        onClick={() => setView('home')}>
        SourceUP<span style={{ color: C.blue }}>-X</span>
      </div>

      {/* Tab switcher (non-home) */}
      {view !== 'home' && (
        <div style={{ display: 'flex', gap: 0 }}>
          {[['search','🔍 Search'], ['chat','💬 Chat']].map(([t, label]) => (
            <button key={t} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              borderBottom: `2.5px solid ${view === t ? C.blue : 'transparent'}`,
              color: view === t ? C.blue : C.muted,
              padding: '0 18px', fontFamily: font, fontWeight: 600,
              fontSize: 14, height: 64, transition: 'all 0.15s', marginBottom: -1,
            }} onClick={() => setView(t)}>{label}</button>
          ))}
        </div>
      )}

      {/* Auth area */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {user ? (
          <>
            <span style={pill(user.plan === 'free' ? C.muted : user.plan === 'enterprise' ? '#7c3aed' : C.blue)}>
              {user.plan === 'free' ? '🆓 Free' : user.plan === 'enterprise' ? '⚡ Enterprise' : '⭐ Pro'}
            </span>
            {user.plan === 'free' && (
              <button style={btn('outlined', 'sm')} onClick={onOpenBilling}>Upgrade</button>
            )}
            <span style={{ fontSize: 13, color: C.muted, maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{user.email}</span>
            <button style={btn('ghost', 'sm')} onClick={onLogout}>Sign out</button>
          </>
        ) : (
          <>
            <button style={btn('ghost', 'sm')} onClick={onOpenAuth}>Sign in</button>
            <button style={{ ...btn('outlined', 'sm'), background: C.blueLight }} onClick={onDemoLogin}>🎬 Demo (Pro)</button>
            <button style={btn('filled', 'sm')} onClick={onOpenAuth}>Get Started</button>
          </>
        )}
      </div>
    </nav>
  );
}