/**
 * components/ChatButton.jsx — Floating chat button (FAB)
 * -------------------------------------------------------
 * Clean bottom-right FAB that expands into a compact chat launcher tooltip.
 *
 * Usage: <ChatButton onClick={() => setShowChat(true)} unreadCount={2} />
 */
import { useState } from 'react';

export default function ChatButton({ onClick, unreadCount = 0 }) {
  const [hovered, setHovered] = useState(false);

  return (
    <div style={s.wrapper}>
      {hovered && (
        <div style={s.tooltip}>
          <span style={s.tooltipDot} />
          SourceBot is online
        </div>
      )}
      <button
        style={{ ...s.fab, transform: hovered ? 'scale(1.08)' : 'scale(1)' }}
        onClick={onClick}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        aria-label="Open SourceBot chat"
      >
        {unreadCount > 0 && (
          <span style={s.badge}>{unreadCount > 9 ? '9+' : unreadCount}</span>
        )}
        <ChatIcon />
      </button>
    </div>
  );
}

function ChatIcon() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff"
         strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>
  );
}

const s = {
  wrapper: {
    position: 'fixed', bottom: 28, right: 28, zIndex: 1000,
    display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 10,
  },
  tooltip: {
    background: '#111827', color: '#fff', fontSize: 13, fontWeight: 500,
    padding: '8px 14px', borderRadius: 20, whiteSpace: 'nowrap',
    boxShadow: '0 4px 16px rgba(0,0,0,0.18)', display: 'flex', alignItems: 'center', gap: 6,
    animation: 'fadeInUp .15s ease',
  },
  tooltipDot: {
    width: 8, height: 8, borderRadius: '50%', background: '#22c55e', display: 'inline-block',
  },
  fab: {
    width: 56, height: 56, borderRadius: '50%',
    background: 'linear-gradient(135deg, #1a56db 0%, #3b82f6 100%)',
    border: 'none', cursor: 'pointer',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    boxShadow: '0 6px 24px rgba(26,86,219,0.4)',
    transition: 'transform .15s ease, box-shadow .15s ease',
    position: 'relative',
  },
  badge: {
    position: 'absolute', top: 0, right: 0,
    background: '#ef4444', color: '#fff', fontSize: 10, fontWeight: 700,
    padding: '2px 5px', borderRadius: 10, minWidth: 18,
    textAlign: 'center', border: '2px solid #fff',
  },
};
