export const C = {
  blue:       '#2563eb',
  blueDark:   '#1d4ed8',
  blueLight:  '#eff6ff',
  indigo:     '#4f46e5',
  green:      '#059669',
  greenLight: '#ecfdf5',
  amber:      '#d97706',
  amberLight: '#fffbeb',
  red:        '#dc2626',
  bg:         '#ffffff',
  surface:    '#f8fafc',
  surface2:   '#f1f5f9',
  border:     '#e2e8f0',
  text:       '#0f172a',
  muted:      '#64748b',
  subtle:     '#94a3b8',
};

export const font = "'DM Sans', 'Segoe UI', sans-serif";
export const mono = "'JetBrains Mono', 'Fira Code', monospace";

export const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Serif+Display&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { margin: 0; font-family: ${font}; background: ${C.bg}; color: ${C.text}; }
  button { font-family: inherit; cursor: pointer; }
  input, textarea, select { font-family: inherit; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 3px; }
  @keyframes pulse { 0%,100%{opacity:.8} 50%{opacity:.4} }
  @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
  @keyframes slideUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes typing { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }
  @keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
  .fade-in { animation: fadeIn 0.3s ease forwards; }
  .slide-up { animation: slideUp 0.4s cubic-bezier(.16,1,.3,1) forwards; }
  .spin { animation: spin 0.8s linear infinite; }
  .typing-dot { animation: typing 1.2s ease-in-out infinite; }
  .typing-dot:nth-child(2) { animation-delay: 0.2s; }
  .typing-dot:nth-child(3) { animation-delay: 0.4s; }
`;

// ─── Style helpers ────────────────────────────────────────────────────────────

export const btn = (variant = 'filled', size = 'md') => {
  const sizes = { sm: { h: 30, px: 12, fs: 12 }, md: { h: 38, px: 18, fs: 14 }, lg: { h: 46, px: 24, fs: 15 } };
  const s = sizes[size];
  const base = {
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6,
    height: s.h, padding: `0 ${s.px}px`, fontSize: s.fs, fontWeight: 600,
    borderRadius: 8, border: 'none', transition: 'all 0.15s',
    whiteSpace: 'nowrap', cursor: 'pointer', fontFamily: font,
  };
  if (variant === 'filled')   return { ...base, background: C.blue, color: '#fff' };
  if (variant === 'outlined') return { ...base, background: 'transparent', color: C.blue, border: `1.5px solid ${C.blue}` };
  if (variant === 'ghost')    return { ...base, background: 'transparent', color: C.muted, border: `1px solid ${C.border}` };
  if (variant === 'danger')   return { ...base, background: '#fef2f2', color: C.red, border: `1px solid #fecaca` };
  if (variant === 'success')  return { ...base, background: C.greenLight, color: C.green, border: `1px solid #a7f3d0` };
};

export const pill = (color = C.blue) => ({
  display: 'inline-flex', alignItems: 'center', gap: 3,
  padding: '3px 10px', borderRadius: 20,
  background: color + '15', color, border: `1px solid ${color}30`,
  fontSize: 11, fontWeight: 600, whiteSpace: 'nowrap',
});

export const inputStyle = {
  border: `1.5px solid ${C.border}`, borderRadius: 8, padding: '10px 14px',
  fontSize: 14, outline: 'none', width: '100%', boxSizing: 'border-box',
  fontFamily: font, background: C.bg, color: C.text,
  transition: 'border-color 0.15s, box-shadow 0.15s',
};

export const card = {
  background: C.bg, border: `1px solid ${C.border}`, borderRadius: 12,
  padding: '16px 20px', transition: 'all 0.2s',
};

export const modalOverlay = {
  position: 'fixed', inset: 0, background: 'rgba(15,23,42,0.6)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 500, padding: 16,
  backdropFilter: 'blur(4px)',
};

export const modalBox = {
  background: C.bg, borderRadius: 16, width: '100%',
  maxWidth: 540, maxHeight: '92vh', overflowY: 'auto',
  boxShadow: '0 24px 80px rgba(0,0,0,0.18)',
  animation: 'slideUp 0.35s cubic-bezier(.16,1,.3,1)',
};
