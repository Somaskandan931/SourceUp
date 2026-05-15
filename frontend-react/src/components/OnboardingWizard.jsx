// components/OnboardingWizard.jsx
import { useState } from 'react';
import { C, btn, modalOverlay, modalBox } from '../styles/tokens';

export default function OnboardingWizard({ onComplete }) {
  const [step, setStep] = useState(0);
  const steps = [
    { icon: '🔍', title: 'Welcome to SourceUp-X', body: 'Find verified global suppliers in seconds — with AI that understands your real business constraints.' },
    { icon: '💬', title: 'Chat or Search', body: 'Type naturally ("Find biodegradable containers from India under ₹2 with FDA certification") or use advanced filters.' },
    { icon: '📨', title: 'Smart RFQ Builder', body: 'After finding a supplier, launch the AI RFQ wizard — choose tone, add requirements, preview & refine your email.' },
    { icon: '💳', title: 'Flexible Billing', body: 'Free tier for exploring. Upgrade to Pro or Enterprise via UPI — plans start at ₹999/month.' },
  ];
  const cur = steps[step];

  return (
    <div style={modalOverlay}>
      <div style={{ ...modalBox, textAlign: 'center', padding: 40 }}>
        <div style={{ fontSize: 56, marginBottom: 16 }}>{cur.icon}</div>
        <h2 style={{ fontFamily: "'DM Serif Display', serif", fontSize: 24, fontWeight: 400, margin: '0 0 10px' }}>{cur.title}</h2>
        <p style={{ color: C.muted, lineHeight: 1.7, margin: '0 0 32px', fontSize: 15 }}>{cur.body}</p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: 6, marginBottom: 28 }}>
          {steps.map((_, i) => (
            <div key={i} style={{ width: i === step ? 20 : 8, height: 8, borderRadius: 4, background: i === step ? C.blue : C.border, transition: 'all 0.3s' }} />
          ))}
        </div>
        <div style={{ display: 'flex', gap: 10, justifyContent: 'center' }}>
          {step > 0 && <button style={btn('ghost')} onClick={() => setStep(s => s - 1)}>← Back</button>}
          {step < steps.length - 1
            ? <button style={btn('filled', 'lg')} onClick={() => setStep(s => s + 1)}>Next →</button>
            : <button style={btn('filled', 'lg')} onClick={onComplete}>Get Started 🚀</button>
          }
        </div>
        <button style={{ marginTop: 18, background: 'none', border: 'none', color: C.subtle, cursor: 'pointer', fontSize: 13 }} onClick={onComplete}>Skip tour</button>
      </div>
    </div>
  );
}
