// components/BillingModal.jsx
import { useState, useEffect } from 'react';
import { C, btn, inputStyle, card, modalOverlay, modalBox, mono } from '../styles/tokens';
import { post, get } from '../utils/api';
import Spinner from './Spinner';

export default function BillingModal({ onClose, currentPlan = 'free' }) {
  const [plans, setPlans]     = useState([]);
  const [step, setStep]       = useState('plans'); // plans | paying | done
  const [msg, setMsg]         = useState('');
  const [orderId, setOrderId] = useState('');
  const [upiId, setUpiId]     = useState('');
  const [amount, setAmount]   = useState(0);
  const [utr, setUtr]         = useState('');
  const [pendingPlan, setPending] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => { get('/auth/billing/plans').then(setPlans).catch(() => {}); }, []);

  const createOrder = async (planId) => {
    setLoading(true); setMsg('');
    try {
      const order = await post('/auth/billing/order', { plan: planId });
      if (!order.order_id) { setMsg(order.detail || 'Order creation failed'); setLoading(false); return; }
      setPending(planId); setOrderId(order.order_id);
      setUpiId(order.upi_id); setAmount(order.amount);
      setStep('paying');
    } catch (e) { setMsg('Error: ' + e.message); }
    setLoading(false);
  };

  const verifyPayment = async () => {
    if (!utr.trim()) { setMsg('Please enter the UTR / Transaction ID'); return; }
    setLoading(true);
    try {
      const res = await post('/auth/billing/verify', { order_id: orderId, upi_transaction_id: utr.trim(), plan: pendingPlan });
      if (res.success) {
        localStorage.setItem('su_plan', pendingPlan);
        localStorage.setItem('su_token', res.new_token);
        setStep('done');
      } else {
        setMsg('❌ Verification failed. Please check the UTR and try again.');
      }
    } catch (e) { setMsg('Error: ' + e.message); }
    setLoading(false);
  };

  const planTitle = pendingPlan.charAt(0).toUpperCase() + pendingPlan.slice(1);

  return (
    <div style={modalOverlay}>
      <div style={{ ...modalBox, maxWidth: 600, padding: 32 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <h2 style={{ fontFamily: "'DM Serif Display', serif", fontSize: 22, fontWeight: 400 }}>
            {step === 'plans' ? '💳 Plans & Pricing' : step === 'paying' ? '📲 Complete Payment' : '🎉 Upgrade Successful!'}
          </h2>
          <button style={{ border: 'none', background: 'none', fontSize: 22, color: C.muted, cursor: 'pointer' }} onClick={onClose}>×</button>
        </div>

        {/* Step: Plans */}
        {step === 'plans' && (
          <>
            {msg && <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, padding: 12, marginBottom: 16, fontSize: 13, color: C.red }}>{msg}</div>}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {plans.map(p => (
                <div key={p.id} style={{ ...card, padding: '20px 24px', border: `2px solid ${p.id === 'pro' ? C.blue : p.id === currentPlan ? C.green : C.border}`, position: 'relative', overflow: 'hidden' }}>
                  {p.id === 'pro' && (
                    <div style={{ position: 'absolute', top: 0, right: 0, background: C.blue, color: '#fff', fontSize: 10, fontWeight: 700, padding: '4px 12px', borderBottomLeftRadius: 8 }}>POPULAR</div>
                  )}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: 16 }}>{p.name}</div>
                      <div style={{ color: C.muted, fontSize: 13, marginTop: 2 }}>
                        {p.price_inr === 0 ? 'Forever free' : `₹${p.price_inr.toLocaleString()}/month`}
                      </div>
                    </div>
                    {p.id === currentPlan
                      ? <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3, padding: '3px 10px', borderRadius: 20, background: C.green + '15', color: C.green, border: `1px solid ${C.green}30`, fontSize: 11, fontWeight: 600 }}>✓ Current Plan</span>
                      : p.id !== 'free'
                      ? <button style={btn(p.id === 'pro' ? 'filled' : 'outlined')} onClick={() => createOrder(p.id)} disabled={loading}>
                          {loading ? <Spinner color={p.id === 'pro' ? '#fff' : C.blue} /> : 'Upgrade →'}
                        </button>
                      : null
                    }
                  </div>
                  <ul style={{ marginTop: 12, paddingLeft: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {p.features.map((f, i) => (
                      <li key={i} style={{ fontSize: 13, color: C.muted, display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: C.green }}>✓</span> {f}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Step: Paying */}
        {step === 'paying' && (
          <div style={{ animation: 'fadeIn 0.3s ease' }}>
            <div style={{ background: C.blueLight, border: `1px solid ${C.blue}30`, borderRadius: 12, padding: 20, marginBottom: 20 }}>
              <div style={{ fontSize: 13, color: C.muted, marginBottom: 8 }}>Payment details</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div><div style={{ fontSize: 11, color: C.subtle, marginBottom: 2 }}>UPI ID</div><div style={{ fontWeight: 700, fontFamily: mono }}>{upiId || 'Not configured'}</div></div>
                <div><div style={{ fontSize: 11, color: C.subtle, marginBottom: 2 }}>Amount</div><div style={{ fontWeight: 700, fontSize: 20, color: C.green }}>₹{amount.toLocaleString()}</div></div>
                <div><div style={{ fontSize: 11, color: C.subtle, marginBottom: 2 }}>Plan</div><div style={{ fontWeight: 600 }}>{planTitle}</div></div>
                <div><div style={{ fontSize: 11, color: C.subtle, marginBottom: 2 }}>Order ID</div><div style={{ fontFamily: mono, fontSize: 11, color: C.muted }}>{orderId.slice(0, 16)}…</div></div>
              </div>
            </div>
            <div style={{ background: C.amberLight, border: `1px solid ${C.amber}30`, borderRadius: 8, padding: '12px 16px', fontSize: 13, color: C.amber, marginBottom: 20 }}>
              📲 Open any UPI app (GPay, PhonePe, Paytm) → Pay to <strong>{upiId}</strong> → Copy the UTR / Transaction ID
            </div>
            <div style={{ marginBottom: 12 }}>
              <label style={{ fontSize: 12, fontWeight: 600, color: C.muted, display: 'block', marginBottom: 6 }}>UTR / UPI Transaction ID</label>
              <input style={inputStyle} placeholder="e.g. 401234567890" value={utr}
                onChange={e => setUtr(e.target.value)}
                onFocus={e => e.target.style.borderColor = C.blue}
                onBlur={e => e.target.style.borderColor = C.border} />
            </div>
            {msg && <div style={{ background: '#fef2f2', borderRadius: 8, padding: 12, fontSize: 13, color: C.red, marginBottom: 12 }}>{msg}</div>}
            <div style={{ display: 'flex', gap: 10 }}>
              <button style={{ ...btn('ghost'), flex: 1 }} onClick={() => setStep('plans')}>← Back</button>
              <button style={{ ...btn('filled'), flex: 2, height: 44 }} onClick={verifyPayment} disabled={loading}>
                {loading ? <><Spinner /> Verifying…</> : '✓ Verify Payment'}
              </button>
            </div>
          </div>
        )}

        {/* Step: Done */}
        {step === 'done' && (
          <div style={{ textAlign: 'center', padding: '20px 0', animation: 'fadeIn 0.4s ease' }}>
            <div style={{ fontSize: 64, marginBottom: 16 }}>🎉</div>
            <h3 style={{ fontSize: 20, marginBottom: 8 }}>You're on {planTitle}!</h3>
            <p style={{ color: C.muted, marginBottom: 24, lineHeight: 1.6 }}>Your account has been upgraded. Refresh the page to unlock all Pro features including AI-powered RFQ drafting, decision traces, and what-if scenarios.</p>
            <button style={btn('filled', 'lg')} onClick={() => window.location.reload()}>Refresh & Continue →</button>
          </div>
        )}
      </div>
    </div>
  );
}
