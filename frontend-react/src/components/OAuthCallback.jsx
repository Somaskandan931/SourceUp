/**
 * components/OAuthCallback.jsx
 * ----------------------------
 * Handles the redirect back from Google OAuth.
 * Mount at the /oauth-callback route in your React app.
 *
 * Backend redirects to:
 *   http://localhost:3000/oauth-callback?token=JWT&email=x@y.com&plan=free
 */
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function OAuthCallback() {
  const navigate = useNavigate();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token  = params.get('token');
    const email  = params.get('email');
    const plan   = params.get('plan');

    if (token && email) {
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify({ email, plan }));
      navigate('/', { replace: true });
    } else {
      navigate('/login?error=oauth_failed', { replace: true });
    }
  }, [navigate]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', background: '#f0f4ff' }}>
      <div style={{ background: 'white', borderRadius: 12, padding: '48px 64px', boxShadow: '0 4px 24px rgba(0,0,0,0.08)', textAlign: 'center' }}>
        <div style={{ fontSize: 40, marginBottom: 16 }}>🔐</div>
        <p style={{ color: '#374151', fontSize: 16 }}>Completing sign-in…</p>
      </div>
    </div>
  );
}
