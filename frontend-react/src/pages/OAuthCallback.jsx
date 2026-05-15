import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function OAuthCallback() {
  const navigate = useNavigate();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);

    const token = params.get('token');
    const email = params.get('email');
    const plan = params.get('plan');
    const name = params.get('name');

    if (token) {
      localStorage.setItem('su_token', token);
      localStorage.setItem('su_email', email || '');
      localStorage.setItem('su_plan', plan || 'free');
      localStorage.setItem('su_name', name || '');

      navigate('/', { replace: true });
    } else {
      navigate('/login', { replace: true });
    }
  }, [navigate]);

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-lg font-medium">
        Logging you in with Google...
      </div>
    </div>
  );
}
