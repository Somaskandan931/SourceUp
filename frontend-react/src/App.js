// App.js — Root component
import { useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import { GLOBAL_CSS, font, C } from './styles/tokens';
import { post } from './utils/api';

import Navbar              from './components/Navbar';
import ChatButton          from './components/ChatButton';
import OnboardingWizard    from './components/OnboardingWizard';
import AuthModal           from './components/AuthModal';
import BillingModal        from './components/BillingModal';
import HomePage            from './components/HomePage';
import SearchResultsView   from './components/SearchResultsView';
import ChatTab             from './components/ChatTab';
import OAuthCallback       from './pages/OAuthCallback';

function SourceUpApp() {
  const [view,        setView]    = useState('home');
  const [searchQ,     setSearchQ] = useState('');
  const [showOnboard, setOnboard] = useState(!localStorage.getItem('su_onboarded'));
  const [showAuth,    setAuth]    = useState(window.location.pathname === '/login');
  const [showBilling, setBilling] = useState(false);
  const [user,        setUser]    = useState(
    localStorage.getItem('su_email')
      ? { email: localStorage.getItem('su_email'), plan: localStorage.getItem('su_plan') }
      : null
  );

  const handleAuthSuccess = (data) => {
    setUser({ email: data.email, plan: data.plan });
    setAuth(false);
  };

  const logout = () => {
    ['su_token','su_email','su_plan','su_company'].forEach(k => localStorage.removeItem(k));
    setUser(null);
    setView('home');
  };

  const startSearch = (q) => { setSearchQ(q); setView('search'); };

  const handleDemoLogin = async () => {
    try {
      const data = await post('/auth/demo-login', {});
      if (data.access_token) {
        localStorage.setItem('su_token', data.access_token);
        localStorage.setItem('su_email', data.email);
        localStorage.setItem('su_plan', data.plan);
        setUser({ email: data.email, plan: data.plan });
      }
    } catch {
      localStorage.setItem('su_token', 'demo_token_' + Date.now());
      localStorage.setItem('su_email', 'demo@sourceup.com');
      localStorage.setItem('su_plan', 'pro');
      setUser({ email: 'demo@sourceup.com', plan: 'pro' });
    }
  };

  return (
    <div style={{ fontFamily: font, background: C.bg, minHeight: '100vh', color: C.text }}>
      <style>{GLOBAL_CSS}</style>

      {showOnboard && (
        <OnboardingWizard onComplete={() => { localStorage.setItem('su_onboarded', '1'); setOnboard(false); }} />
      )}
      {showAuth    && <AuthModal    onClose={() => setAuth(false)}    onSuccess={handleAuthSuccess} />}
      {showBilling && <BillingModal onClose={() => setBilling(false)} currentPlan={user?.plan || 'free'} />}

      <Navbar
        view={view}
        setView={setView}
        user={user}
        onOpenAuth={() => setAuth(true)}
        onOpenBilling={() => setBilling(true)}
        onDemoLogin={handleDemoLogin}
        onLogout={logout}
      />

      <ChatButton onClick={() => setView('chat')} />

      {view === 'home'   && <HomePage          onSearch={startSearch} onOpenChat={() => setView('chat')} onOpenBilling={() => setBilling(true)} onOpenAuth={() => setAuth(true)} user={user} />}
      {view === 'search' && <SearchResultsView key={searchQ} initialQuery={searchQ} onBack={() => setView('home')} />}
      {view === 'chat'   && <ChatTab />}
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/oauth-callback" element={<OAuthCallback />} />
      <Route path="*" element={<SourceUpApp />} />
    </Routes>
  );
}
