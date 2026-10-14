import { useState, useEffect } from 'react';
import { Mode2LiveStudio } from './components/Mode2LiveStudio';
import {
  BrainCircuit,
  Bot,
} from 'lucide-react';

const API_BASE = '';

export function App() {
  const [backendHealth, setBackendHealth] = useState<any>(null);

  useEffect(() => {
    let timeoutId: any;

    const checkHealth = () => {
      fetch(`${API_BASE}/api/health`)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data) => {
          setBackendHealth(data);
          timeoutId = setTimeout(checkHealth, 10000);
        })
        .catch(() => {
          setBackendHealth({ status: 'offline' });
          timeoutId = setTimeout(checkHealth, 2000);
        });
    };

    checkHealth();
    return () => clearTimeout(timeoutId);
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)' }}>
      {/* Top Glass Navbar */}
      <header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 50,
          background: 'rgba(11, 15, 25, 0.85)',
          backdropFilter: 'blur(16px)',
          borderBottom: '1px solid var(--border-subtle)',
          padding: '0.75rem 2rem',
        }}
      >
        <div style={{ maxWidth: '1650px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          {/* Logo & Title */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div
              style={{
                width: 38,
                height: 38,
                borderRadius: '0.6rem',
                background: 'linear-gradient(135deg, #6366f1 0%, #3b82f6 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 16px rgba(99, 102, 241, 0.4)',
              }}
            >
              <BrainCircuit size={22} color="#ffffff" />
            </div>
            <div>
              <h1 style={{ fontSize: '1.05rem', fontWeight: 800, letterSpacing: '-0.02em', color: '#ffffff' }}>
                Predictive Processing Demonstrator
              </h1>
              <p style={{ fontSize: '0.7rem', color: '#9ca3af' }}>
                Multimodal Bayesian Affect Perception & Live HRI Studio (A + V → AV → PWP(C, AV))
              </p>
            </div>
          </div>

          {/* Status Indicators */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {/* Backend API Pill */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.75rem',
                padding: '0.25rem 0.65rem',
                borderRadius: '9999px',
                background:
                  backendHealth?.status === 'online'
                    ? 'rgba(16, 185, 129, 0.12)'
                    : 'rgba(244, 63, 94, 0.12)',
                color: backendHealth?.status === 'online' ? '#34d399' : '#fb7185',
                border: `1px solid ${
                  backendHealth?.status === 'online'
                    ? 'rgba(16, 185, 129, 0.3)'
                    : 'rgba(244, 63, 94, 0.3)'
                }`,
              }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: backendHealth?.status === 'online' ? '#10b981' : '#f43f5e',
                }}
              />
              FastAPI: {backendHealth?.status === 'online' ? 'Connected' : 'Offline'}
            </div>

            {/* pyCub Status */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.75rem',
                padding: '0.25rem 0.65rem',
                borderRadius: '9999px',
                background: backendHealth?.pycub_connected
                  ? 'rgba(99, 102, 241, 0.15)'
                  : 'rgba(255, 255, 255, 0.05)',
                color: backendHealth?.pycub_connected ? '#a5b4fc' : '#9ca3af',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <Bot size={13} />
              iCub Sim: {backendHealth?.pycub_connected ? 'Active' : 'Standby'}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Viewport */}
      <main style={{ flex: 1, maxWidth: '1650px', width: '100%', margin: '0 auto', padding: '1rem 1.5rem 2rem' }}>
        <Mode2LiveStudio apiBase={API_BASE} />
      </main>
    </div>
  );
}

export default App;
