import React, { useState, useEffect, useCallback, useRef } from 'react';
import CitizenPanel from './components/CitizenPanel';
import StaffPanel from './components/StaffPanel';
import VocabularyDrawer from './components/VocabularyDrawer';
import { healthCheck, getVocabulary } from './lib/api';

function generateSessionId() {
  return 'sb-' + Math.random().toString(36).substring(2, 10);
}

export default function App() {
  const [sessionId] = useState(generateSessionId);
  const [health, setHealth] = useState(null);
  const [vocab, setVocab] = useState(null);
  const [showVocab, setShowVocab] = useState(false);
  const [messages, setMessages] = useState([]);
  const [signCaptions, setSignCaptions] = useState(null);
  const [clock, setClock] = useState('');
  const [highContrast, setHighContrast] = useState(false);
  const [largeText, setLargeText] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);

  // Clock
  useEffect(() => {
    const tick = () => {
      const now = new Date();
      setClock(now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // Health check + warm up
  useEffect(() => {
    const check = async () => {
      try {
        const h = await healthCheck();
        setHealth(h);
        setBackendOnline(true);
      } catch {
        setBackendOnline(false);
      }
    };
    check();
    const id = setInterval(check, 30000);
    return () => clearInterval(id);
  }, []);

  // Load vocabulary
  useEffect(() => {
    const load = async () => {
      try {
        const v = await getVocabulary();
        setVocab(v);
      } catch (err) {
        console.error('Failed to load vocabulary:', err);
      }
    };
    if (backendOnline) load();
  }, [backendOnline]);

  // Accessibility toggles
  useEffect(() => {
    document.body.classList.toggle('high-contrast', highContrast);
  }, [highContrast]);
  
  useEffect(() => {
    document.body.classList.toggle('large-text', largeText);
  }, [largeText]);

  const addMessage = useCallback((msg) => {
    setMessages(prev => [...prev, { ...msg, timestamp: Date.now() / 1000 }]);
  }, []);

  return (
    <>
      {/* Header */}
      <header className="header" role="banner">
        <div className="header-brand">
          <span className="logo">⚡ SignBridge</span>
          <span className="counter-label">Counter 04 — Public Service Desk</span>
        </div>
        <div className="header-right">
          <span className="header-clock" aria-label="Current time">{clock}</span>
          
          <span className={`status-pill ${backendOnline ? 'online' : 'offline'}`}>
            <span className="dot"></span>
            {backendOnline ? 'Connected' : 'Offline'}
          </span>
          
          {health?.models_loaded && (
            <span className={`status-pill ${health.models_loaded.static ? 'online' : 'offline'}`}>
              <span className="dot"></span>
              Model {health.models_loaded.static ? 'Ready' : 'Loading'}
            </span>
          )}

          <div className="a11y-controls">
            <button 
              className={`a11y-btn ${highContrast ? 'active' : ''}`}
              onClick={() => setHighContrast(v => !v)}
              aria-label="Toggle high contrast mode"
              title="High Contrast"
            >◐</button>
            <button 
              className={`a11y-btn ${largeText ? 'active' : ''}`}
              onClick={() => setLargeText(v => !v)}
              aria-label="Toggle large text"
              title="Large Text"
            >A+</button>
            <button 
              className="a11y-btn"
              onClick={() => setShowVocab(true)}
              aria-label="Show vocabulary"
              title="Vocabulary"
            >📖</button>
          </div>
        </div>
      </header>

      {/* Main two-panel layout */}
      <main className="main-content" role="main">
        <CitizenPanel 
          sessionId={sessionId}
          messages={messages}
          signCaptions={signCaptions}
          addMessage={addMessage}
          backendOnline={backendOnline}
        />
        <StaffPanel 
          sessionId={sessionId}
          messages={messages}
          addMessage={addMessage}
          setSignCaptions={setSignCaptions}
          vocab={vocab}
          backendOnline={backendOnline}
        />
      </main>

      {/* Footer */}
      <footer className="footer" role="contentinfo">
        <div className="footer-metrics">
          <span className="footer-metric">
            Models: <span className="value">
              {health ? `Static ${health.models_loaded?.static ? '✓' : '✗'} | Temporal ${health.models_loaded?.temporal ? '✓' : '✗'}` : '—'}
            </span>
          </span>
          <span className="footer-metric">
            Vocabulary: <span className="value">
              {vocab ? `${Object.keys(vocab.recognition?.digits?.classes || {}).length + Object.keys(vocab.recognition?.words?.classes || {}).length} signs` : '—'}
            </span>
          </span>
          {health?.uptime_s && (
            <span className="footer-metric">
              Uptime: <span className="value">{Math.round(health.uptime_s)}s</span>
            </span>
          )}
        </div>
        <div className="honesty-banner" role="alert" aria-live="polite">
          ⚠ Supported vocabulary is limited to the trained sign classes. This system does not perform full sign-language translation.
        </div>
      </footer>

      {/* Vocabulary Drawer */}
      {showVocab && (
        <VocabularyDrawer 
          vocab={vocab} 
          onClose={() => setShowVocab(false)} 
        />
      )}
      
      {/* Screen reader announcements */}
      <div className="sr-only" aria-live="polite" id="sr-announce"></div>
    </>
  );
}
