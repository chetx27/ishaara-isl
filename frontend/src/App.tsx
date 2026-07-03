import React, { useState } from 'react';
import { LiveDemo } from './components/LiveDemo';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState<'sign-to-text' | 'text-to-sign'>('sign-to-text');

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo-section">
          <h1>ISHAARA</h1>
          <span className="version-badge">v1.0.0</span>
        </div>
        <nav className="mode-toggle">
          <button 
            className={activeTab === 'sign-to-text' ? 'active' : ''} 
            onClick={() => setActiveTab('sign-to-text')}
          >
            SIGN → TEXT
          </button>
          <button 
            className={activeTab === 'text-to-sign' ? 'active' : ''} 
            onClick={() => setActiveTab('text-to-sign')}
          >
            TEXT → SIGN
          </button>
        </nav>
      </header>

      <main className="app-main">
        {activeTab === 'sign-to-text' ? (
          <section className="mode-section">
            <div className="section-header">
              <h2>REAL-TIME RECOGNITION</h2>
              <p>Translating isolated ISL signs to English in real-time.</p>
            </div>
            <LiveDemo />
          </section>
        ) : (
          <section className="mode-section">
            <div className="section-header">
              <h2>GENERATION (LOOKUP & STITCH)</h2>
              <p>Type English text to synthesize an ISL video response.</p>
            </div>
            <div className="card text-to-sign-container">
              <input type="text" placeholder="TYPE SENTENCE HERE..." />
              <button>GENERATE VIDEO</button>
              
              <div className="video-placeholder">
                <span className="empty-state">VIDEO WILL APPEAR HERE</span>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
