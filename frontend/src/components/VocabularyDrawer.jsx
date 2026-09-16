import React from 'react';

export default function VocabularyDrawer({ vocab, onClose }) {
  const digitClasses = vocab?.recognition?.digits?.classes || {};
  const wordClasses = vocab?.recognition?.words?.classes || {};
  const production = vocab?.production || {};

  return (
    <>
      <div className="vocab-drawer-overlay" onClick={onClose} />
      <div className="vocab-drawer" role="dialog" aria-label="Vocabulary Registry">
        <div className="vocab-drawer-header">
          <h3>📖 Vocabulary Registry</h3>
          <button className="btn btn-secondary btn-sm" onClick={onClose}>✕ Close</button>
        </div>
        <div className="vocab-drawer-body">
          {/* Recognition - Digits */}
          <div className="vocab-section">
            <div className="vocab-section-title">Recognition — Digits (0–9)</div>
            <p style={{ fontSize: 12, color: 'var(--slate-400)', marginBottom: 10 }}>
              Source: ardamavi/Sign-Language-Digits-Dataset | Model: Landmark MLP
            </p>
            <div className="vocab-grid">
              {Object.entries(digitClasses).map(([key, val]) => (
                <div key={key} className="vocab-item">
                  {production[key]?.caption_asset && (
                    <img src={production[key].caption_asset} alt={`Sign for ${key}`} 
                         onError={(e) => { e.target.style.display = 'none'; }} />
                  )}
                  <span className="vocab-item-label">{val.label}</span>
                  <span style={{
                    fontSize: 10,
                    color: val.recognition_supported ? 'var(--success)' : 'var(--danger)'
                  }}>
                    {val.recognition_supported ? '✓ Supported' : '✗ Unsupported'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Recognition - Words */}
          <div className="vocab-section">
            <div className="vocab-section-title">Recognition — ISL Words</div>
            <p style={{ fontSize: 12, color: 'var(--slate-400)', marginBottom: 10 }}>
              Source: vidit031/isl-isolated-8words | Model: GRU Temporal
            </p>
            <div className="vocab-grid">
              {Object.entries(wordClasses).map(([key, val]) => (
                <div key={key} className="vocab-item">
                  {production[key]?.caption_asset && (
                    <img src={production[key].caption_asset} alt={`Sign for ${key}`}
                         onError={(e) => { e.target.style.display = 'none'; }} />
                  )}
                  <span className="vocab-item-label">{val.label}</span>
                  <span style={{
                    fontSize: 10,
                    color: val.recognition_supported ? 'var(--success)' : 'var(--danger)'
                  }}>
                    {val.recognition_supported ? '✓ Supported' : '✗ Unsupported'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Production Coverage */}
          <div className="vocab-section">
            <div className="vocab-section-title">Caption Assets Available</div>
            <div style={{ fontSize: 13, color: 'var(--slate-300)' }}>
              {Object.entries(production).map(([key, val]) => (
                <div key={key} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid var(--navy-600)' }}>
                  <span style={{ textTransform: 'capitalize' }}>{key}</span>
                  <span style={{ color: val.available ? 'var(--success)' : 'var(--danger)', fontSize: 12 }}>
                    {val.available ? '✓ Available' : '✗ Not available'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 16, padding: 12, background: 'var(--warning-bg)', borderRadius: 'var(--radius-sm)', fontSize: 12, color: 'var(--amber-400)' }}>
            ⚠ This vocabulary represents the complete scope of the system. Any word not listed here cannot be recognized or captioned.
          </div>
        </div>
      </div>
    </>
  );
}
