import React, { useState, useEffect, useRef } from 'react';

export default function SignCaptionPlayer({ captions }) {
  const [activeIdx, setActiveIdx] = useState(-1);
  const intervalRef = useRef(null);

  // Auto-play through tokens
  useEffect(() => {
    if (!captions?.tokens?.length) return;
    
    setActiveIdx(0);
    let idx = 0;
    
    intervalRef.current = setInterval(() => {
      idx++;
      if (idx >= captions.tokens.length) {
        clearInterval(intervalRef.current);
        setActiveIdx(-1);
        return;
      }
      setActiveIdx(idx);
    }, 1500);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [captions]);

  const replay = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setActiveIdx(0);
    let idx = 0;
    intervalRef.current = setInterval(() => {
      idx++;
      if (idx >= captions.tokens.length) {
        clearInterval(intervalRef.current);
        setActiveIdx(-1);
        return;
      }
      setActiveIdx(idx);
    }, 1500);
  };

  if (!captions) return null;

  const originalText = captions.tokens?.map(t => t.token).join(' ') || '';

  return (
    <div className="caption-player" aria-label="Sign caption display">
      <div className="caption-sentence">{originalText}</div>
      
      <div className="caption-tokens" role="list">
        {captions.tokens?.map((token, idx) => (
          <div
            key={idx}
            className={`caption-token ${token.available ? 'available' : 'unavailable'} ${activeIdx === idx ? 'active' : ''}`}
            role="listitem"
          >
            {token.available && token.asset ? (
              <img
                src={token.asset}
                alt={`Sign for ${token.token}`}
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            ) : (
              <div style={{
                width: 56, height: 56, display: 'flex', alignItems: 'center',
                justifyContent: 'center', fontSize: 10, color: 'var(--slate-500)',
                textAlign: 'center', lineHeight: 1.2
              }}>
                {token.available ? token.token : 'No sign\navailable'}
              </div>
            )}
            <span className="caption-token-label">{token.token}</span>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="caption-coverage">
          Coverage: {Math.round((captions.coverage || 0) * 100)}% — {captions.notice}
        </span>
        <button className="btn btn-secondary btn-sm" onClick={replay} style={{ flexShrink: 0 }}>
          🔄 Replay
        </button>
      </div>
    </div>
  );
}
