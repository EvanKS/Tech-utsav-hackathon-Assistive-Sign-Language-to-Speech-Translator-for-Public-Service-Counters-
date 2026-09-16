import React, { useState, useEffect, useCallback } from 'react';
import { useHandLandmarks } from '../hooks/useHandLandmarks';
import { useRecognition } from '../hooks/useRecognition';
import { useSpeech } from '../hooks/useSpeech';
import SignCaptionPlayer from './SignCaptionPlayer';
import ConversationLog from './ConversationLog';
import SpellingTray from './SpellingTray';
import { getSignGuide } from '../data/signGuides';

const WORD_CLASSES = ['hello', 'please', 'help', 'water', 'yes', 'no', 'eat', 'go'];
const DIGIT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
const LETTER_CLASSES = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
const GESTURE_ASL_CLASSES = ['HELLO', 'NO', 'SORRY', 'THANKYOU', 'YES'];

export default function CitizenPanel({ sessionId, messages, signCaptions, addMessage, backendOnline }) {
  const { videoRef, canvasRef, isActive, startCamera, stopCamera, landmarks, hasHand, fps } = useHandLandmarks();
  const { prediction, status, latency, bufferFill, sendLandmarks, reset, clearBuffer } = useRecognition(sessionId);
  const { speak, ttsSupported } = useSpeech();

  // Mode: 'temporal' (ISL Words), 'gesture_asl' (ASL Phrases), 'letters' (ASL A-Z), 'static' (ASL Digits)
  const [mode, setMode] = useState('temporal');
  const [isLargeCamera, setIsLargeCamera] = useState(false);
  const [tokenTray, setTokenTray] = useState([]);
  const [spellingBuffer, setSpellingBuffer] = useState([]);
  const [selectedGuideSign, setSelectedGuideSign] = useState(null);

  // Send landmarks to backend according to selected mode
  useEffect(() => {
    if (isActive && backendOnline) {
      sendLandmarks(landmarks, mode);
    }
  }, [landmarks, isActive, backendOnline, mode, sendLandmarks]);

  // When switching modes, clear buffer and reset prediction
  const handleModeChange = (newMode) => {
    setMode(newMode);
    clearBuffer();
    reset();
  };

  // Auto-add emitted stabilized tokens:
  // If in letters mode -> route to spelling buffer
  // Otherwise -> route to token tray
  useEffect(() => {
    if (prediction?.stable?.emitted && prediction.stable.label) {
      const label = prediction.stable.label;
      const lang = prediction.language || (mode === 'temporal' ? 'isl' : 'asl');

      if (mode === 'letters') {
        setSpellingBuffer(prev => [...prev, label]);
      } else {
        setTokenTray(prev => [...prev, {
          label,
          confidence: prediction.stable.confidence,
          language: lang
        }]);
      }

      const announcer = document.getElementById('sr-announce');
      if (announcer) announcer.textContent = `Recognized [${lang.toUpperCase()}]: ${label}`;
    }
  }, [prediction?.stable?.emitted, prediction?.stable?.label, prediction?.language, mode]);

  // Fingerspelling handlers
  const handleSpellingBackspace = useCallback(() => {
    setSpellingBuffer(prev => prev.slice(0, -1));
  }, []);

  const handleSpellingClear = useCallback(() => {
    setSpellingBuffer([]);
  }, []);

  const handleSpellingAddSpace = useCallback(() => {
    setSpellingBuffer(prev => [...prev, ' ']);
  }, []);

  const handleCommitWord = useCallback(() => {
    const word = spellingBuffer.join('').trim();
    if (!word) return;
    setTokenTray(prev => [...prev, { label: word, confidence: 0.95, language: 'asl' }]);
    setSpellingBuffer([]);
  }, [spellingBuffer]);

  const handleSendSpelledWord = useCallback(() => {
    const word = spellingBuffer.join('').trim();
    if (!word) return;
    addMessage({ sender: 'citizen', text: word, confidence: 0.95 });
    setSpellingBuffer([]);
  }, [spellingBuffer, addMessage]);

  // Speak staff messages to citizen
  useEffect(() => {
    const staffMsgs = messages.filter(m => m.sender === 'staff');
    if (staffMsgs.length > 0) {
      const latest = staffMsgs[staffMsgs.length - 1];
      if (ttsSupported && Date.now() / 1000 - latest.timestamp < 2) {
        speak(latest.text);
      }
    }
  }, [messages, speak, ttsSupported]);

  const sendMessage = useCallback(() => {
    if (tokenTray.length === 0) return;
    const text = tokenTray.map(t => t.label).join(' ');
    const avgConf = tokenTray.reduce((s, t) => s + (t.confidence || 0), 0) / tokenTray.length;
    addMessage({ sender: 'citizen', text, confidence: avgConf });
    setTokenTray([]);
  }, [tokenTray, addMessage]);

  const clearTray = useCallback(() => setTokenTray([]), []);

  const removeToken = useCallback((idx) => {
    setTokenTray(prev => prev.filter((_, i) => i !== idx));
  }, []);

  const rawLabel = prediction?.raw?.label || '—';
  const rawConf = prediction?.raw?.confidence || 0;
  const confPercent = Math.round(rawConf * 100);
  const confLevel = rawConf >= 0.7 ? 'high' : rawConf >= 0.4 ? 'medium' : 'low';
  const isRecognized = prediction?.status === 'recognized' || prediction?.stable?.emitted;

  // Debug stabilizer
  const debug = prediction?.debug;
  const modalLabel = debug?.modal_label || '';
  const agreement = debug?.agreement || 0;
  const topK = prediction?.raw?.top_k || [];

  // Determine active visual guide signs and metadata
  let activeSigns = WORD_CLASSES;
  let activeGuideLang = 'ISL';
  let activeGuideTitle = 'Words (ISL isolated video dataset)';
  let isTemporalMode = mode === 'temporal' || mode === 'gesture_asl';

  if (mode === 'gesture_asl') {
    activeSigns = GESTURE_ASL_CLASSES;
    activeGuideLang = 'ASL';
    activeGuideTitle = 'Dynamic Phrases (ASL video sequences)';
  } else if (mode === 'letters') {
    activeSigns = LETTER_CLASSES;
    activeGuideLang = 'ASL';
    activeGuideTitle = 'Fingerspelling Letters (ASL static A–Z)';
  } else if (mode === 'static' || mode === 'digits') {
    activeSigns = DIGIT_CLASSES;
    activeGuideLang = 'ASL';
    activeGuideTitle = 'Numeric Digits (ASL static 0–9)';
  }

  const getSignImgUrl = (sign) => {
    const s = sign.toLowerCase();
    if (mode === 'gesture_asl') {
      return `/signs/asl_${s}.png`;
    }
    return `/signs/${s}.png`;
  };

  return (
    <section className={`panel ${isLargeCamera ? 'panel-cinema' : ''}`} aria-label="Citizen Side">
      {/* Panel Header */}
      <div className="panel-header">
        <div>
          <div className="panel-title">
            <span className="icon">👤</span>
            Citizen / नागरिक
          </div>
          <div className="panel-subtitle">Sign gesture communication desk</div>
        </div>

        <div className="btn-group" style={{ alignItems: 'center' }}>
          {/* Camera size toggle */}
          <button
            className={`btn btn-sm ${isLargeCamera ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setIsLargeCamera(!isLargeCamera)}
            title={isLargeCamera ? 'Switch to normal camera' : 'Enlarge camera view'}
          >
            {isLargeCamera ? '🗗 Normal View' : '⛶ Enlarge Camera'}
          </button>

          {!isActive ? (
            <button className="btn btn-primary btn-sm" onClick={startCamera} id="btn-start-camera">
              📹 Start Camera
            </button>
          ) : (
            <button className="btn btn-danger btn-sm" onClick={() => { stopCamera(); reset(); }} id="btn-stop-camera">
              ⏹ Stop
            </button>
          )}
        </div>
      </div>

      {/* Four-Way Dataset & Mode Selector Tabs with Strict Language Attribution */}
      <div className="mode-selector-bar">
        <div className="mode-tabs">
          <button
            className={`mode-tab ${mode === 'temporal' ? 'active' : ''}`}
            onClick={() => handleModeChange('temporal')}
            id="tab-words"
          >
            <span className="mode-tab-icon">🔤</span>
            <div>
              <div className="mode-tab-title">
                Words <span className="lang-badge lang-badge-isl">ISL</span>
              </div>
              <div className="mode-tab-sub">8 Isolated ISL Words (Video)</div>
            </div>
          </button>

          <button
            className={`mode-tab ${mode === 'gesture_asl' ? 'active' : ''}`}
            onClick={() => handleModeChange('gesture_asl')}
            id="tab-gestures-asl"
          >
            <span className="mode-tab-icon">👋</span>
            <div>
              <div className="mode-tab-title">
                Gestures <span className="lang-badge lang-badge-asl">ASL</span>
              </div>
              <div className="mode-tab-sub">5 Phrases (Hello, Yes, No...)</div>
            </div>
          </button>

          <button
            className={`mode-tab ${mode === 'letters' ? 'active' : ''}`}
            onClick={() => handleModeChange('letters')}
            id="tab-letters"
          >
            <span className="mode-tab-icon">🔠</span>
            <div>
              <div className="mode-tab-title">
                Letters <span className="lang-badge lang-badge-asl">ASL</span>
              </div>
              <div className="mode-tab-sub">A–Z Fingerspelling (Static)</div>
            </div>
          </button>

          <button
            className={`mode-tab ${mode === 'static' ? 'active' : ''}`}
            onClick={() => handleModeChange('static')}
            id="tab-digits"
          >
            <span className="mode-tab-icon">🔢</span>
            <div>
              <div className="mode-tab-title">
                Digits <span className="lang-badge lang-badge-asl">ASL</span>
              </div>
              <div className="mode-tab-sub">0–9 Numbers (Static)</div>
            </div>
          </button>
        </div>
      </div>

      <div className="panel-body">
        {/* Webcam Container */}
        <div
          className={`webcam-container ${isLargeCamera ? 'webcam-large' : 'webcam-normal'} ${hasHand ? 'hand-present' : ''}`}
          role="img"
          aria-label="Webcam feed with hand tracking"
        >
          <video ref={videoRef} playsInline muted style={{ display: isActive ? 'block' : 'none' }} />
          <canvas ref={canvasRef} style={{ display: isActive ? 'block' : 'none' }} />

          {!isActive ? (
            <div className="webcam-placeholder">
              <div className="icon" style={{ fontSize: 64 }}>📹</div>
              <span style={{ fontSize: 16, fontWeight: 600 }}>Click "Start Camera" to begin</span>
              <span style={{ fontSize: 12, color: 'var(--slate-400)' }}>
                Position your hand clearly in front of the camera
              </span>
            </div>
          ) : (
            <>
              {/* Overlay HUD Badges with Language Tag */}
              <div className="camera-hud-top">
                <div className={`hud-badge ${hasHand ? 'hud-tracking' : 'hud-searching'}`}>
                  <span className="dot"></span>
                  {hasHand ? 'HAND IN FRAME' : 'SEARCHING FOR HAND'}
                </div>
                <div className="hud-badge hud-mode">
                  MODE: {mode.toUpperCase()} • <span className={`lang-badge lang-badge-${activeGuideLang.toLowerCase()}`}>{activeGuideLang}</span>
                </div>
              </div>

              {/* Bottom HUD: FPS, Latency, Sequence Buffer */}
              <div className="camera-hud-bottom">
                <div className="hud-stats">
                  <span>{fps} FPS</span>
                  <span>•</span>
                  <span>{latency}ms latency</span>
                </div>

                {isTemporalMode && (
                  <div className="hud-buffer" title="32-frame trajectory buffer for dynamic gesture recognition">
                    <span>Sequence Buffer:</span>
                    <div className="hud-buffer-bar">
                      <div className="hud-buffer-fill" style={{ width: `${bufferFill}%` }} />
                    </div>
                    <span>{bufferFill}%</span>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* Live Recognition & Feedback HUD Card */}
        <div className={`recognition-hud ${isRecognized ? 'recognized' : ''}`}>
          <div className="hud-main-display">
            <div className="hud-sign-symbol">
              {hasHand && rawLabel !== '—' ? (
                <span className="symbol-active">{rawLabel.toUpperCase()}</span>
              ) : (
                <span className="symbol-idle">—</span>
              )}
            </div>

            <div className="hud-sign-info">
              <div className="hud-sign-label">
                {hasHand ? (
                  <>
                    <span className="label-caption">Currently Reading:</span>
                    <strong className="label-value">{rawLabel.toUpperCase()}</strong>
                    <span className={`lang-badge lang-badge-${activeGuideLang.toLowerCase()}`} style={{ marginLeft: 8 }}>
                      {activeGuideLang}
                    </span>
                  </>
                ) : (
                  <span className="label-waiting">Hold or move your hand in front of the camera</span>
                )}
              </div>

              {/* Status and message */}
              <div className="hud-status-line">
                <span className={`status-indicator ${confLevel}`}>
                  {prediction?.message || (hasHand ? 'Analyzing gesture...' : 'Waiting for hand')}
                </span>
                {debug?.modal_label && debug.modal_label !== '__UNKNOWN__' && (
                  <span className="stabilizer-pill">
                    Stabilizer consensus: <strong>{debug.modal_label}</strong> ({Math.round(agreement * 100)}%)
                  </span>
                )}
              </div>

              {/* Live Confidence Bar */}
              {hasHand && (
                <div className="confidence-meter">
                  <div className="confidence-meter-header">
                    <span>Confidence</span>
                    <span>{confPercent}%</span>
                  </div>
                  <div className="confidence-track">
                    <div className={`confidence-progress ${confLevel}`} style={{ width: `${confPercent}%` }} />
                  </div>
                </div>
              )}
            </div>

            {/* Locked emission badge */}
            {prediction?.stable?.emitted && (
              <div className="emitted-flash-badge">
                ✓ Locked: {prediction.stable.label} [{activeGuideLang}]
              </div>
            )}
          </div>

          {/* Top-4 Neural Network Predictions */}
          {hasHand && topK.length > 0 && (
            <div className="topk-predictions">
              <span className="topk-title">Model Probabilities:</span>
              <div className="topk-chips">
                {topK.slice(0, 4).map((p, idx) => (
                  <div
                    key={idx}
                    className={`topk-chip ${p.label === rawLabel ? 'topk-chip-lead' : ''}`}
                  >
                    <span className="topk-name">{p.label}</span>
                    <span className="topk-val">{Math.round(p.prob * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Dataset Sign Reference Guide & Exemplar Strip */}
        <div className="dataset-guide-section">
          <div className="guide-header">
            <span className="guide-title">
              Visual Guide — {activeGuideTitle}
            </span>
            <span className="guide-hint">💡 Click any sign for finger-by-finger formation guide & live practice check:</span>
          </div>
          <div className="guide-strip" role="list">
            {activeSigns.map((s) => {
              const isCurrent = rawLabel.toLowerCase() === s.toLowerCase();
              return (
                <div
                  key={s}
                  className={`guide-card ${isCurrent ? 'guide-card-active' : ''}`}
                  onClick={() => setSelectedGuideSign(s)}
                  role="listitem"
                  title={`Click to view finger-by-finger guide for "${s.toUpperCase()}"`}
                >
                  <img
                    src={getSignImgUrl(s)}
                    alt={`Sign for ${s}`}
                    className="guide-card-img"
                    onError={(e) => { e.target.style.display = 'none'; }}
                  />
                  <span className="guide-card-label">{s.toUpperCase()}</span>
                  {isCurrent && <span className="guide-live-tag">LIVE</span>}
                </div>
              );
            })}
          </div>
        </div>

        {/* Fingerspelling Tray (Active in Letters mode) */}
        {mode === 'letters' && (
          <SpellingTray
            buffer={spellingBuffer}
            onBackspace={handleSpellingBackspace}
            onClear={handleSpellingClear}
            onAddSpace={handleSpellingAddSpace}
            onCommitWord={handleCommitWord}
            onSendDirect={handleSendSpelledWord}
          />
        )}

        {/* Sign Caption Player (for staff messages) */}
        {signCaptions && <SignCaptionPlayer captions={signCaptions} />}

        {/* Message Tray */}
        <div className="message-tray">
          <label style={{ fontSize: 12, color: 'var(--slate-400)', marginBottom: 4, display: 'block' }}>
            Message to Staff (Composed from Recognized Signs)
          </label>
          <div className="token-tray" role="list" aria-label="Composed message tokens">
            {tokenTray.length === 0 && (
              <span style={{ color: 'var(--slate-500)', fontSize: 13 }}>
                Recognized signs will appear here automatically...
              </span>
            )}
            {tokenTray.map((t, i) => (
              <span key={i} className="token-chip" role="listitem">
                {t.label}
                {t.language && (
                  <span className={`lang-badge lang-badge-${t.language.toLowerCase()}`} style={{ marginLeft: 4 }}>
                    {t.language.toUpperCase()}
                  </span>
                )}
                <span className="remove" onClick={() => removeToken(i)} title="Remove">×</span>
              </span>
            ))}
          </div>
          <div className="btn-group">
            <button className="btn btn-primary btn-sm" onClick={sendMessage} disabled={tokenTray.length === 0} id="btn-send-citizen">
              📤 Send to Staff
            </button>
            <button className="btn btn-secondary btn-sm" onClick={clearTray} disabled={tokenTray.length === 0}>
              Clear
            </button>
          </div>
        </div>

        {/* Conversation log (citizen view) */}
        <ConversationLog messages={messages} perspective="citizen" />
      </div>

      {/* Interactive Sign Guide & Practice Matcher Modal */}
      {selectedGuideSign && (() => {
        const guide = getSignGuide(selectedGuideSign);
        const isMatching = hasHand && rawLabel.toLowerCase() === selectedGuideSign.toLowerCase();

        return (
          <div className="modal-backdrop" onClick={() => setSelectedGuideSign(null)}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>
                  <span>{guide?.label || selectedGuideSign.toUpperCase()}</span>
                  <span className={`lang-badge lang-badge-${activeGuideLang.toLowerCase()}`}>{activeGuideLang}</span>
                </h3>
                <button className="btn btn-secondary btn-sm" onClick={() => setSelectedGuideSign(null)}>✕</button>
              </div>

              <div className="modal-body">
                {/* Visual Card + Summary */}
                <div className="guide-modal-visual">
                  <img
                    src={getSignImgUrl(selectedGuideSign)}
                    alt={selectedGuideSign}
                    className="guide-modal-img"
                  />
                  <div className="guide-modal-quickinfo">
                    <div className="guide-modal-summary">
                      {guide?.summary || 'Perform this sign clearly in front of the camera.'}
                    </div>
                    <div className="guide-modal-meta">
                      <span className="hud-badge hud-tracking" style={{ fontSize: 11 }}>
                        {guide?.modality || (isTemporalMode ? 'Dynamic Motion' : 'Static Pose')}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Step-by-Step Finger & Motion Breakdown */}
                <div className="guide-detail-box">
                  <div className="guide-step">
                    <span className="guide-step-icon">🖐️</span>
                    <div className="guide-step-content">
                      <strong>Hand Shape & Finger Positions:</strong>
                      <span>{guide?.handShape || 'Form the hand pose as shown in the exemplar image.'}</span>
                    </div>
                  </div>

                  <div className="guide-step">
                    <span className="guide-step-icon">🔄</span>
                    <div className="guide-step-content">
                      <strong>Movement & Gesture Dynamics:</strong>
                      <span>{guide?.movement || 'Hold steadily in front of the camera for at least 1 second.'}</span>
                    </div>
                  </div>

                  {guide?.tip && (
                    <div className="guide-step">
                      <span className="guide-step-icon">💡</span>
                      <div className="guide-step-content">
                        <strong>Camera Pro-Tip:</strong>
                        <span>{guide.tip}</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Live Practice Matcher (Interactive Feedback) */}
                <div className={`guide-practice-box ${isMatching ? 'matching' : ''}`}>
                  <div className="guide-practice-header">
                    <span>LIVE PRACTICE CHECKER</span>
                    <span>{hasHand ? `${fps} FPS • Tracking Active` : 'Show hand to camera'}</span>
                  </div>

                  {hasHand ? (
                    <div className={`guide-practice-status ${isMatching ? 'match' : 'nomatch'}`}>
                      {isMatching ? (
                        <>
                          <span style={{ fontSize: 22 }}>✅</span>
                          <div>
                            <div>PERFECT MATCH! ({confPercent}% Confidence)</div>
                            <div style={{ fontSize: 12, fontWeight: 400, color: 'var(--slate-300)' }}>
                              Your hand pose accurately matches {selectedGuideSign.toUpperCase()}! Hold steady to emit.
                            </div>
                          </div>
                        </>
                      ) : (
                        <>
                          <span style={{ fontSize: 22 }}>🔍</span>
                          <div>
                            <div>Currently Reading: <strong>{rawLabel.toUpperCase()}</strong> ({confPercent}%)</div>
                            <div style={{ fontSize: 12, fontWeight: 400, color: 'var(--slate-400)' }}>
                              Adjust fingers to match the breakdown above.
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="guide-practice-status nomatch">
                      <span style={{ fontSize: 22 }}>📹</span>
                      <span>Start camera and position your hand in frame to practice live.</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="modal-footer">
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => {
                    setTokenTray(prev => [...prev, {
                      label: selectedGuideSign.toUpperCase(),
                      confidence: 0.98,
                      language: activeGuideLang.toLowerCase()
                    }]);
                    setSelectedGuideSign(null);
                  }}
                >
                  ➕ Insert "{selectedGuideSign.toUpperCase()}" into Message
                </button>
                <button className="btn btn-secondary btn-sm" onClick={() => setSelectedGuideSign(null)}>
                  Close Guide
                </button>
              </div>
            </div>
          </div>
        );
      })()}
    </section>
  );
}
