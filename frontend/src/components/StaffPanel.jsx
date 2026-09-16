import React, { useState, useCallback } from 'react';
import { useSpeech } from '../hooks/useSpeech';
import { textToSign, postMessage as apiPostMessage } from '../lib/api';
import ConversationLog from './ConversationLog';

export default function StaffPanel({ sessionId, messages, addMessage, setSignCaptions, vocab, backendOnline }) {
  const [text, setText] = useState('');
  const { speak, startListening, stopListening, isListening, isSpeaking, ttsSupported, sttSupported } = useSpeech();

  const counterPhrases = vocab?.counter_phrases || [
    "Please wait for your turn.",
    "Please show your ID.",
    "Your token number is {N}.",
    "Go to counter {N}.",
    "Please sign here.",
    "Thank you. You may go."
  ];

  const sendMessage = useCallback(async () => {
    if (!text.trim()) return;
    
    const msg = text.trim();
    
    // Add to local messages immediately
    addMessage({ sender: 'staff', text: msg });
    
    // Speak aloud (primary TTS)
    if (ttsSupported) {
      speak(msg);
    }
    
    // Get sign captions from backend
    if (backendOnline) {
      try {
        const captions = await textToSign(msg);
        setSignCaptions(captions);
      } catch (err) {
        console.error('Text-to-sign error:', err);
        // Still show text even if captions fail
        setSignCaptions({ tokens: [], coverage: 0, notice: 'Caption service unavailable' });
      }
      
      // Post to backend conversation log
      try {
        await apiPostMessage(sessionId, 'staff', msg);
      } catch (err) {
        console.error('Message post error:', err);
      }
    } else {
      // Offline mode - just show text
      setSignCaptions({ tokens: [], coverage: 0, notice: 'Backend offline — text and speech only' });
    }
    
    setText('');
  }, [text, addMessage, speak, ttsSupported, backendOnline, sessionId, setSignCaptions]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  const handleQuickPhrase = useCallback((phrase) => {
    // Handle {N} placeholder
    if (phrase.includes('{N}')) {
      const num = prompt('Enter number:', '5');
      if (num !== null) {
        setText(phrase.replace('{N}', num));
      }
    } else {
      setText(phrase);
    }
  }, []);

  const handleVoiceInput = useCallback(() => {
    if (isListening) {
      stopListening();
    } else {
      startListening((transcript) => {
        setText(prev => prev ? prev + ' ' + transcript : transcript);
      });
    }
  }, [isListening, startListening, stopListening]);

  const replayAudio = useCallback(() => {
    const staffMsgs = messages.filter(m => m.sender === 'staff');
    if (staffMsgs.length > 0) {
      speak(staffMsgs[staffMsgs.length - 1].text);
    }
  }, [messages, speak]);

  // Speak citizen messages aloud for staff
  React.useEffect(() => {
    const citizenMsgs = messages.filter(m => m.sender === 'citizen');
    if (citizenMsgs.length > 0) {
      const latest = citizenMsgs[citizenMsgs.length - 1];
      if (ttsSupported && Date.now() / 1000 - latest.timestamp < 2) {
        speak(`Citizen says: ${latest.text}`);
      }
    }
  }, [messages, speak, ttsSupported]);

  return (
    <section className="panel" aria-label="Staff Side">
      <div className="panel-header">
        <div>
          <div className="panel-title">
            <span className="icon">🏛️</span>
            Staff / कर्मचारी
          </div>
          <div className="panel-subtitle">Type or speak to communicate</div>
        </div>
        <div className="btn-group">
          <button className="btn btn-secondary btn-sm" onClick={replayAudio} title="Replay last audio">
            🔊 Replay
          </button>
        </div>
      </div>

      <div className="panel-body">
        {/* Text Input */}
        <div>
          <textarea
            className="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message to the citizen... (Ctrl+Enter to send)"
            aria-label="Message to citizen"
            id="staff-text-input"
          />
          <div className="btn-group" style={{ marginTop: 8 }}>
            <button className="btn btn-primary" onClick={sendMessage} disabled={!text.trim()} id="btn-send-staff">
              📢 Send & Speak
            </button>
            {sttSupported && (
              <button
                className={`btn ${isListening ? 'btn-danger' : 'btn-secondary'}`}
                onClick={handleVoiceInput}
                id="btn-voice-input"
              >
                {isListening ? '🔴 Stop' : '🎙️ Voice Input'}
              </button>
            )}
          </div>
        </div>

        {/* Quick Phrases */}
        <div>
          <label style={{ fontSize: 12, color: 'var(--slate-400)', marginBottom: 8, display: 'block' }}>
            Quick Phrases
          </label>
          <div className="quick-phrases">
            {counterPhrases.map((phrase, i) => (
              <button
                key={i}
                className="quick-phrase-btn"
                onClick={() => handleQuickPhrase(phrase)}
                title={phrase}
              >
                {phrase.length > 30 ? phrase.substring(0, 30) + '…' : phrase}
              </button>
            ))}
          </div>
        </div>

        {/* Conversation Log */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <label style={{ fontSize: 12, color: 'var(--slate-400)', marginBottom: 8, display: 'block' }}>
            Conversation Log
          </label>
          <ConversationLog messages={messages} perspective="staff" />
        </div>

        {/* Controls */}
        <div className="btn-group">
          <button className="btn btn-danger btn-sm" onClick={() => window.location.reload()}>
            🗑️ Clear Conversation
          </button>
          {isSpeaking && (
            <span style={{ fontSize: 12, color: 'var(--amber-400)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span className="dot" style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--amber-400)', animation: 'pulse 1s infinite' }}></span>
              Speaking...
            </span>
          )}
        </div>
      </div>
    </section>
  );
}
