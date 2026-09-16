import React, { useRef, useEffect } from 'react';

export default function ConversationLog({ messages, perspective }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="conversation-log" role="log" aria-label="Conversation history">
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          height: '100%', color: 'var(--slate-500)', fontSize: 13
        }}>
          No messages yet
        </div>
      </div>
    );
  }

  return (
    <div className="conversation-log" role="log" aria-label="Conversation history">
      {messages.map((msg, idx) => (
        <div key={idx} className={`message-bubble ${msg.sender}`}>
          <div className="message-sender">
            {msg.sender === 'staff' ? '🏛️ Staff' : '👤 Citizen'}
          </div>
          <div className="message-text">{msg.text}</div>
          {msg.confidence != null && (
            <div className="message-confidence">
              Confidence: {Math.round(msg.confidence * 100)}%
            </div>
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
