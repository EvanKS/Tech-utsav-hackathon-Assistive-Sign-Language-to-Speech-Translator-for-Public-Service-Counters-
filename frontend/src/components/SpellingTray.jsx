import React from 'react';

export default function SpellingTray({
  buffer,
  onBackspace,
  onClear,
  onAddSpace,
  onCommitWord,
  onSendDirect,
  disabled = false
}) {
  const currentWord = buffer.join('');

  return (
    <div className="spelling-tray-container" role="region" aria-label="Fingerspelling Word Buffer">
      <div className="spelling-tray-header">
        <div className="spelling-tray-title">
          <span className="icon">🔤</span>
          <span>Fingerspelling Tray (ASL Letters A–Z)</span>
        </div>
        <span className="spelling-tray-hint">
          Accumulates recognized letters into words before sending
        </span>
      </div>

      <div className="spelling-tray-display">
        {buffer.length === 0 ? (
          <span className="spelling-placeholder">Sign letters to build a word...</span>
        ) : (
          <div className="spelling-letters-row" role="list">
            {buffer.map((char, index) => (
              <span key={index} className={`spelling-char-chip ${char === ' ' ? 'char-space' : ''}`} role="listitem">
                {char === ' ' ? '␣' : char}
              </span>
            ))}
            <span className="spelling-cursor" />
          </div>
        )}
      </div>

      {currentWord && (
        <div className="spelling-word-preview">
          <span>Composed: </span>
          <strong>{currentWord.trim()}</strong>
        </div>
      )}

      <div className="spelling-actions">
        <div className="btn-group">
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onBackspace}
            disabled={buffer.length === 0 || disabled}
            title="Remove last letter"
          >
            ⌫ Backspace
          </button>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onAddSpace}
            disabled={buffer.length === 0 || disabled}
            title="Insert space"
          >
            ␣ Space
          </button>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onClear}
            disabled={buffer.length === 0 || disabled}
            title="Clear all letters"
          >
            ✕ Clear
          </button>
        </div>

        <div className="btn-group">
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onCommitWord}
            disabled={!currentWord.trim() || disabled}
            id="btn-commit-word"
            title="Add composed word into the message tray"
          >
            ➕ Add Word to Message
          </button>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={onSendDirect}
            disabled={!currentWord.trim() || disabled}
            id="btn-send-spelled-word"
            title="Send spelled word directly to staff"
          >
            📤 Send Word
          </button>
        </div>
      </div>
    </div>
  );
}
