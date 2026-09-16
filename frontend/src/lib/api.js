/**
 * SignBridge API Client
 * Handles all communication with the FastAPI backend.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

async function request(path, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res;
  } catch (err) {
    console.error(`API Error [${path}]:`, err.message);
    throw err;
  }
}

export async function healthCheck() {
  const res = await request('/api/health');
  return res.json();
}

export async function recognize(sessionId, landmarks, mode = 'static', timestampMs = 0) {
  const res = await request('/api/recognize', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      mode,
      landmarks,
      timestamp_ms: timestampMs,
    }),
  });
  return res.json();
}

export async function textToSign(text) {
  const res = await request('/api/text-to-sign', {
    method: 'POST',
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function postMessage(sessionId, sender, text, confidence = null) {
  const res = await request('/api/message', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      sender,
      text,
      confidence,
    }),
  });
  return res.json();
}

export async function getSession(sessionId) {
  const res = await request(`/api/session/${sessionId}`);
  return res.json();
}

export async function resetSession(sessionId) {
  const res = await request(`/api/session/${sessionId}/reset`, { method: 'POST' });
  return res.json();
}

export async function getVocabulary() {
  const res = await request('/api/vocabulary');
  return res.json();
}

export async function speakText(text, lang = 'en') {
  const res = await request('/api/speak', {
    method: 'POST',
    body: JSON.stringify({ text, lang }),
  });
  return res.blob();
}
