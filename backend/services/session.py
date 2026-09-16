"""
In-memory conversation/session store.
"""
import time
from typing import Dict, List, Any

_sessions: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str) -> Dict[str, Any]:
    """Get or create a session."""
    if session_id not in _sessions:
        _sessions[session_id] = {
            "messages": [],
            "started_at": time.time(),
            "session_id": session_id,
        }
    return _sessions[session_id]


def add_message(session_id: str, sender: str, text: str, confidence: float = None) -> List[Dict]:
    """Add a message to the session log."""
    session = get_session(session_id)
    msg = {
        "sender": sender,
        "text": text,
        "confidence": confidence,
        "timestamp": time.time(),
    }
    session["messages"].append(msg)
    return session["messages"]


def reset_session(session_id: str):
    """Clear session messages."""
    if session_id in _sessions:
        _sessions[session_id]["messages"] = []
