"""
Backend wrapper for SignStabilizer — manages per-session stabilizer instances.
"""
import sys
import os
from typing import Dict

# Add project root to path so we can import from ml/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.temporal_smoothing import SignStabilizer

# Session-scoped stabilizers
_stabilizers: Dict[str, SignStabilizer] = {}


def get_stabilizer(session_id: str, mode: str = "digits") -> SignStabilizer:
    """Get or create a stabilizer for a session and recognition mode."""
    key = f"{session_id}_{mode}"
    if key not in _stabilizers:
        if mode in ["temporal", "words", "gesture_asl"]:
            # Sequence models evaluated over sliding trajectories (higher precision velocity Bi-GRU/LSTM)
            _stabilizers[key] = SignStabilizer(
                window=3,
                min_conf=0.45,
                min_agreement=0.55,
                cooldown_ms=1000,
                exit_frames=2,
            )
        else:
            # Frame-level models (digits, letters) with 87D precision features (sharper softmax)
            _stabilizers[key] = SignStabilizer(
                window=5,
                min_conf=0.60,
                min_agreement=0.60,
                cooldown_ms=850,
                exit_frames=2,
            )
    return _stabilizers[key]


def reset_stabilizer(session_id: str):
    """Reset a session's stabilizer."""
    if session_id in _stabilizers:
        _stabilizers[session_id].reset()


def remove_stabilizer(session_id: str):
    """Remove a session's stabilizer."""
    _stabilizers.pop(session_id, None)
