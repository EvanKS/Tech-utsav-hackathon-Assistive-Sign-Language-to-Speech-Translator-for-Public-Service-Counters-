"""
SignStabilizer — Temporal smoothing and confidence gating for sign recognition.
Converts noisy per-frame predictions into stable, debounced tokens.

Used by the backend and tested independently.
"""
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any

UNKNOWN = "__UNKNOWN__"


class SignStabilizer:
    """
    Converts a noisy per-frame prediction stream into stable, debounced tokens.

    Parameters
    ----------
    window        : int   = 12    # frames held in the ring buffer
    min_conf      : float = 0.80  # per-frame confidence gate
    min_agreement : float = 0.70  # fraction of window that must agree
    cooldown_ms   : int   = 1200  # refractory period after emitting
    exit_frames   : int   = 5     # low-conf frames required to release a held sign
    """

    def __init__(
        self,
        window: int = 12,
        min_conf: float = 0.80,
        min_agreement: float = 0.70,
        cooldown_ms: int = 1200,
        exit_frames: int = 5,
    ):
        self.window_size = window
        self.min_conf = min_conf
        self.min_agreement = min_agreement
        self.cooldown_ms = cooldown_ms
        self.exit_frames = exit_frames

        # Internal state
        self._buffer = deque(maxlen=window)
        self._last_emitted: Optional[str] = None
        self._last_emit_time: float = 0.0
        self._consecutive_unknown: int = 0
        self._held_label: Optional[str] = None

    def push(self, label: str, confidence: float, timestamp_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a single frame prediction.

        Returns a dict with:
            emitted: bool — whether a new token was emitted
            label: str|None — the emitted label
            confidence: float|None — confidence of the emitted label
            status: str — recognized | low_confidence | no_hand_detected | cooldown
            message: str — human-readable status
            debug: dict — window state for UI rendering
        """
        now_ms = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

        # Gate by confidence
        if confidence < self.min_conf:
            effective_label = UNKNOWN
        else:
            effective_label = label

        self._buffer.append((effective_label, confidence))

        # Count modal label in window
        label_counts: Dict[str, int] = {}
        for lbl, _ in self._buffer:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        modal_label = max(label_counts, key=label_counts.get)
        modal_count = label_counts[modal_label]
        agreement = modal_count / len(self._buffer) if self._buffer else 0.0

        # Track consecutive unknowns for exit logic
        if effective_label == UNKNOWN or effective_label != self._held_label:
            self._consecutive_unknown += 1
        else:
            self._consecutive_unknown = 0

        # Release held sign if enough non-matching frames
        if self._consecutive_unknown >= self.exit_frames:
            self._held_label = None

        # Check if in cooldown
        time_since_emit = now_ms - self._last_emit_time
        in_cooldown = time_since_emit < self.cooldown_ms

        # Determine emission
        emitted = False
        emitted_label = None
        emitted_confidence = None

        if (
            modal_label != UNKNOWN
            and agreement >= self.min_agreement
            and not in_cooldown
            and modal_label != self._last_emitted
        ):
            # Also emit if held label changed
            if self._held_label is None or modal_label != self._held_label:
                emitted = True
                emitted_label = modal_label
                # Average confidence of matching frames
                matching_confs = [c for l, c in self._buffer if l == modal_label]
                emitted_confidence = sum(matching_confs) / len(matching_confs) if matching_confs else 0.0

                self._last_emitted = modal_label
                self._last_emit_time = now_ms
                self._held_label = modal_label
                self._consecutive_unknown = 0

        # Determine status
        if emitted:
            status = "recognized"
            message = f"Recognized: {emitted_label}"
        elif in_cooldown and modal_label == self._last_emitted:
            status = "cooldown"
            message = "Sign held — cooldown active"
        elif modal_label == UNKNOWN:
            if effective_label == UNKNOWN and label == "no_hand":
                status = "no_hand_detected"
                message = "No hand detected. Please show your hand to the camera."
            else:
                status = "low_confidence"
                message = "Gesture not confidently recognized. Please try again."
        else:
            status = "low_confidence"
            message = "Hold the sign steady..."

        debug = self.debug_state(now_ms)

        return {
            "emitted": emitted,
            "label": emitted_label,
            "confidence": emitted_confidence,
            "status": status,
            "message": message,
            "debug": debug
        }

    def debug_state(self, now_ms: Optional[int] = None) -> Dict[str, Any]:
        """Return current stabilizer state for UI rendering."""
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)

        window_labels = [(l, round(c, 3)) for l, c in self._buffer]

        label_counts: Dict[str, int] = {}
        for lbl, _ in self._buffer:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        modal_label = max(label_counts, key=label_counts.get) if label_counts else UNKNOWN
        modal_count = label_counts.get(modal_label, 0)
        agreement = modal_count / len(self._buffer) if self._buffer else 0.0

        cooldown_remaining = max(0, self.cooldown_ms - (now_ms - self._last_emit_time))

        return {
            "window": window_labels,
            "modal_label": modal_label,
            "agreement": round(agreement, 3),
            "cooldown_remaining_ms": cooldown_remaining,
            "held_label": self._held_label,
            "consecutive_unknown": self._consecutive_unknown,
            "buffer_size": len(self._buffer),
        }

    def reset(self):
        """Reset stabilizer state."""
        self._buffer.clear()
        self._last_emitted = None
        self._last_emit_time = 0.0
        self._consecutive_unknown = 0
        self._held_label = None
