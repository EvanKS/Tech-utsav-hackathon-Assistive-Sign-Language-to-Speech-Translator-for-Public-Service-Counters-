"""
Unit tests for SignStabilizer temporal smoothing.
Tests: noisy outlier rejection, duplicate suppression, gesture transition,
       all-low-confidence → no emission, cooldown expiry.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from ml.temporal_smoothing import SignStabilizer, UNKNOWN


def test_single_sign_recognition():
    """A consistent sign should be emitted once."""
    s = SignStabilizer(window=5, min_conf=0.7, min_agreement=0.6, cooldown_ms=500, exit_frames=3)
    
    emitted = []
    for i in range(10):
        result = s.push("hello", 0.95, timestamp_ms=i * 100)
        if result["emitted"]:
            emitted.append(result["label"])
    
    assert len(emitted) == 1, f"Expected 1 emission, got {len(emitted)}"
    assert emitted[0] == "hello"
    print("[PASS] test_single_sign_recognition")


def test_noisy_outlier_rejection():
    """Single outlier frames should not cause emission."""
    s = SignStabilizer(window=5, min_conf=0.7, min_agreement=0.6, cooldown_ms=100, exit_frames=3)
    
    emitted = []
    # Fill window with "hello"
    for i in range(4):
        result = s.push("hello", 0.9, timestamp_ms=i * 100)
        if result["emitted"]:
            emitted.append(result["label"])
    
    # Single outlier
    result = s.push("goodbye", 0.85, timestamp_ms=500)
    if result["emitted"]:
        emitted.append(result["label"])
    
    # Back to "hello"
    for i in range(3):
        result = s.push("hello", 0.9, timestamp_ms=600 + i * 100)
        if result["emitted"]:
            emitted.append(result["label"])
    
    # "goodbye" should never have been emitted
    assert "goodbye" not in emitted, f"Outlier 'goodbye' was incorrectly emitted"
    print("[PASS] test_noisy_outlier_rejection")


def test_duplicate_suppression():
    """Same sign held should not spam emissions."""
    s = SignStabilizer(window=5, min_conf=0.7, min_agreement=0.6, cooldown_ms=1000, exit_frames=3)
    
    emitted = []
    for i in range(20):
        result = s.push("hello", 0.95, timestamp_ms=i * 100)
        if result["emitted"]:
            emitted.append(result["label"])
    
    assert len(emitted) == 1, f"Expected 1 emission, got {len(emitted)}: {emitted}"
    print("[PASS] test_duplicate_suppression")


def test_gesture_transition():
    """Changing from one sign to another should emit the new sign."""
    s = SignStabilizer(window=5, min_conf=0.7, min_agreement=0.6, cooldown_ms=100, exit_frames=3)
    
    emitted = []
    
    # Sign "hello"
    for i in range(6):
        result = s.push("hello", 0.9, timestamp_ms=i * 100)
        if result["emitted"]:
            emitted.append(result["label"])
    
    # Transition gap (unknowns to clear the held state) — need exit_frames (3) unknowns
    for i in range(6):
        result = s.push("none", 0.3, timestamp_ms=700 + i * 100)
        if result["emitted"]:
            emitted.append(result["label"])
    
    # Sign "yes" — well past cooldown (100ms), fill window fully
    for i in range(8):
        result = s.push("yes", 0.9, timestamp_ms=2000 + i * 100)
        if result["emitted"]:
            emitted.append(result["label"])
    
    assert "hello" in emitted, "Expected 'hello' to be emitted"
    assert "yes" in emitted, f"Expected 'yes' to be emitted, got {emitted}"
    print("[PASS] test_gesture_transition")


def test_all_low_confidence():
    """No emission when confidence is always below threshold."""
    s = SignStabilizer(window=5, min_conf=0.7, min_agreement=0.6, cooldown_ms=100, exit_frames=3)
    
    emitted = []
    for i in range(15):
        result = s.push("hello", 0.3, timestamp_ms=i * 100)
        if result["emitted"]:
            emitted.append(result["label"])
        assert result["status"] != "recognized", "Should not recognize low-confidence"
    
    assert len(emitted) == 0, f"Expected 0 emissions, got {len(emitted)}"
    print("[PASS] test_all_low_confidence")


def test_cooldown_expiry():
    """After cooldown, a different sign should be emittable."""
    s = SignStabilizer(window=5, min_conf=0.7, min_agreement=0.6, cooldown_ms=500, exit_frames=3)
    
    emitted = []
    
    # Sign "hello"
    for i in range(6):
        result = s.push("hello", 0.9, timestamp_ms=i * 100)
        if result["emitted"]:
            emitted.append(result["label"])
    
    # Clear held state
    for i in range(5):
        result = s.push("x", 0.3, timestamp_ms=700 + i * 100)
    
    # Wait past cooldown, then sign "yes"
    for i in range(6):
        result = s.push("yes", 0.9, timestamp_ms=1500 + i * 100)
        if result["emitted"]:
            emitted.append(result["label"])
    
    assert len(emitted) == 2, f"Expected 2 emissions, got {len(emitted)}: {emitted}"
    print("[PASS] test_cooldown_expiry")


def test_debug_state():
    """debug_state() should return valid data."""
    s = SignStabilizer(window=5)
    s.push("hello", 0.9, timestamp_ms=1000)
    
    debug = s.debug_state(now_ms=1000)
    assert "window" in debug
    assert "agreement" in debug
    assert "cooldown_remaining_ms" in debug
    assert debug["buffer_size"] == 1
    print("[PASS] test_debug_state")


if __name__ == "__main__":
    test_single_sign_recognition()
    test_noisy_outlier_rejection()
    test_duplicate_suppression()
    test_gesture_transition()
    test_all_low_confidence()
    test_cooldown_expiry()
    test_debug_state()
    print("\n[PASS] All smoothing tests passed!")
