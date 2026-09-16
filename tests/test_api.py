"""
Integration tests for FastAPI backend endpoints:
- GET /api/health (all 4 models loaded)
- GET /api/vocabulary (all 4 heads present with language attribution)
- POST /api/recognize (digits - static 63-dim, ASL)
- POST /api/recognize (letters - static 63-dim, ASL)
- POST /api/recognize (words - temporal 2016-dim, ISL)
- POST /api/recognize (gesture_asl - temporal 2016-dim, ASL)
- POST /api/text-to-sign (with fingerspelling fallback decomposition)
- POST /api/message
- GET /api/session/{session_id}
- POST /api/speak
"""
import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.main import app


def test_api_endpoints():
    with TestClient(app) as client:
        # 1. Health check (verifies all 4 models loaded)
        res = client.get("/api/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        health_data = res.json()
        print("[PASS] Health status:", health_data)
        assert health_data["status"] == "healthy"
        assert health_data["models_loaded"]["static"] is True
        assert health_data["models_loaded"]["temporal"] is True
        assert health_data["models_loaded"]["letters"] is True
        assert health_data["models_loaded"]["gesture_asl"] is True

        # 2. Vocabulary endpoint (all 4 heads present)
        res = client.get("/api/vocabulary")
        assert res.status_code == 200, f"Vocabulary check failed: {res.text}"
        vocab = res.json()
        assert "recognition" in vocab
        assert "production" in vocab
        assert len(vocab["recognition"]["digits"]["classes"]) == 10
        assert len(vocab["recognition"]["words"]["classes"]) == 8
        assert len(vocab["recognition"]["letters"]["classes"]) == 26
        assert len(vocab["recognition"]["gesture_asl"]["classes"]) == 5
        assert vocab["recognition"]["words"]["language"] == "isl"
        assert vocab["recognition"]["letters"]["language"] == "asl"
        print(f"[PASS] Vocabulary verified: 10 digits (ASL), 8 words (ISL), 26 letters (ASL), 5 gestures (ASL)")

        # 3. Static digit recognition (single hand 63 landmarks, ASL)
        dummy_landmarks = [0.0] * 63
        res = client.post("/api/recognize", json={
            "session_id": "test-session-1",
            "mode": "digits",
            "landmarks": dummy_landmarks,
            "timestamp_ms": 1000
        })
        assert res.status_code == 200, f"Digit recognize failed: {res.text}"
        pred = res.json()
        print(f"[PASS] Digits recognition: label={pred['raw']['label']}, head={pred['head']}, lang={pred['language']}")
        assert pred["head"] == "digits"
        assert pred["language"] == "asl"
        assert "raw" in pred
        assert "stable" in pred

        # 4. Static letter recognition (single hand 63 landmarks, ASL)
        res = client.post("/api/recognize", json={
            "session_id": "test-session-1",
            "mode": "letters",
            "landmarks": dummy_landmarks,
            "timestamp_ms": 1500
        })
        assert res.status_code == 200, f"Letter recognize failed: {res.text}"
        pred_letter = res.json()
        print(f"[PASS] Letter recognition: label={pred_letter['raw']['label']}, head={pred_letter['head']}, lang={pred_letter['language']}")
        assert pred_letter["head"] == "letters"
        assert pred_letter["language"] == "asl"
        assert pred_letter["raw"]["label"] in [chr(ord('A') + i) for i in range(26)]

        # 5. Temporal word recognition (32 frames of 63 landmarks = 2016 flat floats, ISL)
        dummy_seq = [0.0] * (32 * 63)
        res = client.post("/api/recognize", json={
            "session_id": "test-session-1",
            "mode": "words",
            "landmarks": dummy_seq,
            "timestamp_ms": 2000
        })
        assert res.status_code == 200, f"Temporal words recognize failed: {res.text}"
        pred_words = res.json()
        print(f"[PASS] Temporal words: label={pred_words['raw']['label']}, head={pred_words['head']}, lang={pred_words['language']}")
        assert pred_words["head"] == "words"
        assert pred_words["language"] == "isl"

        # 6. Dynamic ASL phrase gesture recognition (32 frames of 63 landmarks = 2016 flat floats, ASL)
        res = client.post("/api/recognize", json={
            "session_id": "test-session-1",
            "mode": "gesture_asl",
            "landmarks": dummy_seq,
            "timestamp_ms": 2500
        })
        assert res.status_code == 200, f"Gesture ASL recognize failed: {res.text}"
        pred_gesture = res.json()
        print(f"[PASS] ASL gesture: label={pred_gesture['raw']['label']}, head={pred_gesture['head']}, lang={pred_gesture['language']}")
        assert pred_gesture["head"] == "gesture_asl"
        assert pred_gesture["language"] == "asl"
        assert pred_gesture["raw"]["label"] in ["HELLO", "NO", "SORRY", "THANKYOU", "YES"]

        # 7. Captions / Text-to-Sign with Fingerspelling Fallback
        # "Hello" -> whole word sign
        # "cab" -> unmatched word falling back to letters 'c', 'a', 'b'
        # "5" -> digit sign
        res = client.post("/api/text-to-sign", json={
            "text": "Hello cab 5"
        })
        assert res.status_code == 200, f"Text to sign failed: {res.text}"
        captions = res.json()
        tokens = captions["tokens"]
        print(f"[PASS] Text to sign tokens: {[t['token'] for t in tokens]}, coverage: {captions['coverage']}")
        assert captions["coverage"] == 1.0, f"Expected 1.0 coverage via fingerspelling fallback, got {captions['coverage']}"
        assert any(t["token"] == "hello" and t["kind"] == "word" for t in tokens)
        assert any(t["token"] == "5" and t["kind"] == "digit" for t in tokens)
        assert any(t["token"] == "c" and t["kind"] == "letter" for t in tokens)
        assert any(t["token"] == "a" and t["kind"] == "letter" for t in tokens)
        assert any(t["token"] == "b" and t["kind"] == "letter" for t in tokens)

        # 8. Messages and Session
        res = client.post("/api/message", json={
            "session_id": "test-session-1",
            "sender": "staff",
            "text": "Hello citizen, counter 3 is open",
            "confidence": 1.0
        })
        assert res.status_code == 200, f"Message post failed: {res.text}"
        session_data = res.json()
        assert len(session_data["messages"]) >= 1
        print("[PASS] Session message logged successfully")

        # 9. Speak endpoint (TTS)
        res = client.post("/api/speak", json={
            "text": "Hello, counter 3 is open",
            "lang": "en"
        })
        assert res.status_code == 200, f"Speak failed: {res.text}"
        assert res.headers["content-type"] == "audio/mpeg"
        assert len(res.content) > 1000
        print(f"[PASS] TTS audio generated successfully: {len(res.content)} bytes")

    print("\n" + "=" * 60)
    print("ALL API ENDPOINT & MULTI-HEAD INTEGRATION TESTS PASSED (100% SUCCESS)!")
    print("=" * 60)


if __name__ == "__main__":
    test_api_endpoints()
