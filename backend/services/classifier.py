"""
Model loading and prediction service.
Loads trained .keras models at startup and provides prediction interface
across all 4 heads:
1. Static Digits (ASL) — landmark_digits.keras (10 classes: 0-9)
2. Temporal Words (ISL) — temporal_words.keras (8 classes: eat, go, hello, etc.)
3. Static Letters (ASL) — landmark_letters.keras (26 classes: A-Z)
4. Dynamic Gestures (ASL) — temporal_letters_phrases.keras (5 classes: HELLO, NO, etc.)
"""
import os
import json
import numpy as np
from typing import Optional, Dict, List, Tuple

_static_model = None
_temporal_model = None
_letters_model = None
_gesture_asl_model = None

_static_labels: List[str] = []
_temporal_labels: List[str] = []
_letters_labels: List[str] = []
_gesture_asl_labels: List[str] = []

_models_loaded = {
    "static": False,
    "temporal": False,
    "letters": False,
    "gesture_asl": False
}


def load_models(model_dir: str):
    """Load all 4 trained models from disk."""
    global _static_model, _temporal_model, _letters_model, _gesture_asl_model
    global _static_labels, _temporal_labels, _letters_labels, _gesture_asl_labels
    global _models_loaded
    
    # Import tensorflow lazily
    import tensorflow as tf
    
    # 1. Load static digit model (D1 - ASL)
    static_path = os.path.join(model_dir, "landmark_digits.keras")
    static_labels_path = os.path.join(model_dir, "landmark_digits_labels.json")
    if os.path.exists(static_path) and os.path.exists(static_labels_path):
        try:
            _static_model = tf.keras.models.load_model(static_path)
            with open(static_labels_path) as f:
                _static_labels = json.load(f)
            _models_loaded["static"] = True
            print(f"[OK] Static digit model loaded: {len(_static_labels)} classes (ASL)")
        except Exception as e:
            print(f"[ERROR] Failed to load static digit model: {e}")
    else:
        print(f"[WARN] Static digit model not found at {static_path}")
    
    # 2. Load temporal word model (D2 - ISL)
    temporal_path = os.path.join(model_dir, "temporal_words.keras")
    temporal_labels_path = os.path.join(model_dir, "temporal_words_labels.json")
    if os.path.exists(temporal_path) and os.path.exists(temporal_labels_path):
        try:
            _temporal_model = tf.keras.models.load_model(temporal_path)
            with open(temporal_labels_path) as f:
                _temporal_labels = json.load(f)
            _models_loaded["temporal"] = True
            print(f"[OK] Temporal word model loaded: {len(_temporal_labels)} classes (ISL)")
        except Exception as e:
            print(f"[ERROR] Failed to load temporal word model: {e}")
    else:
        print(f"[WARN] Temporal word model not found at {temporal_path}")
        
    # 3. Load static letters model (SignAlphaSet - ASL)
    letters_path = os.path.join(model_dir, "landmark_letters.keras")
    letters_labels_path = os.path.join(model_dir, "landmark_letters_labels.json")
    if os.path.exists(letters_path) and os.path.exists(letters_labels_path):
        try:
            _letters_model = tf.keras.models.load_model(letters_path)
            with open(letters_labels_path) as f:
                _letters_labels = json.load(f)
            _models_loaded["letters"] = True
            print(f"[OK] Static letters model loaded: {len(_letters_labels)} classes (ASL)")
        except Exception as e:
            print(f"[ERROR] Failed to load static letters model: {e}")
    else:
        print(f"[WARN] Static letters model not found at {letters_path}")
        
    # 4. Load dynamic ASL phrase gesture model (SignAlphaSet - ASL)
    gesture_path = os.path.join(model_dir, "temporal_letters_phrases.keras")
    gesture_labels_path = os.path.join(model_dir, "temporal_letters_phrases_labels.json")
    if os.path.exists(gesture_path) and os.path.exists(gesture_labels_path):
        try:
            _gesture_asl_model = tf.keras.models.load_model(gesture_path)
            with open(gesture_labels_path) as f:
                _gesture_asl_labels = json.load(f)
            _models_loaded["gesture_asl"] = True
            print(f"[OK] Temporal ASL gesture model loaded: {len(_gesture_asl_labels)} classes (ASL)")
        except Exception as e:
            print(f"[ERROR] Failed to load temporal ASL gesture model: {e}")
    else:
        print(f"[WARN] Temporal ASL gesture model not found at {gesture_path}")


def get_models_status() -> Dict[str, bool]:
    return _models_loaded.copy()


def get_model_labels() -> Dict[str, List[str]]:
    return {
        "digits": _static_labels.copy(),
        "words": _temporal_labels.copy(),
        "letters": _letters_labels.copy(),
        "gesture_asl": _gesture_asl_labels.copy()
    }



# Import precision feature extractors from ml.preprocess
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from ml.preprocess import compute_precision_features, compute_temporal_precision_features
except ImportError:
    compute_precision_features = None
    compute_temporal_precision_features = None


def predict_static(landmarks: List[float], top_k: int = 5) -> Optional[Dict]:
    """Predict digit from landmark vector (ASL), supporting both right and left hands."""
    if _static_model is None or not _models_loaded["static"]:
        return None
    
    # Check expected feature dimension (87 for precision model, 63 for legacy)
    expected_dim = _static_model.input_shape[-1]
    if expected_dim == 87 and compute_precision_features is not None:
        feat = compute_precision_features(landmarks).reshape(1, -1)
        # Mirrored left-hand evaluation
        lm_arr = np.array(landmarks, dtype=np.float32).reshape(21, 3)
        lm_arr[:, 0] = -lm_arr[:, 0]
        feat_flip = compute_precision_features(lm_arr.flatten()).reshape(1, -1)
        
        probs = _static_model.predict(feat, verbose=0)[0]
        probs_flip = _static_model.predict(feat_flip, verbose=0)[0]
    else:
        x = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        probs = _static_model.predict(x, verbose=0)[0]
        x_flip = x.copy().reshape(1, 21, 3)
        x_flip[:, :, 0] = -x_flip[:, :, 0]
        x_flip = x_flip.reshape(1, 63)
        probs_flip = _static_model.predict(x_flip, verbose=0)[0]
    
    # Pick orientation with higher peak confidence
    if float(np.max(probs_flip)) > float(np.max(probs)):
        probs = probs_flip
    
    sorted_indices = np.argsort(probs)[::-1]
    pred_idx = sorted_indices[0]
    
    return {
        "label": _static_labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "top_k": [
            {"label": _static_labels[i], "prob": float(probs[i])}
            for i in sorted_indices[:top_k]
        ]
    }


def _prepare_temporal_sequence(raw_seq: np.ndarray, expected_len: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample non-zero frames uniformly across expected_len (32),
    matching the exact training distribution from extract_landmarks_from_video.
    Returns: (seq_normal, seq_flipped) both shape (32, 63)
    """
    # Find non-zero active frames
    active_mask = np.any(raw_seq != 0, axis=-1)
    active_frames = raw_seq[active_mask]
    
    if len(active_frames) >= 6:
        # Uniform temporal resampling across gesture
        indices = np.linspace(0, len(active_frames) - 1, expected_len, dtype=int)
        seq = active_frames[indices]
    elif len(raw_seq) == expected_len:
        seq = raw_seq.copy()
    else:
        # Fallback padding
        if len(raw_seq) < expected_len:
            pad = np.zeros((expected_len - len(raw_seq), 63), dtype=np.float32)
            seq = np.concatenate([raw_seq, pad])
        else:
            seq = raw_seq[-expected_len:]
            
    # Mirrored x sequence for dual-hand support
    seq_flip = seq.copy().reshape(expected_len, 21, 3)
    seq_flip[:, :, 0] = -seq_flip[:, :, 0]
    seq_flip = seq_flip.reshape(expected_len, 63)
    
    return seq.astype(np.float32), seq_flip.astype(np.float32)


def predict_temporal(landmark_sequence: List[List[float]], top_k: int = 5) -> Optional[Dict]:
    """Predict word from temporal landmark sequence (ISL) with dual-hand symmetry and uniform resampling."""
    if _temporal_model is None or not _models_loaded["temporal"]:
        return None
    
    raw_seq = np.array(landmark_sequence, dtype=np.float32)
    expected_len = _temporal_model.input_shape[1]
    expected_dim = _temporal_model.input_shape[2]
    
    seq, seq_flip = _prepare_temporal_sequence(raw_seq, expected_len)
    
    if expected_dim == 126 and compute_temporal_precision_features is not None:
        x = compute_temporal_precision_features(seq).reshape(1, expected_len, 126)
        x_flip = compute_temporal_precision_features(seq_flip).reshape(1, expected_len, 126)
    else:
        x = seq.reshape(1, expected_len, 63)
        x_flip = seq_flip.reshape(1, expected_len, 63)
        
    probs = _temporal_model.predict(x, verbose=0)[0]
    probs_flip = _temporal_model.predict(x_flip, verbose=0)[0]
    
    if float(np.max(probs_flip)) > float(np.max(probs)):
        probs = probs_flip
        
    sorted_indices = np.argsort(probs)[::-1]
    pred_idx = sorted_indices[0]
    
    return {
        "label": _temporal_labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "top_k": [
            {"label": _temporal_labels[i], "prob": float(probs[i])}
            for i in sorted_indices[:top_k]
        ]
    }


def predict_letters(landmarks: List[float], top_k: int = 5) -> Optional[Dict]:
    """Predict letter from landmark vector (ASL), supporting both right and left hands."""
    if _letters_model is None or not _models_loaded["letters"]:
        return None
    
    expected_dim = _letters_model.input_shape[-1]
    if expected_dim == 87 and compute_precision_features is not None:
        feat = compute_precision_features(landmarks).reshape(1, -1)
        lm_arr = np.array(landmarks, dtype=np.float32).reshape(21, 3)
        lm_arr[:, 0] = -lm_arr[:, 0]
        feat_flip = compute_precision_features(lm_arr.flatten()).reshape(1, -1)
        
        probs = _letters_model.predict(feat, verbose=0)[0]
        probs_flip = _letters_model.predict(feat_flip, verbose=0)[0]
    else:
        x = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        probs = _letters_model.predict(x, verbose=0)[0]
        x_flip = x.copy().reshape(1, 21, 3)
        x_flip[:, :, 0] = -x_flip[:, :, 0]
        x_flip = x_flip.reshape(1, 63)
        probs_flip = _letters_model.predict(x_flip, verbose=0)[0]
    
    if float(np.max(probs_flip)) > float(np.max(probs)):
        probs = probs_flip
    
    sorted_indices = np.argsort(probs)[::-1]
    pred_idx = sorted_indices[0]
    
    return {
        "label": _letters_labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "top_k": [
            {"label": _letters_labels[i], "prob": float(probs[i])}
            for i in sorted_indices[:top_k]
        ]
    }


def predict_gesture_asl(landmark_sequence: List[List[float]], top_k: int = 5) -> Optional[Dict]:
    """Predict dynamic phrase gesture from sequence (ASL) with dual-hand symmetry and uniform resampling."""
    if _gesture_asl_model is None or not _models_loaded["gesture_asl"]:
        return None
    
    raw_seq = np.array(landmark_sequence, dtype=np.float32)
    expected_len = _gesture_asl_model.input_shape[1]
    expected_dim = _gesture_asl_model.input_shape[2]
    
    seq, seq_flip = _prepare_temporal_sequence(raw_seq, expected_len)
        
    if expected_dim == 126 and compute_temporal_precision_features is not None:
        x = compute_temporal_precision_features(seq).reshape(1, expected_len, 126)
        x_flip = compute_temporal_precision_features(seq_flip).reshape(1, expected_len, 126)
    else:
        x = seq.reshape(1, expected_len, 63)
        x_flip = seq_flip.reshape(1, expected_len, 63)
        
    probs = _gesture_asl_model.predict(x, verbose=0)[0]
    probs_flip = _gesture_asl_model.predict(x_flip, verbose=0)[0]
    
    if float(np.max(probs_flip)) > float(np.max(probs)):
        probs = probs_flip
        
    sorted_indices = np.argsort(probs)[::-1]
    pred_idx = sorted_indices[0]
    
    return {
        "label": _gesture_asl_labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "top_k": [
            {"label": _gesture_asl_labels[i], "prob": float(probs[i])}
            for i in sorted_indices[:top_k]
        ]
    }


