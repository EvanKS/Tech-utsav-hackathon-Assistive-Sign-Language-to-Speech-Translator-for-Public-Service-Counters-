"""
Landmark extraction and normalization using MediaPipe HandLandmarker (Tasks API).
Extracts 21 hand landmarks (x, y, z) = 63-dim feature vector per frame.
"""
import os
import cv2
import numpy as np
import mediapipe as mp

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions

# Path to the hand landmarker model - will be downloaded if needed
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hand_landmarker.task")


def _ensure_model():
    """Download the hand landmarker model if not present."""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    print(f"Downloading HandLandmarker model to {_MODEL_PATH}...")
    urllib.request.urlretrieve(url, _MODEL_PATH)
    print("Model downloaded.")
    return _MODEL_PATH


def create_hand_detector(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5):
    """Create a MediaPipe HandLandmarker detector."""
    model_path = _ensure_model()
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    return HandLandmarker.create_from_options(options)


def extract_landmarks_from_image(image, hands_detector):
    """
    Extract normalized hand landmarks from a single image.
    
    Returns:
        numpy array of shape (63,) or None if no hand detected.
    """
    if image is None:
        return None
    
    # Convert BGR to RGB for MediaPipe
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    results = hands_detector.detect(mp_image)
    
    if not results.hand_landmarks or len(results.hand_landmarks) == 0:
        return None
    
    hand = results.hand_landmarks[0]
    landmarks = []
    for lm in hand:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    return normalize_landmarks(np.array(landmarks, dtype=np.float32))


def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist (landmark 0) and scale by hand size.
    This makes the features translation and scale invariant.
    """
    if landmarks is None or len(landmarks) != 63:
        return landmarks
    
    coords = landmarks.reshape(21, 3)
    
    # Translate: center on wrist
    wrist = coords[0].copy()
    coords = coords - wrist
    
    # Scale: normalize by max distance from wrist
    distances = np.linalg.norm(coords, axis=1)
    max_dist = np.max(distances)
    if max_dist > 1e-6:
        coords = coords / max_dist
    
    return coords.flatten().astype(np.float32)


def extract_landmarks_from_video(video_path, hands_detector, num_frames=32):
    """
    Extract landmark sequences from a video file.
    Uniformly samples num_frames from the video.
    
    Returns:
        numpy array of shape (num_frames, 63) or None if insufficient detections.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return None
    
    # Uniformly sample frame indices
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
    
    all_landmarks = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            all_landmarks.append(np.zeros(63, dtype=np.float32))
            continue
        
        lm = extract_landmarks_from_image(frame, hands_detector)
        if lm is not None:
            all_landmarks.append(lm)
        else:
            all_landmarks.append(np.zeros(63, dtype=np.float32))
    
    cap.release()
    
    # Pad if we got fewer frames than needed
    while len(all_landmarks) < num_frames:
        all_landmarks.append(np.zeros(63, dtype=np.float32))
    
    return np.array(all_landmarks[:num_frames], dtype=np.float32)


def compute_precision_features(landmarks_63):
    """
    Extract enhanced 87-dimensional precision feature representation from 63D raw landmarks:
    - Translation & hand-scale invariant coordinates
    - Canonical in-plane rotation normalization (wrist to middle MCP aligned with +Y)
    - 8 relative fingertip distance ratios (resolves U vs V, C vs G, digit 3)
    - 6 signed depth-order features (resolves thumb-crossing occlusions)
    - 10 joint flexion/curl cosine angles across all fingers
    
    Returns:
        numpy array of shape (87,) dtype float32
    """
    if landmarks_63 is None or len(landmarks_63) != 63:
        return np.zeros(87, dtype=np.float32)
    
    coords = np.array(landmarks_63, dtype=np.float32).reshape(21, 3)
    
    # Translation: center on wrist (point 0)
    coords = coords - coords[0]
    
    # Scale: normalize by max distance from wrist
    max_d = np.max(np.linalg.norm(coords, axis=1))
    if max_d > 1e-6:
        coords = coords / max_d
        
    # Canonical in-plane rotation: align vector (wrist p0 -> middle MCP p9) with +Y axis
    v_mid = coords[9, :2]  # [x, y]
    norm_mid = np.linalg.norm(v_mid)
    if norm_mid > 1e-6:
        theta = np.arctan2(v_mid[0], v_mid[1])  # angle from +Y
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        coords[:, :2] = coords[:, :2] @ R.T
        
    flat_coords = coords.flatten()  # 63 dims
    
    # Explicit fingertip and keypoint distance features (8 dims)
    # 4: thumb tip, 8: index tip, 9: middle MCP, 12: middle tip, 16: ring tip, 20: pinky tip
    d_8_12 = np.linalg.norm(coords[8] - coords[12])      # U vs V separation
    d_4_8 = np.linalg.norm(coords[4] - coords[8])        # C vs G separation
    d_4_12 = np.linalg.norm(coords[4] - coords[12])
    d_4_16 = np.linalg.norm(coords[4] - coords[16])
    d_4_20 = np.linalg.norm(coords[4] - coords[20])
    d_4_9 = np.linalg.norm(coords[4] - coords[9])        # Digit 3 thumb extension
    d_12_16 = np.linalg.norm(coords[12] - coords[16])
    d_16_20 = np.linalg.norm(coords[16] - coords[20])
    dist_feats = np.array([d_8_12, d_4_8, d_4_12, d_4_16, d_4_20, d_4_9, d_12_16, d_16_20], dtype=np.float32)
    
    # Depth-order / occlusion features (6 dims)
    z_diff_4_8 = coords[4, 2] - coords[8, 2]
    z_diff_4_12 = coords[4, 2] - coords[12, 2]
    z_diff_4_16 = coords[4, 2] - coords[16, 2]
    z_diff_4_20 = coords[4, 2] - coords[20, 2]
    z_diff_8_12 = coords[8, 2] - coords[12, 2]
    sign_4_8 = np.sign(z_diff_4_8)  # Explicit "is thumb in front of index" bit
    depth_feats = np.array([z_diff_4_8, z_diff_4_12, z_diff_4_16, z_diff_4_20, z_diff_8_12, sign_4_8], dtype=np.float32)
    
    # Per-finger joint flexion cosine angles (10 dims: 2 angles per finger x 5 fingers)
    finger_joints = [
        (0, 1, 2, 3),   # Thumb: wrist, CMC, MCP, IP
        (0, 5, 6, 7),   # Index: wrist, MCP, PIP, DIP
        (0, 9, 10, 11), # Middle
        (0, 13, 14, 15),# Ring
        (0, 17, 18, 19) # Pinky
    ]
    angle_feats = []
    for base, mcp, pip, dip in finger_joints:
        # MCP joint angle
        v1 = coords[base] - coords[mcp]
        v2 = coords[pip] - coords[mcp]
        cos1 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
        
        # PIP joint angle
        v3 = coords[mcp] - coords[pip]
        v4 = coords[dip] - coords[pip]
        cos2 = np.dot(v3, v4) / (np.linalg.norm(v3) * np.linalg.norm(v4) + 1e-7)
        
        angle_feats.extend([float(np.clip(cos1, -1.0, 1.0)), float(np.clip(cos2, -1.0, 1.0))])
    
    angle_feats = np.array(angle_feats, dtype=np.float32)
    
    return np.concatenate([flat_coords, dist_feats, depth_feats, angle_feats]).astype(np.float32)


def compute_temporal_precision_features(seq_32_63):
    """
    Augment a sequence of (32, 63) landmarks with frame-to-frame velocity deltas:
    Delta_x_t = x_t - x_{t-1}
    Concatenated: [x_t, Delta_x_t] -> shape (32, 126)
    
    Returns:
        numpy array of shape (32, 126) dtype float32
    """
    seq = np.array(seq_32_63, dtype=np.float32)
    if seq.shape != (32, 63):
        return np.zeros((32, 126), dtype=np.float32)
        
    vel = np.zeros_like(seq)
    # Velocity on active consecutive frames
    for t in range(1, 32):
        if np.any(seq[t] != 0) and np.any(seq[t - 1] != 0):
            vel[t] = seq[t] - seq[t - 1]
            
    return np.concatenate([seq, vel], axis=-1).astype(np.float32)

