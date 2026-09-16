"""
Precision landmark feature extractors for SignBridge.
Pure NumPy implementations with ZERO external CV or MediaPipe dependencies.
"""
import numpy as np

def compute_precision_features(landmarks_63):
    """
    Computes 87-dimensional invariant representation from raw 63-dim landmarks:
    - 63 base normalized coords (wrist-centered, scale-normalized, canonical in-plane rotation)
    - 8 fingertip & keypoint distance features (U vs V, C vs G, etc.)
    - 6 depth-order / occlusion features (z-differences + thumb-front sign bit)
    - 10 joint flexion cosine angles (MCP and PIP for 5 fingers)
    
    Total: 63 + 8 + 6 + 10 = 87 features.
    """
    lm = np.array(landmarks_63, dtype=np.float32)
    if lm.shape != (63,):
        return np.zeros(87, dtype=np.float32)
        
    coords = lm.reshape(21, 3)
    
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
        v1 = coords[base] - coords[mcp]
        v2 = coords[pip] - coords[mcp]
        cos1 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
        
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
    for t in range(1, 32):
        if np.any(seq[t] != 0) and np.any(seq[t - 1] != 0):
            vel[t] = seq[t] - seq[t - 1]
            
    return np.concatenate([seq, vel], axis=-1).astype(np.float32)
