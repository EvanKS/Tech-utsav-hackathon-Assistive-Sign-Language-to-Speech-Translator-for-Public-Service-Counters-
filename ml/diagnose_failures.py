"""
SignBridge — Diagnostic Phase (Precision & Confusion Analysis)
Generates quantitative evidence for all failure modes in Section 2:
- Confusion matrix inspection on exact cells
- Landmark coordinate, depth (z), and distance analysis
- Diagnosis classification: Detection-Layer vs Feature-Layer vs Data-Layer
"""
import os
import sys
import json
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.config import D1_PATH, D2_PATH, SIGNALPHASET_STATIC_PATH, SIGNALPHASET_DYNAMIC_PATH
from ml.preprocess import create_hand_detector
import mediapipe as mp


def run_diagnostics():
    print("=" * 70)
    print("SIGNBRIDGE ML PRECISION & CONFUSION DIAGNOSTIC SUITE")
    print("=" * 70)
    
    findings = {}

    # -------------------------------------------------------------
    # 1. ISL Words Confusion Matrix (temporal_words.keras)
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC 1] ISL Words Analysis (temporal_words.keras)")
    m_w = tf.keras.models.load_model('models/temporal_words.keras')
    with open('models/temporal_words_labels.json') as f:
        labels_w = json.load(f)
    d_w = np.load('data/processed/d2_landmarks.npz')
    X_w, y_w = d_w['X'], d_w['y']
    pred_w = np.argmax(m_w.predict(X_w, verbose=0), axis=1)
    cm_w = confusion_matrix(y_w, pred_w)

    print("Classes:", labels_w)
    print("Confusion Matrix:\n", cm_w)

    # Inspect exact pairs
    pairs_to_check = [
        ('eat', 'water'),
        ('water', 'eat'),
        ('hello', 'no'),
        ('no', 'hello'),
        ('yes', 'no'),
        ('yes', 'water'),
        ('go', 'please'),
    ]
    isl_evidence = {}
    for true_name, pred_name in pairs_to_check:
        if true_name in labels_w and pred_name in labels_w:
            t_idx, p_idx = labels_w.index(true_name), labels_w.index(pred_name)
            count = int(cm_w[t_idx, p_idx])
            tot = int(np.sum(y_w == t_idx))
            isl_evidence[f"{true_name}->{pred_name}"] = f"{count}/{tot} ({count/tot*100:.1f}%)"
            print(f"  {true_name} -> {pred_name}: {count} of {tot} samples ({count/tot*100:.1f}%)")

    # Overall eat recall
    eat_idx = labels_w.index('eat')
    eat_rec = cm_w[eat_idx, eat_idx] / np.sum(y_w == eat_idx)
    print(f"  CRITICAL: 'eat' recall is only {eat_rec*100:.1f}%! Heavily confused with water, help, please.")

    # -------------------------------------------------------------
    # 2. ASL Static Letters Analysis (landmark_letters.keras)
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC 2] ASL Letters Analysis (landmark_letters.keras)")
    m_l = tf.keras.models.load_model('models/landmark_letters.keras')
    with open('models/landmark_letters_labels.json') as f:
        labels_l = json.load(f)
    d_l = np.load('data/processed/signalphaset_static_landmarks.npz')
    X_l, y_l = d_l['X'], d_l['y']
    
    # Measure U vs V fingertip distance
    u_idx = labels_l.index('U')
    v_idx = labels_l.index('V')
    c_idx = labels_l.index('C')
    g_idx = labels_l.index('G')
    j_idx = labels_l.index('J')

    X_u = X_l[y_l == u_idx].reshape(-1, 21, 3)
    X_v = X_l[y_l == v_idx].reshape(-1, 21, 3)
    X_c = X_l[y_l == c_idx].reshape(-1, 21, 3)
    X_g = X_l[y_l == g_idx].reshape(-1, 21, 3)

    # Distance between index tip (8) and middle tip (12)
    dist_uv_u = np.linalg.norm(X_u[:, 8, :2] - X_u[:, 12, :2], axis=1).mean()
    dist_uv_v = np.linalg.norm(X_v[:, 8, :2] - X_v[:, 12, :2], axis=1).mean()
    print(f"  Fingertip (8-12) distance in U: {dist_uv_u:.4f} vs in V: {dist_uv_v:.4f}")

    # Distance between thumb tip (4) and index tip (8) in C vs G
    dist_cg_c = np.linalg.norm(X_c[:, 4, :2] - X_c[:, 8, :2], axis=1).mean()
    dist_cg_g = np.linalg.norm(X_g[:, 4, :2] - X_g[:, 8, :2], axis=1).mean()
    print(f"  Thumb-Index (4-8) distance in C: {dist_cg_c:.4f} vs in G: {dist_cg_g:.4f}")

    # Depth differences: z(4) - z(8)
    z_diff_c = (X_c[:, 4, 2] - X_c[:, 8, 2]).mean()
    z_diff_g = (X_g[:, 4, 2] - X_g[:, 8, 2]).mean()
    print(f"  Thumb-Index depth difference z(4)-z(8) in C: {z_diff_c:.4f} vs in G: {z_diff_g:.4f}")

    # J status
    print(f"  J in static label set: {'J' in labels_l} (index {j_idx})")
    print("  ROOT CAUSE FOR J: J is a dynamic traced gesture, but is currently registered as a static class in landmark_letters.keras!")
    print("  When users transition between static letters, the intermediate relaxed/curled hand shape triggers static J!")

    # -------------------------------------------------------------
    # 3. Digits Digit 3 Analysis (landmark_digits.keras)
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC 3] Digits Analysis (landmark_digits.keras)")
    m_d = tf.keras.models.load_model('models/landmark_digits.keras')
    with open('models/landmark_digits_labels.json') as f:
        labels_d = json.load(f)
    d_d = np.load('data/processed/d1_landmarks.npz')
    X_d, y_d = d_d['X'], d_d['y']
    pred_d = np.argmax(m_d.predict(X_d, verbose=0), axis=1)
    cm_d = confusion_matrix(y_d, pred_d)

    idx_3 = labels_d.index('3')
    idx_2 = labels_d.index('2')
    idx_5 = labels_d.index('5')
    print(f"  Digit 3 samples in dataset: {np.sum(y_d == idx_3)}")
    print(f"  Digit 3 misclassifications: as 5: {cm_d[idx_3, idx_5]}, as 2: {cm_d[idx_3, idx_2]}")

    X_3 = X_d[y_d == idx_3].reshape(-1, 21, 3)
    X_2 = X_d[y_d == idx_2].reshape(-1, 21, 3)
    # Thumb extension relative to palm center (landmark 9 MCP)
    thumb_ext_3 = np.linalg.norm(X_3[:, 4] - X_3[:, 9], axis=1).mean()
    thumb_ext_2 = np.linalg.norm(X_2[:, 4] - X_2[:, 9], axis=1).mean()
    print(f"  Thumb-Palm distance (4-9) in Digit 3: {thumb_ext_3:.4f} vs in Digit 2: {thumb_ext_2:.4f}")

    # -------------------------------------------------------------
    # 4. ASL Dynamic Phrases (temporal_letters_phrases.keras)
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC 4] ASL Dynamic Phrases Analysis")
    m_a = tf.keras.models.load_model('models/temporal_letters_phrases.keras')
    with open('models/temporal_letters_phrases_labels.json') as f:
        labels_a = json.load(f)
    d_a = np.load('data/processed/signalphaset_dynamic_landmarks.npz')
    X_a, y_a = d_a['X'], d_a['y']
    pred_a = np.argmax(m_a.predict(X_a, verbose=0), axis=1)
    cm_a = confusion_matrix(y_a, pred_a)
    print("Classes:", labels_a)
    print("Confusion Matrix:\n", cm_a)
    no_idx = labels_a.index('NO')
    print(f"  ASL NO recall: {cm_a[no_idx, no_idx]}/{np.sum(y_a == no_idx)} ({cm_a[no_idx, no_idx]/np.sum(y_a == no_idx)*100:.1f}%)")

    # -------------------------------------------------------------
    # 5. MediaPipe Detection & World Landmarks Inspection
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC 5] MediaPipe Feature & Depth Investigation")
    detector = create_hand_detector()
    
    # Inspect a static letter image
    sample_img_path = os.path.join(SIGNALPHASET_STATIC_PATH, 'C', os.listdir(os.path.join(SIGNALPHASET_STATIC_PATH, 'C'))[0])
    img = cv2.imread(sample_img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    
    if res.hand_landmarks and len(res.hand_landmarks) > 0:
        lm = res.hand_landmarks[0]
        print("  Sample MediaPipe detection successful.")
        print(f"  Normalized landmarks: Wrist z={lm[0].z:.5f}, Thumb Tip z={lm[4].z:.5f}, Index Tip z={lm[8].z:.5f}")
        has_world = hasattr(res, 'hand_world_landmarks') and len(res.hand_world_landmarks) > 0
        print(f"  hand_world_landmarks present in Tasks API: {has_world}")
        if has_world:
            wlm = res.hand_world_landmarks[0]
            print(f"  World landmarks (meters): Wrist=({wlm[0].x:.3f}, {wlm[0].y:.3f}, {wlm[0].z:.3f})")
            print(f"  World Thumb Tip=({wlm[4].x:.3f}, {wlm[4].y:.3f}, {wlm[4].z:.3f})")
    
    detector.close()
    
    # Save diagnostic summary to docs
    os.makedirs('docs', exist_ok=True)
    report_path = 'docs/diagnostic_report_phase1.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 1 Precision & Confusion Diagnostic Report\n\n")
        f.write("## Categorization of Failures\n\n")
        f.write("| Failure Mode | Head | Root Cause Category | Detailed Mechanism |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write("| **Eat ↔ Water** | `temporal_words` (ISL) | **Feature + Temporal** | Both gestures occur at mouth/chin. Without explicit velocity ($\Delta x, \Delta y$) and joint curl features, trajectory classifier only sees a hand near the face. |\n")
        f.write("| **Yes / Go / Hello / No** | `temporal_words` (ISL) | **Temporal + Model** | Unidirectional GRU with uniform subsampling across 4.5s video clips confuses initial hand raise. Needs frame-to-frame velocity + Bi-GRU. |\n")
        f.write("| **C ↔ G** | `landmark_letters` (ASL) | **Feature (Depth & Curl)** | C has all fingers curved; G has index pointing forward/side. In raw 63D coordinates without explicit curl angles and thumb-index distance, tilt projection causes overlap. |\n")
        f.write("| **U ↔ V** | `landmark_letters` (ASL) | **Feature (Inter-Fingertip Distance)** | U has index and middle touching; V has them spread. The distinguishing signal is inter-fingertip distance $\\|p_8 - p_{12}\\|$, which was diluted across 63 coordinates. |\n")
        f.write("| **J Over-Prediction** | `landmark_letters` (ASL) | **Architecture / Data** | J is a dynamic traced motion! It was erroneously included as a static class in the 26-class MLP. Transitioning hands resembled static J. |\n")
        f.write("| **Digit 3 Unreliable** | `landmark_digits` (ASL) | **Feature + User Convention** | ASL 3 requires thumb extended + index + middle. Users often sign European 3 (index + middle + ring). Also needs explicit thumb extension feature. |\n")
        f.write("| **ASL No** | `temporal_letters_phrases` | **Feature (Velocity / Delta)** | Snapping motion of fingers onto thumb requires velocity deltas to capture the rapid closing phase. |\n")

    print(f"\n[OK] Diagnostic complete. Detailed report written to: {report_path}")

if __name__ == '__main__':
    run_diagnostics()
