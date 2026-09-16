"""
Comprehensive Verification Script for Landmark Precision & Confusion-Pair Fixes.
Directly re-evaluates all failure cases from Section 2 against the saved models.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from ml.preprocess import compute_precision_features, compute_temporal_precision_features
from ml.dataset import (
    load_letters_landmarks, load_d1_landmarks,
    load_d2_landmarks, load_dynamic_asl_landmarks,
    split_d1_data, split_d2_data
)
from ml.config import MODEL_DIR

def verify_all():
    results = {}
    print("=" * 70)
    print("SIGNBRIDGE PRECISION VERIFICATION SUITE")
    print("=" * 70)
    
    # ----------------------------------------------------
    # 1. Letters: C vs G, U vs V, and J exclusion
    # ----------------------------------------------------
    print("\n[1/4] Verifying ASL Letters (C vs G, U vs V, J-exclusion)...")
    letters_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "landmark_letters.keras"))
    with open(os.path.join(MODEL_DIR, "landmark_letters_labels.json")) as f:
        letters_labels = json.load(f)
        
    X_raw, y_raw, label_names_raw = load_letters_landmarks()
    j_idx = label_names_raw.index('J') if 'J' in label_names_raw else -1
    mask = (y_raw != j_idx)
    X_filtered = X_raw[mask]
    y_filtered = y_raw[mask]
    y_remapped = np.array([idx if idx < j_idx else idx - 1 for idx in y_filtered], dtype=np.int32)
    
    X_feats = np.array([compute_precision_features(sample) for sample in X_filtered], dtype=np.float32)
    _, _, X_test, _, _, y_test = split_d1_data(X_feats, y_remapped, random_state=42)
    
    y_pred = np.argmax(letters_model.predict(X_test, verbose=0), axis=1)
    cm_letters = confusion_matrix(y_test, y_pred)
    
    c_idx = letters_labels.index('C')
    g_idx = letters_labels.index('G')
    u_idx = letters_labels.index('U')
    v_idx = letters_labels.index('V')
    
    results['letters'] = {
        "c_to_g": int(cm_letters[c_idx, g_idx]),
        "g_to_c": int(cm_letters[g_idx, c_idx]),
        "u_to_v": int(cm_letters[u_idx, v_idx]),
        "v_to_u": int(cm_letters[v_idx, u_idx]),
        "j_in_static_labels": "J" in letters_labels,
        "overall_acc": float(np.mean(y_pred == y_test))
    }
    print(f"  C -> G errors: {results['letters']['c_to_g']} / {cm_letters[c_idx].sum()}")
    print(f"  G -> C errors: {results['letters']['g_to_c']} / {cm_letters[g_idx].sum()}")
    print(f"  U -> V errors: {results['letters']['u_to_v']} / {cm_letters[u_idx].sum()}")
    print(f"  V -> U errors: {results['letters']['v_to_u']} / {cm_letters[v_idx].sum()}")
    print(f"  J in static classifier: {results['letters']['j_in_static_labels']} (Target: False)")
    print(f"  Overall Letters Test Acc: {results['letters']['overall_acc'] * 100:.2f}%")
    
    # ----------------------------------------------------
    # 2. Digits: Digit 3 Registration
    # ----------------------------------------------------
    print("\n[2/4] Verifying ASL Digits (Digit 3 Registration)...")
    digits_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "landmark_digits.keras"))
    with open(os.path.join(MODEL_DIR, "landmark_digits_labels.json")) as f:
        digits_labels = json.load(f)
        
    X_raw, y_raw, label_names = load_d1_landmarks()
    X_feats = np.array([compute_precision_features(sample) for sample in X_raw], dtype=np.float32)
    _, _, X_test, _, _, y_test = split_d1_data(X_feats, y_raw, random_state=42)
    
    y_pred = np.argmax(digits_model.predict(X_test, verbose=0), axis=1)
    cm_digits = confusion_matrix(y_test, y_pred)
    idx_3 = digits_labels.index('3')
    
    total_3 = int(cm_digits[idx_3].sum())
    correct_3 = int(cm_digits[idx_3, idx_3])
    results['digits'] = {
        "digit_3_correct": correct_3,
        "digit_3_total": total_3,
        "digit_3_recall": correct_3 / total_3,
        "overall_acc": float(np.mean(y_pred == y_test))
    }
    print(f"  Digit 3 Correct: {correct_3} / {total_3} ({results['digits']['digit_3_recall'] * 100:.1f}%)")
    print(f"  Overall Digits Test Acc: {results['digits']['overall_acc'] * 100:.2f}%")
    
    # ----------------------------------------------------
    # 3. ISL Words: Eat vs Water, Hello vs No, Yes / Go / Hello
    # ----------------------------------------------------
    print("\n[3/4] Verifying ISL Words (Eat vs Water, Hello vs No, Cluster)...")
    words_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "temporal_words.keras"))
    with open(os.path.join(MODEL_DIR, "temporal_words_labels.json")) as f:
        words_labels = json.load(f)
        
    X_raw, y_raw, groups, _ = load_d2_landmarks()
    X_feats = np.array([compute_temporal_precision_features(seq) for seq in X_raw], dtype=np.float32)
    _, X_val, _, y_val = split_d2_data(X_feats, y_raw, groups, test_size=0.25, random_state=42)
    
    y_pred = np.argmax(words_model.predict(X_val, verbose=0), axis=1)
    cm_words = confusion_matrix(y_val, y_pred, labels=range(len(words_labels)))
    
    eat_idx = words_labels.index('eat')
    water_idx = words_labels.index('water')
    hello_idx = words_labels.index('hello')
    no_idx = words_labels.index('no')
    yes_idx = words_labels.index('yes')
    go_idx = words_labels.index('go')
    
    results['words'] = {
        "eat_to_water": int(cm_words[eat_idx, water_idx]),
        "water_to_eat": int(cm_words[water_idx, eat_idx]),
        "hello_to_no": int(cm_words[hello_idx, no_idx]),
        "no_to_hello": int(cm_words[no_idx, hello_idx]),
        "yes_to_go_hello": int(cm_words[yes_idx, go_idx] + cm_words[yes_idx, hello_idx]),
        "val_acc": float(np.mean(y_pred == y_val))
    }
    print(f"  Eat -> Water errors: {results['words']['eat_to_water']} / {cm_words[eat_idx].sum()}")
    print(f"  Water -> Eat errors: {results['words']['water_to_eat']} / {cm_words[water_idx].sum()}")
    print(f"  Hello -> No errors: {results['words']['hello_to_no']} / {cm_words[hello_idx].sum()}")
    print(f"  No -> Hello errors: {results['words']['no_to_hello']} / {cm_words[no_idx].sum()}")
    print(f"  Yes -> Go/Hello errors: {results['words']['yes_to_go_hello']} / {cm_words[yes_idx].sum()}")
    print(f"  Overall Words Val Acc: {results['words']['val_acc'] * 100:.2f}%")
    
    # ----------------------------------------------------
    # 4. ASL Dynamic Phrases: NO class
    # ----------------------------------------------------
    print("\n[4/4] Verifying ASL Dynamic Phrases (NO Gesture)...")
    asl_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "temporal_letters_phrases.keras"))
    with open(os.path.join(MODEL_DIR, "temporal_letters_phrases_labels.json")) as f:
        asl_labels = json.load(f)
        
    X_raw, y_raw, groups, _ = load_dynamic_asl_landmarks(classes=["HELLO", "NO", "SORRY", "THANKYOU", "YES"])
    X_feats = np.array([compute_temporal_precision_features(seq) for seq in X_raw], dtype=np.float32)
    _, X_val, _, y_val = split_d2_data(X_feats, y_raw, groups, test_size=0.3, random_state=42)
    
    y_pred = np.argmax(asl_model.predict(X_val, verbose=0), axis=1)
    cm_asl = confusion_matrix(y_val, y_pred, labels=range(len(asl_labels)))
    no_idx = asl_labels.index('NO')
    
    total_no = int(cm_asl[no_idx].sum())
    correct_no = int(cm_asl[no_idx, no_idx])
    results['asl_gestures'] = {
        "no_correct": correct_no,
        "no_total": total_no,
        "no_recall": correct_no / (total_no + 1e-7),
        "val_acc": float(np.mean(y_pred == y_val))
    }
    print(f"  ASL 'NO' Correct: {correct_no} / {total_no} ({results['asl_gestures']['no_recall'] * 100:.1f}%)")
    print(f"  Overall ASL Gestures Val Acc: {results['asl_gestures']['val_acc'] * 100:.2f}%")
    
    print("\n" + "=" * 70)
    print("ALL VERIFICATIONS COMPLETE — SUMMARY")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    
    # Save verification report
    report_path = os.path.join(os.path.dirname(__file__), "..", "docs", "precision_verification_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Report written to {report_path}")

if __name__ == "__main__":
    verify_all()
