"""
Dataset loaders for D1 (static digit images) and D2 (ISL word videos).
Extracts landmarks and caches them as .npz files for fast reloading.
"""
import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from .preprocess import create_hand_detector, extract_landmarks_from_image, extract_landmarks_from_video
from .config import (
    D1_PATH, D2_PATH, SIGNALPHASET_STATIC_PATH, SIGNALPHASET_DYNAMIC_PATH,
    PROCESSED_DIR, TEMPORAL_SEQUENCE_LENGTH
)


def load_d1_landmarks(force_reprocess=False):
    """
    Load D1 digit dataset landmarks.
    Caches to data/processed/d1_landmarks.npz.
    
    Returns:
        X: numpy array (N, 63)
        y: numpy array (N,) integer labels 0-9
        label_names: list of string labels
    """
    cache_path = os.path.join(PROCESSED_DIR, "d1_landmarks.npz")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    if os.path.exists(cache_path) and not force_reprocess:
        data = np.load(cache_path)
        return data['X'], data['y'], [str(i) for i in range(10)]
    
    print("Extracting D1 landmarks (this takes a few minutes)...")
    hands = create_hand_detector(static_image_mode=True, min_detection_confidence=0.3)
    
    X_list = []
    y_list = []
    label_names = [str(i) for i in range(10)]
    
    d1_path = os.path.normpath(D1_PATH)
    
    for class_idx, class_name in enumerate(label_names):
        class_dir = os.path.join(d1_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: Class directory not found: {class_dir}")
            continue
        
        files = sorted([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPG'))])
        
        detected = 0
        for fname in files:
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            lm = extract_landmarks_from_image(img, hands)
            if lm is not None:
                X_list.append(lm)
                y_list.append(class_idx)
                detected += 1
        
        print(f"  Class {class_name}: {detected}/{len(files)} images had hand detected")
    
    hands.close()
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    np.savez(cache_path, X=X, y=y)
    print(f"D1 landmarks cached: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, label_names


def load_d2_landmarks(force_reprocess=False):
    """
    Load D2 ISL word video landmarks.
    Caches to data/processed/d2_landmarks.npz.
    
    Returns:
        X: numpy array (N, TEMPORAL_SEQUENCE_LENGTH, 63)
        y: numpy array (N,)
        groups: numpy array (N,) video IDs for GroupShuffleSplit
        label_names: list of string class names
    """
    cache_path = os.path.join(PROCESSED_DIR, "d2_landmarks.npz")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    if os.path.exists(cache_path) and not force_reprocess:
        data = np.load(cache_path)
        return data['X'], data['y'], data['groups'], list(data['label_names'])
    
    print("Extracting D2 video landmarks...")
    hands = create_hand_detector(static_image_mode=False, min_detection_confidence=0.3)
    
    d2_path = os.path.normpath(D2_PATH)
    
    # Discover classes from directory names
    class_dirs = sorted([d for d in os.listdir(d2_path) 
                         if os.path.isdir(os.path.join(d2_path, d)) and d != '.cache'])
    label_names = class_dirs
    
    X_list = []
    y_list = []
    group_list = []
    video_id = 0
    
    for class_idx, class_name in enumerate(label_names):
        class_dir = os.path.join(d2_path, class_name)
        videos = sorted([f for f in os.listdir(class_dir) if f.lower().endswith('.mp4')])
        
        for vname in videos:
            video_path = os.path.join(class_dir, vname)
            seq = extract_landmarks_from_video(video_path, hands, num_frames=TEMPORAL_SEQUENCE_LENGTH)
            
            if seq is not None:
                X_list.append(seq)
                y_list.append(class_idx)
                group_list.append(video_id)
                video_id += 1
        
        print(f"  Class '{class_name}': {len(videos)} videos processed")
    
    hands.close()
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(group_list, dtype=np.int32)
    
    np.savez(cache_path, X=X, y=y, groups=groups, label_names=np.array(label_names))
    print(f"D2 landmarks cached: {X.shape[0]} sequences of shape {X.shape[1:]}")
    
    return X, y, groups, label_names


def split_d1_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """Split D1 data into train/val/test sets."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_state, stratify=y_trainval
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_d2_data(X, y, groups, test_size=0.3, random_state=42):
    """
    Split D2 by video file (group-aware split) to prevent data leakage.
    With only 7 videos per class, we do a simple train/val split.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def load_letters_landmarks(max_per_class=300, force_reprocess=False):
    """
    Load SignAlphaSet static letters (A-Z) landmarks.
    Caches to data/processed/signalphaset_static_landmarks.npz.
    
    Returns:
        X: numpy array (N, 63)
        y: numpy array (N,) integer labels 0-25
        label_names: list of string labels ['A'...'Z']
    """
    cache_path = os.path.join(PROCESSED_DIR, "signalphaset_static_landmarks.npz")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    label_names = [chr(ord('A') + i) for i in range(26)]
    
    if os.path.exists(cache_path) and not force_reprocess:
        data = np.load(cache_path)
        return data['X'], data['y'], list(data['label_names'])
    
    print(f"Extracting SignAlphaSet static landmarks (up to {max_per_class} per class)...")
    hands = create_hand_detector(static_image_mode=True, min_detection_confidence=0.3)
    
    static_path = os.path.normpath(SIGNALPHASET_STATIC_PATH)
    
    X_list = []
    y_list = []
    
    for class_idx, class_name in enumerate(label_names):
        class_dir = os.path.join(static_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: Letter directory not found: {class_dir}")
            continue
        
        files = sorted([f for f in os.listdir(class_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        # Subsample to max_per_class
        if max_per_class and len(files) > max_per_class:
            indices = np.linspace(0, len(files) - 1, max_per_class, dtype=int)
            files = [files[i] for i in indices]
        
        detected = 0
        for fname in files:
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            lm = extract_landmarks_from_image(img, hands)
            if lm is not None:
                X_list.append(lm)
                y_list.append(class_idx)
                detected += 1
        
        print(f"  Letter {class_name}: {detected}/{len(files)} images extracted", flush=True)
    
    hands.close()
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    np.savez(cache_path, X=X, y=y, label_names=np.array(label_names))
    print(f"SignAlphaSet static cached: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, label_names


def load_dynamic_asl_landmarks(classes=None, force_reprocess=False):
    """
    Load dynamic ASL phrase gesture landmarks.
    Caches to data/processed/signalphaset_dynamic_landmarks.npz.
    
    Returns:
        X: numpy array (N, TEMPORAL_SEQUENCE_LENGTH, 63)
        y: numpy array (N,)
        groups: numpy array (N,) video/clip IDs for GroupShuffleSplit
        label_names: list of string class names
    """
    cache_path = os.path.join(PROCESSED_DIR, "signalphaset_dynamic_landmarks.npz")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    if classes is None:
        classes = ["HELLO", "NO", "SORRY", "THANKYOU", "YES"]
    
    if os.path.exists(cache_path) and not force_reprocess:
        data = np.load(cache_path)
        return data['X'], data['y'], data['groups'], list(data['label_names'])
    
    print(f"Extracting SignAlphaSet dynamic landmarks for {classes}...")
    hands = create_hand_detector(static_image_mode=False, min_detection_confidence=0.3)
    
    dynamic_path = os.path.normpath(SIGNALPHASET_DYNAMIC_PATH)
    
    X_list = []
    y_list = []
    group_list = []
    clip_id = 0
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dynamic_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: Dynamic class directory not found: {class_dir}")
            continue
        
        # Check for pre-extracted frame folders
        frame_dirs = sorted([d for d in os.listdir(class_dir)
                             if os.path.isdir(os.path.join(class_dir, d)) and d.endswith('_frames')])
        
        if frame_dirs:
            for fdir in frame_dirs:
                frame_dir_path = os.path.join(class_dir, fdir)
                frame_files = sorted([f for f in os.listdir(frame_dir_path) if f.lower().endswith(('.jpg', '.png'))])
                
                if not frame_files:
                    continue
                
                # Sample 32 frames uniformly
                if len(frame_files) >= TEMPORAL_SEQUENCE_LENGTH:
                    indices = np.linspace(0, len(frame_files) - 1, TEMPORAL_SEQUENCE_LENGTH, dtype=int)
                else:
                    indices = np.arange(len(frame_files))
                
                seq = []
                for idx in indices:
                    img = cv2.imread(os.path.join(frame_dir_path, frame_files[idx]))
                    lm = extract_landmarks_from_image(img, hands) if img is not None else None
                    if lm is not None:
                        seq.append(lm)
                    else:
                        seq.append(np.zeros(63, dtype=np.float32))
                
                while len(seq) < TEMPORAL_SEQUENCE_LENGTH:
                    seq.append(np.zeros(63, dtype=np.float32))
                
                X_list.append(np.array(seq[:TEMPORAL_SEQUENCE_LENGTH], dtype=np.float32))
                y_list.append(class_idx)
                group_list.append(clip_id)
                clip_id += 1
            print(f"  Class '{class_name}': {len(frame_dirs)} frame sequences processed", flush=True)
        else:
            # Fallback to source videos if no pre-extracted frame folders
            videos = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.avi', '.mp4'))])
            for vname in videos:
                video_path = os.path.join(class_dir, vname)
                seq = extract_landmarks_from_video(video_path, hands, num_frames=TEMPORAL_SEQUENCE_LENGTH)
                if seq is not None:
                    X_list.append(seq)
                    y_list.append(class_idx)
                    group_list.append(clip_id)
                    clip_id += 1
            print(f"  Class '{class_name}': {len(videos)} videos processed via VideoCapture", flush=True)
    
    hands.close()
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(group_list, dtype=np.int32)
    
    np.savez(cache_path, X=X, y=y, groups=groups, label_names=np.array(classes))
    print(f"Dynamic ASL landmarks cached: {X.shape[0]} sequences of shape {X.shape[1:]}")
    
    return X, y, groups, classes

