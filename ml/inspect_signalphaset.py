"""
Inspect SignAlphaSet static and dynamic datasets.
Measures file counts, dimensions, corruptions, MediaPipe detection rate,
and video/frame alignment.
"""
import os
import sys
import glob
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.config import SIGNALPHASET_STATIC_PATH, SIGNALPHASET_DYNAMIC_PATH
from ml.preprocess import create_hand_detector, extract_landmarks_from_image

def inspect_datasets():
    print("=" * 70)
    print("SIGNALPHASET DATASET INSPECTION")
    print("=" * 70)
    
    static_path = os.path.normpath(SIGNALPHASET_STATIC_PATH)
    dynamic_path = os.path.normpath(SIGNALPHASET_DYNAMIC_PATH)
    
    print(f"Static path: {static_path} (exists: {os.path.exists(static_path)})")
    print(f"Dynamic path: {dynamic_path} (exists: {os.path.exists(dynamic_path)})")
    
    # 1. Static Dataset Inspection
    print("\n--- 1. STATIC DATASET (A-Z) ---")
    letters = sorted([d for d in os.listdir(static_path) if os.path.isdir(os.path.join(static_path, d))])
    print(f"Found {len(letters)} letter folders: {letters}")
    
    detector = create_hand_detector(static_image_mode=True, min_detection_confidence=0.3)
    
    static_stats = {}
    total_static_images = 0
    sample_dims = set()
    corrupt_count = 0
    
    for letter in letters:
        folder = os.path.join(static_path, letter)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_static_images += len(files)
        
        # Test detection rate on 25 sample images per letter
        sample_files = files[:25]
        detected = 0
        valid_images = 0
        for f in sample_files:
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path)
            if img is None:
                corrupt_count += 1
                continue
            valid_images += 1
            sample_dims.add(img.shape)
            
            lm = extract_landmarks_from_image(img, detector)
            if lm is not None:
                detected += 1
                
        det_rate = (detected / valid_images * 100) if valid_images > 0 else 0
        static_stats[letter] = {
            "total_files": len(files),
            "sample_tested": valid_images,
            "detected": detected,
            "detection_rate_pct": round(det_rate, 1)
        }
        print(f"Letter {letter}: {len(files)} files | Detection: {detected}/{valid_images} ({det_rate:.1f}%)", flush=True)
        
    print(f"\nTotal Static Images: {total_static_images}")
    print(f"Image Dimensions observed: {sample_dims}")
    print(f"Corrupt images encountered: {corrupt_count}")
    
    # 2. Dynamic Dataset Inspection
    print("\n--- 2. DYNAMIC DATASET (31 Classes) ---")
    dynamic_classes = sorted([d for d in os.listdir(dynamic_path) if os.path.isdir(os.path.join(dynamic_path, d))])
    print(f"Found {len(dynamic_classes)} dynamic class folders: {dynamic_classes}")
    
    dynamic_stats = {}
    total_videos = 0
    total_frame_dirs = 0
    
    for c in dynamic_classes:
        folder = os.path.join(dynamic_path, c)
        videos = [f for f in os.listdir(folder) if f.lower().endswith(('.avi', '.mp4'))]
        frame_dirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and d.endswith('_frames')]
        
        total_videos += len(videos)
        total_frame_dirs += len(frame_dirs)
        
        # Sample first video stats
        v_fps, v_dur, v_res = 0, 0, (0, 0)
        v_frames_count = 0
        if videos:
            v_path = os.path.join(folder, videos[0])
            cap = cv2.VideoCapture(v_path)
            if cap.isOpened():
                v_fps = cap.get(cv2.CAP_PROP_FPS)
                v_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                v_res = (w, h)
                v_dur = round(v_frames_count / v_fps, 2) if v_fps > 0 else 0
                cap.release()
                
        # Sample pre-extracted frames count
        p_frames_count = 0
        if frame_dirs:
            p_frames = [f for f in os.listdir(os.path.join(folder, frame_dirs[0])) if f.endswith('.jpg')]
            p_frames_count = len(p_frames)
            
        dynamic_stats[c] = {
            "videos": len(videos),
            "frame_dirs": len(frame_dirs),
            "fps": v_fps,
            "duration_s": v_dur,
            "resolution": v_res,
            "video_frames": v_frames_count,
            "extracted_frames": p_frames_count,
            "aligned": (v_frames_count == p_frames_count)
        }
        print(f"Class '{c}': {len(videos)} videos, {len(frame_dirs)} frame dirs | Vid Frames: {v_frames_count}, Extracted: {p_frames_count} | Dur: {v_dur}s @ {v_fps}fps")
        
    detector.close()
    
    return {
        "static_stats": static_stats,
        "total_static": total_static_images,
        "dynamic_stats": dynamic_stats,
        "total_dynamic_videos": total_videos
    }

if __name__ == "__main__":
    inspect_datasets()
