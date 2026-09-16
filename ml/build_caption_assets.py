"""
Build sign caption assets from the datasets.
- For digits (D1): Select highest-landmark-confidence exemplar per class, save as PNG.
- For words (D2): Extract a representative frame from each video, save as PNG.
"""
import os
import sys
import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.preprocess import create_hand_detector
from ml.config import D1_PATH, D2_PATH, SIGNALPHASET_STATIC_PATH, SIGNALPHASET_DYNAMIC_PATH


def build_digit_assets(output_dir):
    """Select best exemplar image per digit class and save as clean PNG."""
    print("Building digit caption assets...")
    hands = create_hand_detector(static_image_mode=True, min_detection_confidence=0.3)
    
    d1_path = os.path.normpath(D1_PATH)
    
    for digit in range(10):
        class_dir = os.path.join(d1_path, str(digit))
        if not os.path.isdir(class_dir):
            print(f"  Class dir not found: {class_dir}")
            continue
        
        best_img = None
        best_conf = -1
        
        files = sorted([f for f in os.listdir(class_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPG'))])
        
        # Check up to 30 images to find best
        for fname in files[:30]:
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = hands.detect(mp_image)
            
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                hand = results.hand_landmarks[0]
                zs = [lm.z for lm in hand]
                quality = np.std(zs)
                
                if quality > best_conf:
                    best_conf = quality
                    best_img = img.copy()
            elif best_img is None:
                # Fallback if no landmarks detected in first frame
                best_img = img.copy()
        
        if best_img is not None:
            # Resize to 200x200 with padding
            h, w = best_img.shape[:2]
            scale = 200.0 / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(best_img, (new_w, new_h))
            
            # Add white padding
            canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
            y_off = (200 - new_h) // 2
            x_off = (200 - new_w) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            
            out_path = os.path.join(output_dir, f"{digit}.png")
            cv2.imwrite(out_path, canvas)
            print(f"  Digit {digit} saved")
        else:
            print(f"  No good exemplar for digit {digit}")
    
    hands.close()


def build_word_assets(output_dir):
    """Extract a representative frame from each word video class."""
    print("Building word caption assets...")
    hands = create_hand_detector(static_image_mode=True, min_detection_confidence=0.3)
    
    d2_path = os.path.normpath(D2_PATH)
    
    class_dirs = sorted([d for d in os.listdir(d2_path)
                         if os.path.isdir(os.path.join(d2_path, d)) and d != '.cache'])
    
    for class_name in class_dirs:
        class_dir = os.path.join(d2_path, class_name)
        videos = sorted([f for f in os.listdir(class_dir) if f.lower().endswith('.mp4')])
        
        if not videos:
            print(f"  No videos for {class_name}")
            continue
        
        best_frame = None
        best_quality = -1
        
        # Check first 3 videos
        for vname in videos[:3]:
            video_path = os.path.join(class_dir, vname)
            cap = cv2.VideoCapture(video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                continue
            
            mid_frames = [total_frames // 3, total_frames // 2, 2 * total_frames // 3]
            
            for frame_idx in mid_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = hands.detect(mp_image)
                
                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    hand = results.hand_landmarks[0]
                    zs = [lm.z for lm in hand]
                    quality = float(np.std(zs))
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_frame = frame.copy()
                elif best_frame is None:
                    best_frame = frame.copy()
            
            cap.release()
        
        if best_frame is not None:
            h, w = best_frame.shape[:2]
            scale = 200.0 / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(best_frame, (new_w, new_h))
            
            canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
            y_off = (200 - new_h) // 2
            x_off = (200 - new_w) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            
            out_path = os.path.join(output_dir, f"{class_name.lower()}.png")
            cv2.imwrite(out_path, canvas)
            print(f"  {class_name} saved")
        else:
            print(f"  No good frame for {class_name}")
    
    hands.close()


def build_letter_assets(output_dir):
    """Select best exemplar image for each letter A-Z from SignAlphaSet static."""
    print("Building letter caption assets (A-Z)...")
    hands = create_hand_detector(static_image_mode=True, min_detection_confidence=0.3)
    
    static_path = os.path.normpath(SIGNALPHASET_STATIC_PATH)
    letters = [chr(ord('A') + i) for i in range(26)]
    
    for letter in letters:
        class_dir = os.path.join(static_path, letter)
        if not os.path.isdir(class_dir):
            continue
        
        best_img = None
        best_conf = -1
        files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png'))])
        
        for fname in files[:20]:
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = hands.detect(mp_image)
            
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                hand = results.hand_landmarks[0]
                zs = [lm.z for lm in hand]
                quality = np.std(zs)
                if quality > best_conf:
                    best_conf = quality
                    best_img = img.copy()
            elif best_img is None:
                best_img = img.copy()
                
        if best_img is not None:
            h, w = best_img.shape[:2]
            scale = 200.0 / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(best_img, (new_w, new_h))
            
            canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
            y_off = (200 - new_h) // 2
            x_off = (200 - new_w) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            
            out_path = os.path.join(output_dir, f"{letter.lower()}.png")
            cv2.imwrite(out_path, canvas)
            print(f"  Letter {letter} saved", flush=True)
            
    hands.close()


def build_asl_phrase_assets(output_dir):
    """Extract representative frames for dynamic ASL phrase gestures."""
    print("Building dynamic ASL phrase assets...")
    hands = create_hand_detector(static_image_mode=True, min_detection_confidence=0.3)
    
    dynamic_path = os.path.normpath(SIGNALPHASET_DYNAMIC_PATH)
    phrases = ["HELLO", "NO", "SORRY", "THANKYOU", "YES"]
    
    for phrase in phrases:
        class_dir = os.path.join(dynamic_path, phrase)
        if not os.path.isdir(class_dir):
            continue
        
        # Look for pre-extracted frames first
        frame_dirs = sorted([d for d in os.listdir(class_dir)
                             if os.path.isdir(os.path.join(class_dir, d)) and d.endswith('_frames')])
        best_frame = None
        best_quality = -1
        
        if frame_dirs:
            for fdir in frame_dirs[:3]:
                fdir_path = os.path.join(class_dir, fdir)
                frames = sorted([f for f in os.listdir(fdir_path) if f.lower().endswith(('.jpg', '.png'))])
                if not frames:
                    continue
                mid_indices = [len(frames) // 3, len(frames) // 2, 2 * len(frames) // 3]
                for idx in mid_indices:
                    fpath = os.path.join(fdir_path, frames[idx])
                    frame = cv2.imread(fpath)
                    if frame is None:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    results = hands.detect(mp_image)
                    if results.hand_landmarks and len(results.hand_landmarks) > 0:
                        hand = results.hand_landmarks[0]
                        zs = [lm.z for lm in hand]
                        quality = float(np.std(zs))
                        if quality > best_quality:
                            best_quality = quality
                            best_frame = frame.copy()
                    elif best_frame is None:
                        best_frame = frame.copy()
                        
        if best_frame is not None:
            h, w = best_frame.shape[:2]
            scale = 200.0 / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(best_frame, (new_w, new_h))
            
            canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
            y_off = (200 - new_h) // 2
            x_off = (200 - new_w) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            
            # Save as asl_<phrase>.png to avoid overwriting ISL words if names overlap
            out_path = os.path.join(output_dir, f"asl_{phrase.lower()}.png")
            cv2.imwrite(out_path, canvas)
            # Also save thankyou.png and sorry.png directly since they have no ISL conflict
            if phrase.lower() in ["thankyou", "sorry"]:
                cv2.imwrite(os.path.join(output_dir, f"{phrase.lower()}.png"), canvas)
            print(f"  ASL Phrase {phrase} saved", flush=True)
            
    hands.close()


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "signs")
    os.makedirs(output_dir, exist_ok=True)
    
    build_digit_assets(output_dir)
    build_word_assets(output_dir)
    build_letter_assets(output_dir)
    build_asl_phrase_assets(output_dir)
    
    print("\nAll caption assets built successfully!")
