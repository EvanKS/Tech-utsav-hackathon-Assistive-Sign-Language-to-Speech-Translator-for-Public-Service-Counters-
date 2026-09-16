"""
Generate clear, full-picture demonstration images for all sign vocabulary.
- ISL Words: Peak gesture frame showing the full signer (upper body + active hands).
- ASL Dynamic Phrases: Full-body gesture frames from ASL_dynamic.
- ASL Digits (0-9): Clean, high-resolution hand exemplars.
- ASL Letters (A-Z): Clean, high-resolution hand exemplars.
"""
import os
import sys
import glob
import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.preprocess import create_hand_detector

SIGNS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "signs"))
D2_PATH = "c:/Users/Evan KS/OneDrive/Scans/Desktop/hackathon tech utsav/dataset vedio dataset"
D1_PATH = "c:/Users/Evan KS/OneDrive/Scans/Desktop/hackathon tech utsav/Sign-Language-Digits-Dataset-master/Sign-Language-Digits-Dataset-master/Dataset"
STATIC_LETTERS_PATH = "c:/Users/Evan KS/OneDrive/Scans/Desktop/hackathon tech utsav/SignAlphaSet dataset/SignAlphaSet/SignAlphaSet/SignAlphaSet"
DYNAMIC_ASL_PATH = "c:/Users/Evan KS/OneDrive/Scans/Desktop/hackathon tech utsav/SignAlphaSet dataset/SignAlphaSet/ASL_dynamic/ASL_dynamic"

def extract_peak_frame_isl(class_dir, detector):
    """Scan all videos in class_dir, find frame where hand is most raised and active."""
    vids = sorted(glob.glob(os.path.join(class_dir, "*.mp4")))
    if not vids:
        return None
        
    best_frame = None
    min_y = 999.0
    
    # Try all videos
    for vpath in vids:
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 5:
            cap.release()
            continue
            
        start = int(total * 0.10)
        end = int(total * 0.90)
        
        for i in range(start, end, 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)
            
            if res.hand_landmarks and len(res.hand_landmarks) > 0:
                wrist_y = res.hand_landmarks[0][0].y
                tip_y = res.hand_landmarks[0][8].y
                # Weighted hand height
                score_y = (wrist_y + tip_y) / 2.0
                
                # Check if this is the most elevated active sign
                if score_y < min_y:
                    min_y = score_y
                    best_frame = frame.copy()
                    
        cap.release()
        # If we found a very clear raised hand (above upper torso, y < 0.45), we're good
        if min_y < 0.45:
            break
            
    return best_frame

def crop_full_signer(frame):
    """Crop upper body portrait of the signer showing face, chest, and raised hands."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    # Keep top 85% height (head down to waist), center horizontally around signer
    crop_h = int(h * 0.88)
    crop_w = int(crop_h * 0.85) # 4:5 portrait aspect ratio
    
    start_x = max(0, (w - crop_w) // 2)
    end_x = min(w, start_x + crop_w)
    
    cropped = frame[0:crop_h, start_x:end_x]
    # Resize to crisp 400x480
    return cv2.resize(cropped, (400, 480), interpolation=cv2.INTER_LANCZOS4)

def process_isl_all():
    print("Generating full-picture ISL word demonstrations...")
    detector = create_hand_detector(static_image_mode=True, min_detection_confidence=0.3)
    classes = ['eat', 'go', 'hello', 'help', 'no', 'please', 'water', 'yes']
    
    for c in classes:
        cdir = os.path.join(D2_PATH, c)
        f = extract_peak_frame_isl(cdir, detector)
        if f is not None:
            clean = crop_full_signer(f)
            out_path = os.path.join(SIGNS_DIR, f"{c}.png")
            cv2.imwrite(out_path, clean)
            print(f"  [OK] ISL {c} saved full picture: {clean.shape}")
        else:
            print(f"  [WARN] Could not find peak frame for {c}")

def process_asl_dynamic_all():
    print("Generating full-picture ASL dynamic phrase demonstrations...")
    detector = create_hand_detector(static_image_mode=True, min_detection_confidence=0.3)
    phrases = ["HELLO", "NO", "SORRY", "THANKYOU", "YES"]
    
    for p in phrases:
        pdir = os.path.join(DYNAMIC_ASL_PATH, p)
        frame_dirs = sorted(glob.glob(os.path.join(pdir, "*_frames")))
        best_frame = None
        min_y = 999.0
        
        for fdir in frame_dirs[:3]:
            files = sorted(glob.glob(os.path.join(fdir, "*.jpg")))
            # Check frames in middle 60%
            for fpath in files[int(len(files)*0.15):int(len(files)*0.85):2]:
                img = cv2.imread(fpath)
                if img is None: continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_img)
                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    y = res.hand_landmarks[0][0].y
                    if y < min_y:
                        min_y = y
                        best_frame = img.copy()
            if min_y < 0.45:
                break
                
        if best_frame is not None:
            clean = crop_full_signer(best_frame)
            out_path = os.path.join(SIGNS_DIR, f"asl_{p.lower()}.png")
            cv2.imwrite(out_path, clean)
            print(f"  [OK] ASL {p} saved full picture: {clean.shape}")

def process_digits_clean():
    print("Generating clear ASL digit exemplars...")
    for d in range(10):
        ddir = os.path.join(D1_PATH, str(d))
        files = sorted(glob.glob(os.path.join(ddir, "*.JPG")) + glob.glob(os.path.join(ddir, "*.png")) + glob.glob(os.path.join(ddir, "*.jpg")))
        if files:
            img = cv2.imread(files[len(files) // 2])
            if img is not None:
                h, w = img.shape[:2]
                # Center square crop
                min_dim = min(h, w)
                sx = (w - min_dim) // 2
                sy = (h - min_dim) // 2
                sq = img[sy:sy+min_dim, sx:sx+min_dim]
                clean = cv2.resize(sq, (360, 360), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(os.path.join(SIGNS_DIR, f"{d}.png"), clean)
                print(f"  [OK] Digit {d} saved clean exemplar")

def process_letters_clean():
    print("Generating clear ASL letter exemplars...")
    for i in range(26):
        letter = chr(ord('A') + i)
        ldir = os.path.join(STATIC_LETTERS_PATH, letter)
        files = sorted(glob.glob(os.path.join(ldir, "*.jpg")) + glob.glob(os.path.join(ldir, "*.png")))
        if files:
            img = cv2.imread(files[len(files) // 2])
            if img is not None:
                h, w = img.shape[:2]
                min_dim = min(h, w)
                sx = (w - min_dim) // 2
                sy = (h - min_dim) // 2
                sq = img[sy:sy+min_dim, sx:sx+min_dim]
                clean = cv2.resize(sq, (360, 360), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(os.path.join(SIGNS_DIR, f"{letter.lower()}.png"), clean)

def main():
    os.makedirs(SIGNS_DIR, exist_ok=True)
    process_isl_all()
    process_asl_dynamic_all()
    process_digits_clean()
    process_letters_clean()
    print("\nDONE: All demonstration pictures are now clear, full-resolution, and perfectly framed!")

if __name__ == "__main__":
    main()
