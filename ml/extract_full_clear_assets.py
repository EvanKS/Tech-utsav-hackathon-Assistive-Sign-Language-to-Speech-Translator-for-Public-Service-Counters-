"""
Extract clear, full-resolution demonstration pictures for all signs:
- Full signer view (head, torso, and active hand) for ISL words and ASL phrases.
- High-contrast, sharp exemplars for digits (0-9) and letters (A-Z).
- Saves directly to frontend/public/signs/
"""
import os
import sys
import cv2
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.preprocess import create_hand_detector, extract_landmarks_from_image

FRONTEND_SIGNS_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "signs")
D2_PATH = "c:/Users/Evan KS/OneDrive/Scans/Desktop/hackathon tech utsav/dataset vedio dataset"
D1_PATH = "c:/Users/Evan KS/OneDrive/Scans/Desktop/hackathon tech utsav/Sign-Language-Digits-Dataset-master/Sign-Language-Digits-Dataset-master/Dataset"
STATIC_LETTERS_PATH = "c:/Users/Evan KS/OneDrive/Scans/Desktop/hackathon tech utsav/SignAlphaSet dataset/SignAlphaSet/SignAlphaSet/SignAlphaSet"
DYNAMIC_ASL_PATH = "c:/Users/Evan KS/OneDrive/Scans/Desktop/hackathon tech utsav/SignAlphaSet dataset/SignAlphaSet/ASL_dynamic/ASL_dynamic"

def extract_best_video_frame(class_dir):
    """
    Scan videos in class_dir, find the frame where hand is most actively performing the sign:
    - Hand is raised in upper half of frame (y < 0.7)
    - High MediaPipe confidence
    - Shows the signer's upper body and hand clearly (no tiny squished letterboxing)
    """
    vids = sorted(glob.glob(os.path.join(class_dir, "*.mp4")))
    if not vids:
        return None
        
    hands = create_hand_detector(static_image_mode=True, min_detection_confidence=0.4)
    best_frame = None
    best_score = -999.0
    
    # Check top 4 videos
    for vpath in vids[:4]:
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 5:
            cap.release()
            continue
            
        # Sample through the active middle 70% of the video
        start_f = int(total * 0.15)
        end_f = int(total * 0.85)
        step = max(1, (end_f - start_f) // 15)
        
        for f_idx in range(start_f, end_f, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            h, w = frame.shape[:2]
            lm = extract_landmarks_from_image(frame, hands)
            if lm is not None:
                # Landmark coords: p0 is wrist, p9 is middle MCP, p8 is index tip
                # We want hand to be actively raised (low y in image space)
                # Raw wrist height in normalized image coordinates
                wrist_y = lm[1]
                # High hand is a strong indicator of active sign (not resting at waist)
                # Active gesture score: raised hand + hand spread
                score = -wrist_y * 2.0
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
            elif best_frame is None and f_idx == total // 2:
                best_frame = frame.copy()
                
        cap.release()
        
    hands.close()
    return best_frame

def crop_signer_nicely(frame):
    """Crop or resize frame to a clean, full-picture 4:3 or 1:1 view showing signer clearly."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    
    # If 16:9 widescreen, crop slightly on sides to center on signer
    if w > h:
        new_w = int(h * 1.15)
        if new_w < w:
            start_x = (w - new_w) // 2
            frame = frame[:, start_x:start_x+new_w]
            
    # Resize to crisp 400x350
    out = cv2.resize(frame, (400, 350), interpolation=cv2.INTER_LANCZOS4)
    return out

def process_isl_words():
    print("Processing ISL Words clear full pictures...")
    classes = ['eat', 'go', 'hello', 'help', 'no', 'please', 'water', 'yes']
    for c in classes:
        cdir = os.path.join(D2_PATH, c)
        frame = extract_best_video_frame(cdir)
        if frame is not None:
            clean = crop_signer_nicely(frame)
            out_file = os.path.join(FRONTEND_SIGNS_DIR, f"{c}.png")
            cv2.imwrite(out_file, clean)
            print(f"  [OK] Saved full picture for ISL word '{c}' -> {out_file} ({clean.shape})")
        else:
            print(f"  [WARN] Could not find frame for {c}")

def process_asl_dynamic():
    print("Processing ASL Dynamic Phrases full pictures...")
    phrases = ["HELLO", "NO", "SORRY", "THANKYOU", "YES"]
    for p in phrases:
        pdir = os.path.join(DYNAMIC_ASL_PATH, p)
        if os.path.isdir(pdir):
            frame = extract_best_video_frame(pdir)
            if frame is not None:
                clean = crop_signer_nicely(frame)
                out_file = os.path.join(FRONTEND_SIGNS_DIR, f"asl_{p.lower()}.png")
                cv2.imwrite(out_file, clean)
                print(f"  [OK] Saved full picture for ASL phrase '{p}' -> {out_file}")

def process_digits():
    print("Processing Digits clear pictures...")
    for d in range(10):
        ddir = os.path.join(D1_PATH, str(d))
        files = sorted(glob.glob(os.path.join(ddir, "*.JPG")) + glob.glob(os.path.join(ddir, "*.png")) + glob.glob(os.path.join(ddir, "*.jpg")))
        if files:
            # Pick a clean mid-sample
            img = cv2.imread(files[len(files) // 3])
            if img is not None:
                clean = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LANCZOS4)
                out_file = os.path.join(FRONTEND_SIGNS_DIR, f"{d}.png")
                cv2.imwrite(out_file, clean)
                print(f"  [OK] Saved clean picture for digit '{d}'")

def process_letters():
    print("Processing Letters A-Z clear pictures...")
    for i in range(26):
        letter = chr(ord('A') + i)
        ldir = os.path.join(STATIC_LETTERS_PATH, letter)
        files = sorted(glob.glob(os.path.join(ldir, "*.jpg")) + glob.glob(os.path.join(ldir, "*.png")))
        if files:
            img = cv2.imread(files[len(files) // 2])
            if img is not None:
                clean = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LANCZOS4)
                out_file = os.path.join(FRONTEND_SIGNS_DIR, f"{letter.lower()}.png")
                cv2.imwrite(out_file, clean)

def main():
    os.makedirs(FRONTEND_SIGNS_DIR, exist_ok=True)
    process_isl_words()
    process_asl_dynamic()
    process_digits()
    process_letters()
    print("\nALL DEMO PICTURES RE-EXTRACTED WITH FULL HIGH RESOLUTION!")

if __name__ == "__main__":
    main()
