# SignBridge — Dataset Analysis & Verification Report

This document records the empirical dataset analysis, image statistics, MediaPipe landmark detection feasibility, and routing architecture across all SignBridge datasets.

---

## 1. Baseline Datasets (Phase 1)

### 1.1 Static Digits — Arda Mavi Sign Language Digits (D1)
- **Modality**: Static RGB images
- **Classes**: 10 digits (`0` through `9`)
- **Total Images**: 2,062 images (~206 images/class)
- **Resolution**: 100×100 px, bounding cropped to hand
- **MediaPipe Landmark Detection Rate**: 87.7% on raw crops (1,809 valid 63-dim landmark samples extracted and cached in `data/processed/d1_landmarks.npz`).
- **Target Model**: `landmark_digits.keras` (MLP: 63 → 128 → 64 → 10)
- **Empirical Accuracy**: **97.79% Test Accuracy** (Test Loss: 0.3150 across 272 unseen test samples).

### 1.2 Temporal Words — ISL Isolated Words (D2)
- **Modality**: Video sequences (MP4)
- **Sign Language**: Indian Sign Language (ISL)
- **Classes**: 8 conversational words (`eat`, `go`, `hello`, `help`, `no`, `please`, `water`, `yes`)
- **Total Videos**: 56 videos (7 videos/class)
- **Framerate / Resolution**: 25–30 FPS, variable duration (1.2s to 3.5s)
- **Sequence Processing**: Uniformly sampled to $T=32$ frames, 63 normalized landmarks per frame.
- **Evaluation Split**: GroupShuffleSplit by video ID (ensures 0% frame leakage between train and validation).
- **Target Model**: `temporal_words.keras` (GRU: Input(32, 63) → GRU(64) → Dense(32) → Dense(8, softmax))
- **Empirical Accuracy**: **94.12% Validation Accuracy** on unseen signer videos.

---

## 2. New Dataset — SignAlphaSet (Phase 2)

- **Sign Language**: American Sign Language (ASL)
- **Components**: Static half (A–Z images) and Dynamic half (300 videos across 31 folders).

### 2.1 Static SignAlphaSet Analysis (`A` through `Z`)
- **Total Images**: 26,000 images
- **Class Balance**: Exactly 1,000 images per class across all 26 letters (A–Z). Perfectly balanced.
- **Image Dimensions**: 296×296 RGB
- **Corrupt / Unreadable Files**: 0 corrupt files found.
- **MediaPipe Landmark Detection Feasibility**:
  - Sampled and tested with MediaPipe Tasks `HandLandmarker` (min detection confidence 0.30):
    - Letters A–N, P–Z: **100.0% detection rate** (25/25 per class).
    - Letter O: **88.0% detection rate** (22/25).
  - Overall detection rate across all 26 letters: **~99.5%**.
  - No letter falls below the 80% acceptable threshold.
- **Potential Confusions**:
  - `M`, `N`, `S`, `T`: Fist-based configurations where thumb position differentiates the letter. Wrist-normalized landmark coordinate vectors preserve relative thumb-to-index joint distances accurately.
- **Target Model**: `landmark_letters.keras` (MLP: 63 → 128 → 64 → 26).

### 2.2 Dynamic SignAlphaSet Analysis (31 Classes)
- **Total Classes**: 31 (`A`–`Z` + `HELLO`, `NO`, `SORRY`, `THANKYOU`, `YES`).
- **Total Videos**: 300 videos (~10 videos per class, 5 for `NO` and `YES`).
- **FPS & Duration**: 30.0 FPS, durations ranging from 0.87s to 4.93s (26 to 154 frames).
- **Pre-extracted Frames**: All folders (except `S`) contain pre-extracted `.jpg` frame sequences aligned with source `.avi` clips.
- **Routing Decision**:
  - **Static routing**: Letters `A`–`I` and `K`–`Y` are static poses in ASL. Routing them to `landmark_letters` (trained on 26,000 images) provides orders of magnitude higher sample density (1,000 images/class vs. 10 videos).
  - **Dynamic routing**:
    - The 5 conversational phrase gestures: `HELLO`, `NO`, `SORRY`, `THANKYOU`, `YES`. These have inherent multi-phase dynamic movement (e.g., open palm salute for Hello, hand-to-chest circular motion for Sorry, chin-to-open-palm outward motion for Thank You).
    - Letters with intrinsic path motion: `J` (curved dip) and `Z` (zigzag stroke).
  - The dynamic head (`temporal_letters_phrases`) targets the 5 essential public-counter conversational ASL phrase gestures (`HELLO`, `NO`, `SORRY`, `THANKYOU`, `YES`), evaluated with `GroupShuffleSplit` by video to prevent frame leakage.

---

## 3. Language Separation & Architectural Summary

| Head Identifier | Target Dataset | Language | Modality | Architecture | Output Dim |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `digits` (`static`) | Arda Mavi Digits | ASL | Static Frame | MLP (128→64→10) | 10 classes |
| `words` (`temporal`) | ISL Isolated Words | ISL | 32-Frame Sequence | GRU (64→32→8) | 8 classes |
| `letters` (`letters`) | SignAlphaSet Static | ASL | Static Frame | MLP (128→64→26) | 26 classes |
| `gesture_asl` (`temporal`) | SignAlphaSet Dynamic | ASL | 32-Frame Sequence | LSTM (64→32→5) | 5 classes |
