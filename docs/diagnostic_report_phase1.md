# Phase 1 Precision & Confusion Diagnostic Report

## Categorization of Failures

| Failure Mode | Head | Root Cause Category | Detailed Mechanism |
| :--- | :--- | :--- | :--- |
| **Eat ↔ Water** | `temporal_words` (ISL) | **Feature + Temporal** | Both gestures occur at mouth/chin. Without explicit velocity ($\Delta x, \Delta y$) and joint curl features, trajectory classifier only sees a hand near the face. |
| **Yes / Go / Hello / No** | `temporal_words` (ISL) | **Temporal + Model** | Unidirectional GRU with uniform subsampling across 4.5s video clips confuses initial hand raise. Needs frame-to-frame velocity + Bi-GRU. |
| **C ↔ G** | `landmark_letters` (ASL) | **Feature (Depth & Curl)** | C has all fingers curved; G has index pointing forward/side. In raw 63D coordinates without explicit curl angles and thumb-index distance, tilt projection causes overlap. |
| **U ↔ V** | `landmark_letters` (ASL) | **Feature (Inter-Fingertip Distance)** | U has index and middle touching; V has them spread. The distinguishing signal is inter-fingertip distance $\|p_8 - p_{12}\|$, which was diluted across 63 coordinates. |
| **J Over-Prediction** | `landmark_letters` (ASL) | **Architecture / Data** | J is a dynamic traced motion! It was erroneously included as a static class in the 26-class MLP. Transitioning hands resembled static J. |
| **Digit 3 Unreliable** | `landmark_digits` (ASL) | **Feature + User Convention** | ASL 3 requires thumb extended + index + middle. Users often sign European 3 (index + middle + ring). Also needs explicit thumb extension feature. |
| **ASL No** | `temporal_letters_phrases` | **Feature (Velocity / Delta)** | Snapping motion of fingers onto thumb requires velocity deltas to capture the rapid closing phase. |
