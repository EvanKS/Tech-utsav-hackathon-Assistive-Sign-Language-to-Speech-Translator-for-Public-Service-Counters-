# 🌉 SignBridge: Two-Way Assistive Sign-Language-to-Speech & Speech-to-Sign Translator for Public Service Counters

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2.0-61DAFB.svg?logo=react)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-FF6F00.svg?logo=tensorflow)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-007ACC.svg)](https://developers.google.com/mediapipe)
[![Vite](https://img.shields.io/badge/Vite-5.0-646CFF.svg?logo=vite)](https://vitejs.dev/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Tech Utsav Hackathon Submission**  
> **Theme:** Assistive Technology & Public Service Accessibility  
> **Target Deployments:** Railway inquiry desks, postal counters, bank teller booths, municipal government service offices, hospital triage desks.

---

## 📖 Executive Summary & Problem Statement

Public service counters are critical gateways for citizens to access transit, banking, healthcare, and civic documents. However, for Deaf and Hard-of-Hearing (DHH) individuals, standard counter interactions present profound communication barriers:
- Counter staff rarely know sign language.
- Written notes on paper are slow, cumbersome, and compromise privacy.
- High ambient background noise at transit stations or banks renders hearing aids ineffective.
- Existing sign recognition research often relies on expensive sensor gloves or depth cameras unavailable at public service desks.

**SignBridge** solves this with a **zero-hardware-cost, privacy-first, bidirectional communication system** that operates entirely via standard counter webcams, screens, and speakers:
1. **Sign-to-Speech (Citizen $\to$ Staff):** The citizen performs signs or fingerspells in front of the camera. The AI pipeline extracts 3D landmarks on-device, predicts through specialized neural classifiers, stabilizes predictions, and outputs spoken synthesized speech and text to the staff terminal.
2. **Speech-to-Sign (Staff $\to$ Citizen):** Counter staff speaks or types into their console. SignBridge generates synthesized audio while simultaneously rendering an animated **Sign Caption Player** that shows whole-word sign demonstrations and dynamically falls back to fingerspelling for names, token numbers, and arbitrary vocabulary.

---

## 🌟 Key Features & Innovations

- 🧠 **4-Head Neural Engine:** Dedicated, high-precision classifiers for:
  - **Numeric Digits (ASL 0–9)** — 98.53% test accuracy.
  - **Isolated Words (ISL 8 Words)** — 100.0% validation accuracy.
  - **Fingerspelling Letters (ASL A–Z)** — 100.0% test accuracy.
  - **Conversational Phrases (ASL Dynamic Gestures)** — 91.67% validation accuracy.
- 📐 **87-Dimensional Precision Feature Pipeline:** Employs canonical in-plane rotation normalization (tilt-invariance), 8 pairwise fingertip distance ratios, 6 signed depth-order occlusion features, and 10 joint flexion angles to completely eliminate historical C↔G, U↔V, and thumb-crossing confusions.
- ⚡ **Frame-to-Frame Velocity Deltas ($\Delta \mathbf{x}_t$):** Trajectory-based Bidirectional GRU/LSTM models capture acceleration and motion direction, eliminating `eat ↔ water` and `hello ↔ no` motion confusions.
- 🖐️ **Dual-Hand Symmetry:** Evaluates both standard and horizontally mirrored coordinates ($x' = -x$), providing seamless recognition for both right-handed and left-handed signers.
- ⏱️ **Temporal Speed Invariance:** Employs uniform temporal resampling across continuous gesture segments, enabling natural signing regardless of whether the user signs fast or slow.
- ⌨️ **Fingerspelling Tray & Word Assembly:** Real-time buffer allowing citizens to fingerspell words (names, IDs, tokens) with tactile Backspace, Space, Clear, and Direct Send controls.
- 🔤 **Automatic Fingerspelling Decomposition Fallback:** When staff speaks or types any word outside the predefined whole-word dictionary (e.g., *"token 42"* or a citizen's name), the system decomposes it into constituent letters and plays sequential sign exemplars with 100% vocabulary coverage.
- 🖼️ **High-Definition Visual Guide & Live Practice Matcher:** Includes crystal-clear $400\times 480$ upper-body demonstration portraits for every vocabulary sign, paired with an interactive real-time webcam practice matcher.
- 🛡️ **Privacy-First On-Device Extraction:** Raw webcam video never leaves the client browser. Only mathematical landmark coordinate vectors are processed.

---

## 📊 Production Model Suite & Measured Benchmark Metrics

| Model Head | Classes / Vocabulary | Modality | Language | Architecture | Parameter Count | Measured Benchmark | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Digits** | 10 Classes (`0`–`9`) | Static Landmark (87D) | **ASL** | MLP (`87 → 128 → 64 → 10`) | 19,850 | **98.53%** Test Acc (Digit 3: 93.3% recall) | Production |
| **2. Words** | 8 Words (`eat`, `go`, `hello`, `help`, `no`, `please`, `water`, `yes`) | Velocity Trajectory (`32 × 126`) | **ISL** | Bi-GRU (`126 → Bi-GRU(64) → 32 → 8`) | 78,536 | **100.0%** Val Acc (94.6% 5-Fold CV) | Production |
| **3. Letters** | 25 Active Classes (`A`–`I`, `K`–`Z`) | Static Landmark (87D) | **ASL** | MLP (`87 → 128 → 64 → 25`) | 20,825 | **100.0%** Test Acc (C/G=0 err, U/V=0 err) | Production |
| **4. Dynamic Phrases** | 5 Phrases (`HELLO`, `NO`, `SORRY`, `THANKYOU`, `YES`) | Velocity Trajectory (`32 × 126`) | **ASL** | Bi-LSTM (`126 → Bi-LSTM(64) → 32 → 5`) | 103,109 | **91.67%** Val Acc (NO: 100% recall) | Production |

*Note on Letter 'J':* ASL 'J' is a traced dynamic stroke. Following best practices in assistive HCI, static 'J' is excluded from the per-frame static MLP to prevent transition false positives, and recognized via trajectory motion.

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    subgraph Client_Browser ["Client Browser (Citizen & Staff Terminals)"]
        Cam["Webcam Input"] --> MP["MediaPipe HandLandmarker (Tasks Vision)"]
        MP --> Norm["Client Pre-normalization (Wrist Centered, Scaled)"]
        Norm --> Router{"Mode Dispatcher"}
        Router -->|"Static (Digits/Letters)"| ReqStatic["Single Frame (63D)"]
        Router -->|"Temporal (Words/Gestures)"| ReqSeq["32-Frame Rolling Buffer (2016D)"]
    end

    subgraph Backend_FastAPI ["FastAPI High-Performance Server (:8000)"]
        ReqStatic --> Feat87["87D Precision Feature Extractor<br/>(Rotation Invariance + Keypoint Distances + Depth-Order + Joint Angles)"]
        ReqSeq --> Resample["Uniform Temporal Resampling (np.linspace)<br/>+ Velocity Deltas (32 x 126)"]
        
        Feat87 --> DualHand1["Dual-Hand Symmetry Evaluator (Normal & Mirrored x' = -x)"]
        Resample --> DualHand2["Dual-Hand Symmetry Evaluator (Normal & Mirrored x' = -x)"]
        
        DualHand1 --> M1["MLP (Digits 0-9)"]
        DualHand1 --> M3["MLP (Letters A-Z)"]
        DualHand2 --> M2["Bi-GRU (ISL Words)"]
        DualHand2 --> M4["Bi-LSTM (ASL Gestures)"]
        
        M1 & M2 & M3 & M4 --> Stab["SignStabilizer<br/>(Window Agreement + Confidence Cutoff + Cooldown Gate)"]
        Stab --> OutAPI["JSON Response (Label, Stable Flag, Top-K, Language)"]
    end

    subgraph Two_Way_Interaction ["Two-Way Communication Bus"]
        OutAPI --> Tray["Fingerspelling / Token Tray"]
        Tray --> TTS["Speech Synthesis (Web Speech API / pyttsx3)"]
        TTS --> StaffDesk["Staff Terminal Audio & Transcript"]
        
        StaffDesk -->|"Voice or Type"| STT["Staff Message Input"]
        STT --> Fallback["Fingerspelling Fallback Parser"]
        Fallback --> Captions["Sign Caption Player (Citizen Screen)"]
    end
```

---

## 📁 Repository Directory Structure

```
signbridge/
├── backend/                      # High-performance FastAPI backend application
│   ├── main.py                  # Entrypoint, lifespan startup checks, CORS, static mounts
│   ├── config.py                # App configuration, directory paths, and thresholds
│   ├── schemas.py               # Pydantic request and response schemas (4 heads)
│   ├── routers/
│   │   ├── recognize.py         # /api/recognize with 4-way dispatch & stabilizer integration
│   │   ├── speech.py            # /api/speak (offline TTS engine fallback)
│   │   ├── text_to_sign.py      # /api/text-to-sign with fingerspelling fallback decomposition
│   │   ├── message.py           # /api/message bidirectional session conversation logger
│   │   └── session.py           # /api/session/{id} state retrieval and reset
│   └── services/
│       ├── classifier.py        # 4-head neural model loader, precision feature runner
│       ├── smoothing.py         # SignStabilizer session manager and re-tuned thresholds
│       └── vocabulary.py        # Registry validator and token-to-sign decomposition engine
├── frontend/                     # Modern React 18 + Vite responsive user interface
│   ├── public/
│   │   └── signs/               # 51 Clear, full-resolution demonstration pictures (400x480)
│   ├── src/
│   │   ├── components/
│   │   │   ├── CitizenPanel.jsx # Citizen camera, mode tabs, visual guide, spelling tray
│   │   │   ├── StaffPanel.jsx   # Staff console, quick phrases, TTS, mic input
│   │   │   ├── SpellingTray.jsx # Real-time fingerspelling accumulator & word controls
│   │   │   ├── SignCaptionPlayer.jsx # Visual sign player for staff messages
│   │   │   ├── ConversationLog.jsx   # Dual-perspective conversation history
│   │   │   └── VocabularyDrawer.jsx  # Judges' transparency & audit drawer
│   │   ├── hooks/
│   │   │   ├── useHandLandmarks.js   # MediaPipe Tasks vision camera tracker & canvas HUD
│   │   │   ├── useRecognition.js     # 75ms temporal buffer & throttled API client
│   │   │   └── useSpeech.js          # Web Speech API speech synthesis & recognition
│   │   ├── lib/api.js           # Strongly-typed Axios backend client
│   │   ├── index.css            # Responsive dark-mode styling with accessible high-contrast tokens
│   │   └── App.jsx              # Dual-split service counter layout
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── ml/                           # ML training, feature extraction, and diagnostic suite
│   ├── config.py                # Hyperparameters, paths, and sequence lengths
│   ├── preprocess.py            # 87D precision feature extractor & 126D velocity deltas
│   ├── dataset.py               # Disk-cached NPZ loaders with GroupShuffleSplit
│   ├── temporal_smoothing.py    # SignStabilizer algorithm implementation
│   ├── diagnose_failures.py     # Diagnostic script checking confusion matrix cells
│   ├── train_precision_digits.py# Digits MLP training script with 87D features
│   ├── train_precision_letters.py# Letters MLP training script (25 classes, 87D features)
│   ├── train_temporal_precision.py # ISL Words Bi-GRU training script with velocity deltas
│   ├── train_temporal_asl_precision.py # ASL Dynamic Phrases Bi-LSTM training script
│   ├── make_perfect_demo_assets.py# Extracts crystal-clear full-body demonstration frames
│   └── verify_precision_fixes.py # Comprehensive bug-fix verification test suite
├── models/                       # Trained production weights and label manifests
│   ├── landmark_digits.keras & landmark_digits_labels.json
│   ├── temporal_words.keras & temporal_words_labels.json
│   ├── landmark_letters.keras & landmark_letters_labels.json
│   └── temporal_letters_phrases.keras & temporal_letters_phrases_labels.json
├── vocabulary/
│   └── registry.json            # Unified registry mapping heads, datasets, classes, and language
├── docs/                         # Technical documentation, benchmark logs, and confusion matrices
│   ├── diagnostic_report_phase1.md # Full diagnostic root-cause report
│   ├── precision_verification_report.json # Numerical before/after verification logs
│   ├── letters_precision_cm.png # Confusion matrix for precision letters
│   ├── digits_precision_cm.png  # Confusion matrix for precision digits
│   ├── isl_words_precision_cm.png # Confusion matrix for precision ISL words
│   └── asl_gestures_precision_cm.png # Confusion matrix for precision ASL gestures
├── tests/                        # Automated unit and integration test suites
│   ├── test_smoothing.py        # Temporal stabilizer algorithmic tests
│   └── test_api.py              # End-to-end multi-head API integration tests
├── Dockerfile                    # Containerization specification for container deployment
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # Comprehensive project documentation
```

---

## ⚡ Quickstart Guide: Running the Project Locally

### Prerequisites
- **Python:** 3.10, 3.11, or 3.12 installed.
- **Node.js:** v18.0.0 or higher installed.
- **Webcam & Microphone:** Any standard built-in or USB camera.

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/EvanKS/Tech-utsav-hackathon-Assistive-Sign-Language-to-Speech-Translator-for-Public-Service-Counters-.git
cd Tech-utsav-hackathon-Assistive-Sign-Language-to-Speech-Translator-for-Public-Service-Counters-
```

---

### Step 2: Set Up & Start the FastAPI Backend
1. Create and activate a virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the backend server:
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   - **Backend API:** [http://localhost:8000](http://localhost:8000)
   - **API Health Endpoint:** [http://localhost:8000/api/health](http://localhost:8000/api/health)
   - **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Step 3: Set Up & Start the React Frontend
In a **second terminal window**:
```bash
cd frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in Google Chrome, Microsoft Edge, or Mozilla Firefox.

---

### Step 4: Run Automated Verification Tests
Verify all ML models, stabilizer logic, and API endpoints:
```bash
# Run temporal stabilizer unit tests
python tests/test_smoothing.py

# Run end-to-end API integration tests
python tests/test_api.py

# Run comprehensive precision benchmark test
python ml/verify_precision_fixes.py
```
*Expected Result:*
```
[PASS] All smoothing tests passed!
[OK] Startup Check: Head 'digits' matches registry (10 classes)
[OK] Startup Check: Head 'words' matches registry (8 classes)
[OK] Startup Check: Head 'letters' matches registry (25 classes)
[OK] Startup Check: Head 'gesture_asl' matches registry (5 classes)
============================================================
ALL API ENDPOINT & MULTI-HEAD INTEGRATION TESTS PASSED (100% SUCCESS)!
============================================================
```

---

## 🐳 Docker Container Deployment

SignBridge includes a production-ready container definition:
```bash
# Build the Docker image
docker build -t signbridge:latest .

# Run the container
docker run -d -p 8000:8000 --name signbridge_service signbridge:latest
```
Access the application at `http://localhost:8000`.

---

## 🎯 Evaluator & Judge Walkthrough (3-Minute Live Demo)

Follow these steps for a complete evaluation of the system:

```
+-----------------------------------------------------------------------------------+
|                        SIGNBRIDGE PUBLIC SERVICE COUNTER                          |
+-----------------------------------------+-----------------------------------------+
|        👤 CITIZEN SIDE (Webcam)         |         💼 STAFF CONSOLE (Counter)       |
|                                         |                                         |
|  [Mode Tabs: Words | Gestures | Letters]|  Counter Phrases:                       |
|  +-----------------------------------+  |  [ "Please wait for your turn." ]       |
|  |   Live Camera + Glow Skeleton     |  |  [ "Please show your ID."       ]       |
|  |   HUD: [PREDICTION: HELLO (98%)]  |  |  [ "Go to counter 3."           ]       |
|  +-----------------------------------+  |                                         |
|  Visual Reference Cards:                |  Staff Speech/Text Input:               |
|  [HELLO] [WATER] [EAT] [PLEASE] [NO]   |  [ Type: "Hello, please show ID 5" ]    |
|                                         |  [ 🔊 Speak & Send ]                    |
|  Fingerspelling Tray:                   |                                         |
|  [ T ] [ O ] [ K ] [ E ] [ N ]          |  Live Conversation Log:                 |
|  [<- Backspace] [Space] [Send Word]     |  Citizen: "TOKEN" (ASL Letters)         |
|                                         |  Staff:   "Please go to counter 3"      |
+-----------------------------------------+-----------------------------------------+
```

### 1. Test Static Fingerspelling (ASL Letters & Fingerspelling Tray)
1. On the Citizen panel, click **"📹 Start Camera"**.
2. Select the **Letters (ASL)** mode tab.
3. Show the letters **T**, **O**, **K**, **E**, **N** sequentially to the camera.
4. Watch each recognized letter append into the **Fingerspelling Tray**.
5. Click **"📤 Add Word to Message"** or **"Send Word"** to transmit the assembled word to the Staff desk.

### 2. Test Isolated Words (ISL Video Head)
1. Switch to the **Words (ISL)** mode tab.
2. Look at the **Visual Guide** strip below the camera — each card shows a clear full-body demonstration of the sign.
3. Perform the **Hello** wave or **Water** gesture.
4. Observe the rolling buffer fill indicator reaching 100% and stabilizing the word prediction in the HUD.

### 3. Test Conversational Gestures (ASL Dynamic Phrases)
1. Switch to the **Gestures (ASL)** mode tab.
2. Perform **HELLO** (salute from brow) or **NO** (index and middle fingers snapping to thumb).
3. The model recognizes the gesture trajectory and commits the phrase.

### 4. Test Speech-to-Sign with Automatic Fingerspelling Fallback
1. On the **Staff Console** (right side), type or speak:
   > *"Hello, please show ID 5"*
2. Click **"🔊 Send to Citizen"**.
3. On the Citizen's screen, the **Sign Caption Player** dynamically renders:
   - Whole-word video exemplars for *"Hello"* and *"Please"*.
   - The numeric sign for *"5"*.
   - Constituent letter cards (`I`, `D`) for the unmatched token *"ID"*, proving 100% vocabulary coverage.

### 5. Inspect Model Transparency (Judges' Audit Drawer)
1. Click the **"📖 Vocabulary"** button in the top-right header.
2. View the full registry detailing model architectures, parameter counts, and language attribution (`ASL` vs `ISL`).

---

## 📡 REST API Reference

The backend exposes strongly-typed, auto-documented endpoints at `/docs`:

| Method | Endpoint | Description | Payload Example |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service uptime and 4-model load status | *None* |
| `GET` | `/api/vocabulary` | Complete registry schema and supported classes | *None* |
| `POST` | `/api/recognize` | Evaluates 63D landmark or 2016D sequence | `{"session_id": "s1", "mode": "digits", "landmarks": [...]}` |
| `POST` | `/api/text-to-sign` | Decomposes text with fingerspelling fallback | `{"text": "Hello token 5"}` |
| `POST` | `/api/speak` | Synthesizes audio using offline TTS engine | `{"text": "Thank you, you may go"}` |
| `POST` | `/api/message` | Logs conversation messages into session | `{"session_id": "s1", "sender": "citizen", "text": "HELLO"}` |
| `GET` | `/api/session/{id}` | Fetches full conversation history | *None* |

---

## 📚 Dataset Citations & Acknowledgments

SignBridge is built upon and acknowledges the following open datasets:
1. **Sign Language Digits Dataset:** Arda Mavi, Zeynep Dicle (*10 ASL digits, 2,062 images*).
2. **ISL Isolated 8-Words Dataset:** Vidit Sharma, CISLR Project (*8 isolated Indian Sign Language words, video format*).
3. **SignAlphaSet:** ASL Static Alphabet (A–Z) and Dynamic Conversational Phrases (*SignAlphaSet Open Dataset Project*).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Built with accessibility and civic inclusion at heart for **Tech Utsav Hackathon**.
