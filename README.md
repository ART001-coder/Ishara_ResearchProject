#  Ishara: ISL to Grammatically Correct Sentence Translator

> **An Indian Sign Language (ISL) recognition system with LLM-assisted sentence reconstruction.**  
> *Reference Document: [implementation_plan.md](implementation_plan.md)*

---

##  Project Overview

**Ishara** bridges the communication gap between Deaf/Hard-of-Hearing individuals who use Indian Sign Language (ISL) and hearing individuals in healthcare and reception environments.

The system processes real-time video feeds from standard webcams, extracts body and hand poses using **MediaPipe Holistic**, classifies temporal keypoint sequences into ISL word glosses using a **Bidirectional LSTM with Attention**, buffers deduplicated predictions, and utilizes the **Google Gemini API (gemini-2.0-flash)** to reconstruct raw sign sequences into natural, grammatically correct English sentences.

---

##  System Architecture

```
┌──────────┐    ┌──────────────────┐    ┌────────────────┐    ┌──────────────┐    ┌────────────────────┐    ┌────────────┐
│  Webcam  │───▶│ MediaPipe        │───▶│ Sequence       │───▶│ LSTM / GRU   │───▶│ Gemini API         │───▶│ Streamlit  │
│  OpenCV  │    │ Holistic         │    │ Windowing      │    │ Classifier   │    │ Sentence Builder   │    │ Display    │
│  30 FPS  │    │ Keypoint Extract │    │ 30-frame buf   │    │ word + conf  │    │ gloss → sentence   │    │ UI         │
└──────────┘    └──────────────────┘    └────────────────┘    └──────────────┘    └────────────────────┘    └────────────┘
   Stage 1            Stage 2               Stage 3               Stage 4               Stage 5                Stage 6
```

- **Stage 1 (Video Capture):** OpenCV captures 640x480 video stream at 30 FPS.
- **Stage 2 (Keypoint Extraction):** MediaPipe Holistic extracts 225-dimensional feature vectors per frame (33 pose + 21 left hand + 21 right hand landmarks, 3D coordinates).
- **Stage 3 (Sequence Windowing):** NumPy ring buffer maintains a rolling window of 30 frames (`1, 30, 225`).
- **Stage 4 (Sign Classifier):** PyTorch Bi-LSTM with Attention classifies sequences into vocabulary glosses with confidence scores.
- **Stage 5 (Sentence Builder):** Deduplicating Gloss Buffer collects valid predictions and sends them to Gemini 2.0 Flash (with offline rule-based fallback).
- **Stage 6 (Streamlit UI):** Interactive web dashboard renders real-time keypoint overlays, detected gloss history, and final reconstructed sentences.

---

##  Repository Structure

```
Ishara/
├── README.md                          # Project documentation
├── requirements.txt                   # Environment dependencies
├── config.yaml                        # Central system hyperparameters & configuration
├── .gitignore                         # Data/checkpoint exclusion rules
├── IShara_Project_Proposal.md         # Initial project proposal
├── implementation_plan.md             # Complete 14-day technical implementation plan
│
├── data/
│   ├── raw/                           # Raw video datasets (INCLUDE, CISLR)
│   ├── processed/                     # Extracted keypoints (.npy tensors)
│   └── vocabulary.json                # Master word mapping (65 words)
│
├── src/
│   ├── data/                          # Data acquisition, extraction & augmentation
│   ├── model/                         # Neural network architectures & training loop
│   ├── inference/                     # Predictor, gloss buffer & Gemini LLM client
│   └── utils/                         # Config loader, metrics & visualizers
│
├── app/
│   └── streamlit_app.py               # Main UI application
│
├── tests/                             # Unit testing suite
└── demo/                              # Presentation scripts & backup media
```

---

##  Quick Start & Environment Setup

1. **Clone & Setup Environment:**
   ```bash
   git clone <repository_url>
   cd Ishara
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set API Credentials:**
   Set your Google Gemini API key as an environment variable:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"  # On Windows PowerShell: $env:GEMINI_API_KEY="your_key"
   ```

3. **Run Streamlit Web Application:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

##  Key References in `implementation_plan.md`

- **Resolved Technical Decisions:** Section 1
- **System Architecture & Layer Specs:** Section 4
- **Dataset Strategy (No Manual Recording):** Section 5
- **Training Configuration & Augmentation:** Section 6
- **Known Pitfalls & Mitigations (Bugs D1-D7, M1-M6, I1-I5):** Section 8
- **14-Day Detailed Execution Schedule:** Section 9
- **Contingency Escalation & Kill-Switch:** Section 11
