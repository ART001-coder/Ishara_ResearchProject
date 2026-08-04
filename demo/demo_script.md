# 🤟 Ishara Demo Presentation Script & Walkthrough

> **Reference Document:** [implementation_plan.md](../implementation_plan.md)  
> **Schedule Reference:** Section 9 -> Day 11 (Task 11.4) & Day 14 (Demo Day Execution)

---

## ⏱️ Presentation Timing Breakdown (15 Minutes Total)

- **00:00 - 03:00 (3 min):** Introduction & Problem Statement (Member C)
- **03:00 - 06:00 (3 min):** System Architecture & Pipeline Breakdown (Member B)
- **06:00 - 11:00 (5 min):** **LIVE WEBCAM DEMONSTRATION** (All Members)
- **11:00 - 13:00 (2 min):** Quantitative Results & Confusion Analysis (Member A)
- **13:00 - 15:00 (2 min):** Q&A & Technical Discussion (All Members)

---

## 🎬 Live Demo Sign Sequence Protocol

### Demo Scenario: Hospital Reception Desk Assistance

During the live demonstration, the designated signer stands in front of the webcam (good lighting, plain background) and executes the following pre-curated sign sequences:

#### Sequence 1: Greeting & Identification
- **Signed Gestures:** `hello` -> `name` -> `what`
- **Detected Glosses:** `["hello" (0.95), "name" (0.88), "what" (0.75)]`
- **Gemini Output:** `"Hello, what is your name?"`

#### Sequence 2: Medical Assistance Request
- **Signed Gestures:** `doctor` -> `need` -> `help`
- **Detected Glosses:** `["doctor" (0.92), "need" (0.85), "help" (0.78)]`
- **Gemini Output:** `"I need help from the doctor."`

#### Sequence 3: Symptom Reporting
- **Signed Gestures:** `stomach` -> `pain` -> `bad`
- **Detected Glosses:** `["stomach" (0.91), "pain" (0.95), "bad" (0.80)]`
- **Gemini Output:** `"I have bad stomach pain."`

#### Sequence 4: Prescription Inquiry
- **Signed Gestures:** `medicine` -> `when` -> `see`
- **Detected Glosses:** `["medicine" (0.88), "when" (0.72), "see" (0.65)]`
- **Gemini Output:** `"When should I take the medicine?"`

---

## 🚨 Emergency Protocol (If Live Demo Fails on Stage)

If lighting conditions, camera hardware, or network issues cause live webcam mispredictions:

1. **DO NOT ATTEMPT LIVE DEBUGGING ON STAGE.**
2. Member C seamlessly transitions to pre-recorded video backup:
   ```bash
   # Play backup video recording (Section 11.2 R4/R6 Mitigation)
   vlc demo/backup_recording.mp4
   ```
3. Continue narrative explaining the system architecture and offline pre-recorded evaluation results.
