# Ishara — ISL-to-Text Translation System

## Comprehensive Implementation Plan (14 Days · 3 Members)

The Ishara project translates Indian Sign Language (ISL) gestures from a webcam into grammatically correct English sentences. It uses MediaPipe for pose extraction, a Bidirectional LSTM for sign classification, and Google Gemini for sentence reconstruction.

---

## 1. Resolved Decisions (formerly Open Questions)

All previously open questions are resolved below with the most practical choices for a 2-week sprint with 3 developers.

### 1.1 Framework: **PyTorch**

| Considered | Verdict | Why |
|---|---|---|
| PyTorch | ✅ **Selected** | Eager execution makes debugging trivial; dominant in research; `torch.jit` for easy export; better DataLoader ergonomics |
| TensorFlow | ❌ Rejected | Heavier setup; graph mode debugging is painful for rapid prototyping; Keras abstraction hides errors |

### 1.2 Mode: **Real-time webcam (primary) + pre-recorded fallback**

| Considered | Verdict | Why |
|---|---|---|
| Real-time webcam | ✅ **Primary** | Required for live demo; more impressive for presentation; proves system works end-to-end |
| Pre-recorded video | ✅ **Backup** | Pre-recorded demo video as insurance if live demo fails on stage |

### 1.3 Vocabulary Size: **50–80 words (INCLUDE-50 as baseline)**

| Considered | Verdict | Why |
|---|---|---|
| 20–50 words | Too small for a meaningful demo sentence | |
| **50–80 words** | ✅ **Selected** | Sweet spot: enough for realistic hospital sentences, trainable in 2 weeks, INCLUDE-50 gives us 50 words out of the box |
| 100–150 words | Too ambitious for 3 people in 14 days | |

### 1.4 Data: **Public datasets only — NO manual recording**

| Considered | Verdict | Why |
|---|---|---|
| Manual recording | ❌ Rejected | Recording + labelling + QA burns an entire day per 15 words; 3 people can't afford this |
| **Datasets only** | ✅ **Selected** | INCLUDE (263 words, 4,287 videos) covers our vocabulary; CISLR fills gaps; vocabulary is constrained to what's available |

### 1.5 LLM API: **Google Gemini (gemini-2.0-flash)**

| Considered | Verdict | Why |
|---|---|---|
| **Gemini Flash** | ✅ **Selected** | Lowest latency (~200ms); generous free tier; excellent at structured grammar tasks; Python SDK available |
| OpenAI GPT | ❌ Rejected | Higher cost; no free tier; overkill for short prompts |
| Local LLM | ❌ Rejected | Requires GPU RAM we need for training; adds deployment complexity |

### 1.6 UI Framework: **Streamlit**

| Considered | Verdict | Why |
|---|---|---|
| **Streamlit** | ✅ **Selected** | Zero frontend code; built-in webcam support; deploys in 1 command; good enough for demo |
| Flask/React | ❌ Rejected | Requires separate frontend build; 2× the work for the same demo result |

---

## 2. Critical Changes from the Original Proposal

> [!IMPORTANT]
> The original proposal was written for a **5-person** team. We are **3 people**. The following decisions tighten scope.

| Original Proposal | Revised Plan | Rationale |
|---|---|---|
| 100–150 word vocabulary | **50–80 words** | Fewer classes = faster training, less confusion-pair debugging |
| 5 dedicated roles | **3 members, blended roles** | Each member owns one pillar but helps others during integration |
| Team-recorded data for gaps | **No manual recording — datasets only** | Recording is the #1 time sink; public datasets cover enough vocabulary |
| Full confusion analysis | **Top-10 confused pairs only** | Deep analysis on all 80 classes is unrealistic; fix the worst offenders |
| Separate integration lead | **All 3 members share integration** | No dedicated person; integration is a daily standup responsibility |
| OpenHands fine-tuning | **MediaPipe + LSTM/GRU** | OpenHands is archived/unmaintained; MediaPipe + custom LSTM is the standard path |

---

## 3. Team Structure (3 Members)

| Member | Pillar | Primary Responsibility | Secondary |
|---|---|---|---|
| **A** | **Data & Pipeline** | Dataset sourcing, keypoint extraction, preprocessing, augmentation | Dataset quality validation, integration testing |
| **B** | **ML & Training** | Model architecture (LSTM/GRU), training loop, evaluation, optimization | Data augmentation experiments, bug triage |
| **C** | **App & LLM** | Webcam UI (Streamlit), LLM prompt engineering, end-to-end integration, demo | Evaluation scripting, presentation |

---

## 4. System Architecture (Complete Detail)

### 4.1 High-Level Pipeline

```
┌──────────┐    ┌──────────────────┐    ┌────────────────┐    ┌──────────────┐    ┌────────────────────┐    ┌────────────┐
│  Webcam  │───▶│ MediaPipe        │───▶│ Sequence       │───▶│ LSTM / GRU   │───▶│ Gemini API         │───▶│ Streamlit  │
│  OpenCV  │    │ Holistic         │    │ Windowing      │    │ Classifier   │    │ Sentence Builder   │    │ Display    │
│  30 FPS  │    │ Keypoint Extract │    │ 30-frame buf   │    │ word + conf  │    │ gloss → sentence   │    │ UI         │
└──────────┘    └──────────────────┘    └────────────────┘    └──────────────┘    └────────────────────┘    └────────────┘
   Stage 1            Stage 2               Stage 3               Stage 4               Stage 5                Stage 6
```

### 4.2 Stage-by-Stage Detail

| Stage | Input | Output | Tech | Latency |
|---|---|---|---|---|
| 1. **Video Capture** | Webcam @ 30 FPS | BGR frames (640×480) | OpenCV `VideoCapture` | < 1ms |
| 2. **Keypoint Extraction** | Single frame | 225-dim float vector | MediaPipe Holistic | ~15ms |
| 3. **Sequence Windowing** | Stream of 225-dim vectors | `(1, 30, 225)` tensor | NumPy ring buffer | < 1ms |
| 4. **Sign Classification** | `(1, 30, 225)` tensor | `(word_id, confidence)` | PyTorch Bi-LSTM | ~10ms |
| 5. **Sentence Reconstruction** | List of `(word, conf)` | English sentence string | Gemini 2.0 Flash | ~200ms |
| 6. **Display** | Glosses + sentence | Side-by-side UI | Streamlit | Immediate |

**Total: ~230ms per sign** (target < 500ms ✅)

---

### 4.3 Stage 1 — Video Capture (OpenCV)

```python
# How it works:
cap = cv2.VideoCapture(0)               # 0 = default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()              # frame: numpy (480, 640, 3) uint8 BGR
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe needs RGB
    # → pass frame_rgb to Stage 2
```

**Key details:**
- Resolution: 640×480 (reduce to 320×240 if laptop is slow)
- FPS: 30 target, but we process every frame independently
- Color: OpenCV reads BGR, MediaPipe needs RGB — must convert

---

### 4.4 Stage 2 — Keypoint Extraction (MediaPipe Holistic)

**What MediaPipe gives us per frame:**

| Body Part | Landmark Count | × (x, y, z) | Dims | Used? |
|---|---|---|---|---|
| Pose (torso, arms, legs) | 33 | × 3 | 99 | ✅ Yes |
| Left Hand (fingers, palm) | 21 | × 3 | 63 | ✅ Yes |
| Right Hand (fingers, palm) | 21 | × 3 | 63 | ✅ Yes |
| Face (mesh) | 468 | × 3 | 1404 | ❌ No (too many dims, Day 6 decision) |
| **Total used** | **75** | | **225** | |

```python
# Per-frame extraction:
mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,              # 0=fast, 1=balanced, 2=accurate
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

results = mp_holistic.process(frame_rgb)

# Extract into flat array:
keypoints = []

# Pose: 33 landmarks × 3
if results.pose_landmarks:
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])     # normalized 0.0–1.0
else:
    keypoints.extend([0.0] * 99)                    # zeros if not detected

# Left Hand: 21 × 3
if results.left_hand_landmarks:
    for lm in results.left_hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])
else:
    keypoints.extend([0.0] * 63)

# Right Hand: 21 × 3
if results.right_hand_landmarks:
    for lm in results.right_hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])
else:
    keypoints.extend([0.0] * 63)

frame_vector = np.array(keypoints, dtype=np.float32)  # shape: (225,)
```

**Critical detail:** When a hand is NOT visible, MediaPipe returns `None` — we fill with zeros. If > 30% of frames in a video have missing hands, we skip that video (Bug D5).

**Coordinate system:** All values are normalized to `[0.0, 1.0]` relative to the frame dimensions. `(0, 0)` = top-left corner. `z` represents depth (relative, not metric).

---

### 4.5 Stage 3 — Sequence Windowing (Ring Buffer)

Signs happen over time (~0.5–3 seconds). We collect 30 consecutive frames into a single tensor:

```python
# Ring buffer collects frames:
buffer = deque(maxlen=30)              # rolling window of 30 frames

# Each frame:
buffer.append(frame_vector)            # frame_vector: (225,)

# When buffer is full:
if len(buffer) == 30:
    sequence = np.array(buffer)        # shape: (30, 225)
    tensor = torch.FloatTensor(sequence).unsqueeze(0)  # shape: (1, 30, 225)
    # → pass tensor to Stage 4
```

**For training:** Each video is pre-processed into a fixed-length `(30, 225)` sequence.
- Videos shorter than 30 frames: **pad with zeros** at the end
- Videos longer than 30 frames: **randomly crop** a 30-frame window (augmentation)

---

### 4.6 Stage 4 — Sign Classifier: Bidirectional LSTM (Primary Model)

This is the core ML model. Full implementation detail:

#### 4.6.1 Complete Model Architecture

```python
class ISLRecognizer(nn.Module):
    """
    Bidirectional LSTM with Attention for Isolated Sign Recognition.
    
    Input:  (batch, seq_len=30, input_dim=225)
    Output: (batch, num_classes)  — log-probabilities per class
    """
    def __init__(self, input_dim=225, hidden_dim=256, num_layers=2,
                 num_classes=65, dropout=0.3, classifier_dropout=0.4):
        super().__init__()
        
        # --- Layer 1: Input Normalization ---
        # Normalizes each keypoint dimension across the batch
        # Prevents different body sizes from dominating
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # --- Layer 2-3: Bidirectional LSTM ---
        # Processes sequence in BOTH directions (forward + backward)
        # Each direction has hidden_dim=256, concatenated = 512
        self.lstm = nn.LSTM(
            input_size=input_dim,       # 225
            hidden_size=hidden_dim,     # 256
            num_layers=num_layers,      # 2 stacked layers
            batch_first=True,           # input shape: (batch, seq, features)
            bidirectional=True,         # forward + backward = 2 × hidden_dim
            dropout=dropout             # 0.3 between LSTM layers
        )
        
        # --- Layer 4: Attention Pooling ---
        # Learns which frames are most important for classification
        self.attention = AttentionPooling(hidden_dim * 2)  # 512
        
        # --- Layer 5: Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),     # 512 → 512
            nn.ReLU(),
            nn.Dropout(classifier_dropout),                 # 0.4
            nn.Linear(hidden_dim * 2, num_classes)          # 512 → 65
        )
    
    def forward(self, x):
        # x shape: (batch, 30, 225)
        
        # Step 1: Normalize inputs
        # BatchNorm1d expects (batch, features), so we reshape
        batch, seq_len, features = x.shape
        x = x.reshape(batch * seq_len, features)    # (batch*30, 225)
        x = self.input_norm(x)                       # (batch*30, 225)
        x = x.reshape(batch, seq_len, features)      # (batch, 30, 225)
        
        # Step 2: LSTM processing
        # lstm_out contains hidden states for ALL timesteps
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch, 30, 512)  — 256 forward + 256 backward
        
        # Step 3: Attention pooling
        # Weighted combination of all 30 timestep outputs
        context = self.attention(lstm_out)
        # context shape: (batch, 512)
        
        # Step 4: Classification
        logits = self.classifier(context)
        # logits shape: (batch, 65)
        
        return logits  # raw logits (CrossEntropyLoss applies softmax internally)
```

#### 4.6.2 Attention Pooling Layer

```python
class AttentionPooling(nn.Module):
    """
    Learns which timesteps (frames) matter most for classification.
    
    Instead of taking the last frame (loses early info) or averaging
    all frames (treats all equally), attention learns to focus on
    the peak/discriminative frames of each sign.
    
    Input:  (batch, seq_len, hidden_dim)   e.g., (32, 30, 512)
    Output: (batch, hidden_dim)            e.g., (32, 512)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weight = nn.Linear(hidden_dim, 1, bias=True)
    
    def forward(self, lstm_output):
        # lstm_output: (batch, 30, 512)
        
        # Score each timestep
        scores = self.attention_weight(lstm_output)  # (batch, 30, 1)
        scores = scores.squeeze(-1)                   # (batch, 30)
        
        # Convert to probabilities (which frames to focus on)
        alpha = torch.softmax(scores, dim=1)          # (batch, 30)
        
        # Weighted sum of LSTM outputs
        alpha = alpha.unsqueeze(-1)                    # (batch, 30, 1)
        context = (alpha * lstm_output).sum(dim=1)     # (batch, 512)
        
        return context
```

**Why attention matters:** For the sign "hello" (a wave), the most informative frame is the hand at its highest point — not the frames before/after the wave. Attention learns to weight that peak frame highest.

#### 4.6.3 Tensor Shape Flow (Complete Trace)

```
Input video: 30 frames of webcam

Frame 1:  (480, 640, 3) uint8
Frame 2:  (480, 640, 3) uint8
...
Frame 30: (480, 640, 3) uint8
        │
        ▼ MediaPipe (per frame)
        
Keypoints per frame: (225,) float32
        │
        ▼ Stack 30 frames
        
Sequence: (30, 225) float32
        │
        ▼ Batch (e.g., 32 samples)
        
Batch input: (32, 30, 225)
        │
        ▼ BatchNorm1d
        
After norm: (32, 30, 225)        — same shape, normalized values
        │
        ▼ Bi-LSTM Layer 1
        
LSTM L1 out: (32, 30, 512)      — 256 forward + 256 backward per timestep
        │
        ▼ Bi-LSTM Layer 2
        
LSTM L2 out: (32, 30, 512)      — refined representations
        │
        ▼ Attention Pooling
        
Attention weights: (32, 30)      — probability over 30 frames (sums to 1.0)
Context vector:    (32, 512)     — weighted sum of all timesteps
        │
        ▼ Dense(512→512) + ReLU + Dropout(0.4)
        
Hidden: (32, 512)
        │
        ▼ Dense(512→65)
        
Logits: (32, 65)                 — raw scores per class
        │
        ▼ softmax (during inference only)
        
Probabilities: (32, 65)         — probability distribution over 65 words
        │
        ▼ argmax + max
        
Output: predicted_class=8 (="doctor"), confidence=0.92
```

#### 4.6.4 Parameter Count Breakdown

| Layer | Parameters | Calculation |
|---|---|---|
| BatchNorm1d(225) | 450 | 225 × 2 (weight + bias) |
| LSTM Layer 1 | 985,088 | 4 × ((225+512) × 256) × 2 directions |
| LSTM Layer 2 | 1,572,864 | 4 × ((512+512) × 256) × 2 directions |
| Attention Linear | 513 | 512 × 1 + 1 bias |
| Dense(512→512) | 262,656 | 512 × 512 + 512 |
| Dense(512→65) | 33,345 | 512 × 65 + 65 |
| **Total** | **~2.85M** | Trains in ~5 min/epoch on T4 GPU |

#### 4.6.5 Fallback Model: GRU

```python
class ISLRecognizerGRU(nn.Module):
    """
    Simpler GRU model. Use if LSTM is too slow or overfitting.
    ~400K params vs ~2.85M for LSTM.
    """
    def __init__(self, input_dim=225, hidden_dim=128, num_layers=2,
                 num_classes=65, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,         # Unidirectional (simpler)
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 30, 225)
        gru_out, _ = self.gru(x)          # (batch, 30, 128)
        pooled = gru_out.mean(dim=1)      # (batch, 128) — global avg pool
        logits = self.classifier(pooled)  # (batch, 65)
        return logits
```

---

### 4.7 Training Pipeline (Complete Detail)

#### 4.7.1 Dataset Class

```python
class ISLDataset(Dataset):
    """
    Loads pre-extracted .npy keypoint files.
    Each .npy file = one video = variable-length sequence of (N, 225) frames.
    Returns fixed-length (30, 225) tensors via padding/cropping.
    """
    def __init__(self, file_paths, labels, seq_len=30, augment=False):
        self.file_paths = file_paths    # List of .npy file paths
        self.labels = labels            # List of integer class IDs
        self.seq_len = seq_len          # Fixed sequence length (30)
        self.augment = augment          # Apply augmentation during training
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        keypoints = np.load(self.file_paths[idx])  # (num_frames, 225)
        label = self.labels[idx]
        
        # --- Pad or crop to fixed length ---
        if len(keypoints) >= self.seq_len:
            # Random crop (augmentation) or center crop (eval)
            if self.augment:
                start = random.randint(0, len(keypoints) - self.seq_len)
            else:
                start = (len(keypoints) - self.seq_len) // 2
            keypoints = keypoints[start:start + self.seq_len]
        else:
            # Pad with zeros at the end
            pad = np.zeros((self.seq_len - len(keypoints), 225), dtype=np.float32)
            keypoints = np.concatenate([keypoints, pad], axis=0)
        
        # --- Apply augmentation ---
        if self.augment:
            keypoints = self.apply_augmentation(keypoints)
        
        return torch.FloatTensor(keypoints), torch.LongTensor([label]).squeeze()
    
    def apply_augmentation(self, kp):
        # Random scaling: ±10%
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            kp = kp * scale
        
        # Random translation: shift x,y by ±5%
        if random.random() > 0.5:
            shift = np.random.uniform(-0.05, 0.05, size=(1, 225))
            kp = kp + shift.astype(np.float32)
        
        # Gaussian noise: simulate MediaPipe jitter
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.01, kp.shape).astype(np.float32)
            kp = kp + noise
        
        # Frame dropout: zero out 1-3 random frames
        if random.random() > 0.5:
            n_drop = random.randint(1, 3)
            drop_idx = random.sample(range(len(kp)), n_drop)
            kp[drop_idx] = 0.0
        
        return kp
```

#### 4.7.2 Training Loop

```python
def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        # batch_x: (batch, 30, 225)   batch_y: (batch,)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)                        # (batch, 65)
        loss = criterion(logits, batch_y)              # scalar
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients — Bug M2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / len(dataloader), correct / total

# --- Full training setup ---
model = ISLRecognizer(num_classes=65).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

# Class-weighted loss (handles imbalance — Bug M4)
class_counts = [count_per_class]  # from dataset analysis
weights = 1.0 / torch.FloatTensor(class_counts)
weights = weights / weights.sum() * len(class_counts)
criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

# Early stopping
best_val_acc = 0
patience_counter = 0

for epoch in range(100):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    
    # Early stopping (patience=10)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch}")
            break
```

#### 4.7.3 Inference (Getting prediction + confidence)

```python
model.eval()
with torch.no_grad():
    logits = model(input_tensor)               # (1, 65)
    probabilities = torch.softmax(logits, dim=1)  # (1, 65)
    confidence, predicted_class = probabilities.max(dim=1)
    
    word = id_to_word[predicted_class.item()]   # e.g., "doctor"
    conf = confidence.item()                     # e.g., 0.92
    
    if conf < 0.5:
        word = "UNKNOWN"                         # Below confidence threshold
```

---

### 4.8 Stage 5 — Gloss Buffer (Deduplication)

Between the classifier and the LLM, a buffer handles the noisy stream of per-frame predictions:

```python
class GlossBuffer:
    """
    Deduplicates the stream of predictions.
    
    Problem: Classifier runs every frame, so "doctor" might be predicted
    30 times in a row. We only want it once.
    
    Logic:
    1. Suppress consecutive identical predictions
    2. Reject predictions below confidence threshold
    3. Accumulate unique glosses until 3+ words or 5-second timeout
    4. Send accumulated glosses to LLM for sentence building
    """
    def __init__(self, conf_threshold=0.5, min_glosses=3, timeout_sec=5.0):
        self.conf_threshold = conf_threshold
        self.min_glosses = min_glosses
        self.timeout_sec = timeout_sec
        self.glosses = []                # [(word, confidence), ...]
        self.last_word = None
        self.last_add_time = time.time()
    
    def add_prediction(self, word, confidence):
        # Skip low confidence
        if confidence < self.conf_threshold:
            return
        
        # Skip consecutive duplicates
        if word == self.last_word:
            return
        
        self.glosses.append((word, confidence))
        self.last_word = word
        self.last_add_time = time.time()
    
    def should_send(self):
        """Check if we have enough glosses to build a sentence."""
        if len(self.glosses) >= self.min_glosses:
            return True
        if len(self.glosses) > 0 and (time.time() - self.last_add_time) > self.timeout_sec:
            return True
        return False
    
    def flush(self):
        """Return accumulated glosses and reset buffer."""
        result = list(self.glosses)
        self.glosses = []
        self.last_word = None
        return result
```

```
Example data flow:

Frame-by-frame predictions:
  Frame 1:  hello (0.95)
  Frame 2:  hello (0.93)  ← suppressed (consecutive duplicate)
  Frame 3:  hello (0.91)  ← suppressed
  Frame 4:  UNKNOWN (0.35)← suppressed (below threshold)
  Frame 5:  doctor (0.88)
  Frame 6:  doctor (0.85) ← suppressed
  Frame 7:  help (0.72)
  Frame 8:  help (0.69)   ← suppressed
  Frame 9:  need (0.61)

Buffer after dedup: [hello (0.95), doctor (0.88), help (0.72), need (0.61)]
                                     ↓
4 glosses ≥ min_glosses (3) → send to LLM
```

---

### 4.9 Stage 5 — Sentence Reconstruction (Gemini API)

```python
class SentenceBuilder:
    """
    Converts raw glosses into grammatically correct sentences.
    Primary: Gemini API. Fallback: rule-based template.
    """
    SYSTEM_PROMPT = """You are an Indian Sign Language (ISL) translator.

You will receive a sequence of English words (glosses) detected from ISL signing,
along with a confidence score (0.0 to 1.0) for each word.

Rules:
1. Reconstruct into ONE grammatically correct English sentence.
2. Confidence ≥ 0.7 = HIGH — preserve exactly.
3. Confidence < 0.7 = LOW — may reinterpret if context helps.
4. DO NOT invent new information.
5. You may add articles, prepositions, tense markers.
6. Context: hospital/reception scenario.
7. Output ONLY the final sentence.

Examples:
Input:  [doctor (0.92), need (0.85), help (0.78)]
Output: I need help from the doctor.

Input:  [stomach (0.91), pain (0.95), bad (0.80)]
Output: I have bad stomach pain.

Input:  [hello (0.99), name (0.88), what (0.75)]
Output: Hello, what is your name?

Input:  [medicine (0.88), when (0.72), see (0.65)]
Output: When should I take the medicine?

Input:  [water (0.90), drink (0.82), please (0.95)]
Output: Can I please have some water to drink?"""

    def __init__(self, api_key, timeout=3, use_fallback=True):
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        genai.configure(api_key=api_key)
        self.timeout = timeout
        self.use_fallback = use_fallback
        # Pre-cache demo sentences for instant response
        self.cache = {}
    
    def build_sentence(self, glosses):
        """
        glosses: list of (word, confidence) tuples
        Returns: grammatically correct English sentence
        """
        # Check cache first (for demo reliability)
        cache_key = tuple(w for w, c in glosses)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Format input
        gloss_str = ", ".join(f"{w} ({c:.2f})" for w, c in glosses)
        prompt = f"Glosses: [{gloss_str}]"
        
        try:
            response = self.model.generate_content(
                [self.SYSTEM_PROMPT, prompt],
                generation_config={"temperature": 0.3, "max_output_tokens": 100}
            )
            return response.text.strip()
        except Exception:
            if self.use_fallback:
                return self._template_fallback(glosses)
            return " ".join(w for w, c in glosses)
    
    def _template_fallback(self, glosses):
        """Rule-based fallback when API is unavailable."""
        words = [w for w, c in glosses]
        
        # Simple SVO reordering heuristic
        subjects = {"I", "you", "mother", "father", "family", "doctor", "who"}
        verbs = {"help", "sit", "wait", "come", "go", "eat", "drink", "sleep",
                 "walk", "stop", "give", "see", "need", "hurt"}
        
        subj = [w for w in words if w in subjects]
        verb = [w for w in words if w in verbs]
        rest = [w for w in words if w not in subjects and w not in verbs]
        
        if not subj:
            subj = ["I"]
        
        sentence = " ".join(subj + verb + rest)
        return sentence.capitalize() + "."
    
    def preload_cache(self, demo_sentences):
        """Pre-cache demo sentences for zero-latency during presentation."""
        for glosses, sentence in demo_sentences:
            cache_key = tuple(w for w, c in glosses)
            self.cache[cache_key] = sentence
```

---

### 4.10 Stage 6 — Streamlit UI

```
┌──────────────────────────────────────────────────────────┐
│                    🤟 ISHARA                              │
│          ISL to Text Translation System                   │
├────────────────────────┬─────────────────────────────────┤
│                        │  Detected Signs:                 │
│   📹 Live Webcam       │  ┌───────────────────────────┐  │
│   Feed with            │  │ doctor ██████████░░ 0.92  │  │
│   Keypoint Overlay     │  │ help   ████████░░░░ 0.78  │  │
│   (MediaPipe drawing)  │  │ need   ████████░░░░ 0.85  │  │
│                        │  └───────────────────────────┘  │
│                        │                                  │
│                        │  📝 Translated Sentence:         │
│                        │  ┌───────────────────────────┐  │
│                        │  │ "I need help from the     │  │
│                        │  │  doctor."                 │  │
│                        │  └───────────────────────────┘  │
├────────────────────────┴─────────────────────────────────┤
│  [▶ Start]  [⏹ Stop]  [🗑 Clear]     Avg Confidence: 85% │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Dataset Strategy (Datasets Only — No Manual Recording)

> [!IMPORTANT]
> **No manual recording.** All training data comes from public datasets. Vocabulary is constrained to words available in these datasets. If a word isn't available with ≥ 8 samples, we replace it with a synonym or drop it.

### 5.1 Primary: INCLUDE Dataset (IIT Madras / AI4Bharat)

| Property | Value |
|---|---|
| Words | 263 words across 15 categories |
| Videos | 4,287 total (avg ~16 per word) |
| Subset | INCLUDE-50: 50 words, 958 videos (for rapid prototyping) |
| Splits | `train.csv` (3,475 videos), `test.csv` (817 videos) — signer-aware |
| Download | `http://bit.ly/include_dl` |
| Website | [sign-language.ai4bharat.org](https://sign-language.ai4bharat.org/) |

### 5.2 Supplementary (gap-fill only)

| Dataset | Words | Videos | Access | Use |
|---|---|---|---|---|
| **CISLR** (Joshi et al.) | 4,765 | 7,050 | HuggingFace | Fill medical words missing from INCLUDE |
| **Kaggle ISL datasets** | Varies | Varies | Kaggle | Extra samples for low-count classes |
| **ISLTranslate** (Joshi et al.) | 1,036 | Sentence-level | GitHub | Reference for ISL grammar (LLM prompt design only) |

### 5.3 Vocabulary Themes (all from public datasets)

| Category | Example Words | Count | Source |
|---|---|---|---|
| Greetings | hello, goodbye, thank you, please, sorry, yes, no | 8 | INCLUDE |
| Medical | doctor, medicine, hospital, pain, fever, head, stomach, heart, eye, ear | 13 | INCLUDE + CISLR |
| Actions | help, sit, wait, come, go, eat, drink, sleep, walk, stop, give, see | 12 | INCLUDE |
| Questions | what, where, when, how, who | 5 | INCLUDE |
| People/Places | mother, father, family, home, school, name, I, you | 8 | INCLUDE |
| Numbers/Time | one, two, three, four, five, today, tomorrow, morning, night | 9 | INCLUDE |
| Descriptors | good, bad, hot, cold, big, small, happy, water, food, telephone | 10 | INCLUDE |
| **Total** | | **~65** | |

### 5.4 Multi-Dataset Integration Rules

1. **Re-extract everything** — Run ALL videos (INCLUDE + any supplementary) through our own MediaPipe pipeline to get consistent keypoint format
2. **Label alignment** — Map all labels to a single `vocabulary.json` ID mapping
3. **Class balancing** — Use augmentation to bring low-count classes up to median sample count
4. **Split by signer** — Use INCLUDE's provided signer-aware split; for supplementary data, hold out entire video groups

### 5.5 Data Splits

| Split | Source | Purpose |
|---|---|---|
| **Train** | INCLUDE `train.csv` subset + supplementary samples | Model training |
| **Validation** | 15% of train, held out by signer | Hyperparameter tuning, early stopping |
| **Test** | INCLUDE `test.csv` subset (untouched until final eval) | Final accuracy reporting |

---

## 6. Training Configuration

### 6.1 Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | Better generalization than Adam |
| Learning Rate | 1e-3 → 1e-4 (cosine schedule) | Start aggressive, decay smoothly |
| Batch Size | 32–64 | Adjust based on GPU RAM |
| Epochs | 50–100 (early stopping) | Patience = 10 epochs |
| Sequence Length | 30 frames | ~1 second at 30 FPS |
| Gradient Clipping | max_norm = 1.0 | Prevents exploding gradients |
| Weight Decay | 0.01 | Regularization via AdamW |
| Label Smoothing | 0.1 | Prevents overconfident predictions |
| Loss Function | CrossEntropyLoss (class-weighted) | Handles class imbalance |
| Seed | 42 | Reproducibility |

### 6.2 Data Augmentation (Keypoint-Specific)

> [!WARNING]
> **Do NOT apply horizontal flip** — many ISL signs are hand-specific. Flipping changes the sign's meaning.

| Augmentation | What it does | Implementation |
|---|---|---|
| Random Scaling | Scale all keypoints ±10% | `kp *= uniform(0.9, 1.1)` |
| Random Translation | Shift x, y by ±5% | `kp[:, :2] += uniform(-0.05, 0.05)` |
| Random Rotation | Rotate 2D projection ±15° | Rotation matrix on (x, y) |
| Gaussian Noise | Simulate MediaPipe jitter | `kp += normal(0, 0.01)` |
| Temporal Crop | Random 30-frame window from longer clip | `start = randint(0, len-30)` |
| Speed Perturbation | Resample ±20% speed | `np.interp` on temporal axis |
| Frame Dropout | Zero out 1–3 random frames | Simulates tracking loss |

### 6.3 Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU | Google Colab free tier (T4) | Colab Pro (A100) or local RTX 3060+ |
| RAM | 8 GB | 16 GB |
| Storage | 20 GB | 50 GB |
| Webcam | Built-in laptop camera | External 720p+ USB |
| Python | 3.10+ | 3.11 |

---

## 7. Project File Structure

```
Ishara/
├── README.md
├── requirements.txt
├── config.yaml                      # All hyperparams, paths, vocab
├── .gitignore
│
├── data/
│   ├── raw/                         # Raw dataset videos (gitignored)
│   │   ├── include/                 # INCLUDE dataset
│   │   └── supplementary/           # CISLR / Kaggle data
│   ├── processed/                   # Extracted keypoints (.npy)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── vocabulary.json
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
│
├── src/
│   ├── data/
│   │   ├── download_include.py      # Download & filter INCLUDE
│   │   ├── download_supplementary.py # CISLR/Kaggle download
│   │   ├── extract_keypoints.py     # MediaPipe extraction pipeline
│   │   ├── augmentation.py          # Keypoint augmentation transforms
│   │   └── dataset.py               # PyTorch Dataset + DataLoader
│   │
│   ├── model/
│   │   ├── lstm_model.py            # Primary Bi-LSTM architecture
│   │   ├── gru_model.py             # Fallback GRU architecture
│   │   ├── attention.py             # Attention pooling layer
│   │   └── train.py                 # Training loop + logging
│   │
│   ├── inference/
│   │   ├── predictor.py             # Real-time sign prediction
│   │   ├── gloss_buffer.py          # Deduplication & buffering
│   │   └── sentence_builder.py      # Gemini API + template fallback
│   │
│   └── utils/
│       ├── config.py                # Config loader (reads config.yaml)
│       ├── metrics.py               # Accuracy, confusion matrix
│       └── visualize.py             # Keypoint visualization helpers
│
├── app/
│   └── streamlit_app.py             # Main Streamlit application
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/
│   ├── test_keypoint_extraction.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_sentence_builder.py
│
└── demo/
    ├── demo_script.md
    ├── backup_recording.mp4
    └── presentation.pptx
```

---

## 8. Known Bugs, Pitfalls & Mitigations

### 8.1 Data Bugs

| # | Bug | Likelihood | Impact | Fix |
|---|---|---|---|---|
| D1 | **Data leakage via normalization** — normalizing before split | HIGH | Fake 95% accuracy | Compute mean/std on train set ONLY |
| D2 | **Signer leakage** — same signer in train and test | HIGH | Inflated accuracy | Use INCLUDE's signer-aware split |
| D3 | **INCLUDE download fails** — old mirrors | MEDIUM | Blocks Day 1 | Download early; INCLUDE-50 as fallback |
| D4 | **Corrupt/missing videos** | LOW | Fewer samples | Validate counts; drop classes < 8 samples |
| D5 | **MediaPipe fails on bad videos** | MEDIUM | NaN in training | Skip videos with > 30% missing landmarks |
| D6 | **Cross-dataset inconsistency** (INCLUDE vs CISLR) | HIGH | Model learns dataset artifacts | Re-extract ALL through our MediaPipe pipeline |
| D7 | **Desired word not in any dataset** | MEDIUM | Vocabulary gap | Replace with synonym; never record |

### 8.2 Model Bugs

| # | Bug | Likelihood | Impact | Fix |
|---|---|---|---|---|
| M1 | **Overfitting** (~16 samples/word) | VERY HIGH | Low test accuracy | Aggressive augmentation, dropout, early stopping, label smoothing |
| M2 | **Vanishing gradients** in deep LSTM | MEDIUM | Loss plateaus | Gradient clipping, try GRU fallback |
| M3 | **NaN loss** | MEDIUM | Training crashes | Check for NaN in preprocessing; reduce LR |
| M4 | **Class imbalance** | HIGH | Predicts majority class | Weighted CrossEntropyLoss; oversample minorities |
| M5 | **Confused sign pairs** (e.g., hot vs fever) | HIGH | Wrong predictions | Confusion matrix on Day 8; merge or remove worst pairs |
| M6 | **Sequence length mismatch** (0.5s–3s signs) | HIGH | Padding/truncation issues | Dynamic padding; test 20 and 40 frame windows |

### 8.3 Integration Bugs

| # | Bug | Likelihood | Impact | Fix |
|---|---|---|---|---|
| I1 | **MediaPipe + OpenCV version conflict** | HIGH | Import errors | Pin versions in requirements.txt Day 1 |
| I2 | **Webcam latency** > 33ms/frame | MEDIUM | Laggy demo | Profile early; reduce resolution |
| I3 | **Gemini API timeout during demo** | MEDIUM | Demo freezes | 3-second timeout + template fallback |
| I4 | **Streamlit re-runs on every click** | HIGH | Webcam restarts | `st.session_state` for persistence |
| I5 | **Model too large for Git** | LOW | Can't share | `.gitignore` checkpoints; share via Drive |

### 8.4 Demo-Day Bugs

| # | Bug | Likelihood | Impact | Fix |
|---|---|---|---|---|
| DD1 | **Live demo fails** (different room lighting) | HIGH | Embarrassing | Pre-recorded backup video always ready |
| DD2 | **Signer signs differently than training data** | HIGH | Wrong prediction | Practice demo with exact trained signs |
| DD3 | **API key exposed in demo** | LOW | Security issue | Environment variables only |

---

## 9. 14-Day Work Breakdown (Detailed)

> [!IMPORTANT]
> Each day is split into **Morning (M)** and **Afternoon (A)** blocks. Each task lists the **owner**, the **exact deliverable**, and the **done criteria** (how you know it's finished). If a task's done criteria isn't met by end of day, it carries to the next morning as priority.

---

### ═══════════════════════════════════════
### PHASE 1: FOUNDATION (Days 1–4)
### ═══════════════════════════════════════

---

#### 📅 DAY 1 — Project Setup & Data Download
*All 3 members work together. No one works alone today.*

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 1.1 | Create Git repository, set up branching (`main` + feature branches) | C | Repo exists, all 3 can push |
| 1.2 | Create full project folder structure (Section 7 — all directories, empty `__init__.py` files) | C | `tree` command matches Section 7 |
| 1.3 | Write `requirements.txt` with pinned versions (Section 12) | C | `pip install -r requirements.txt` succeeds on all 3 machines |
| 1.4 | Write `.gitignore` (exclude `data/raw/`, `*.npy`, `checkpoints/`, `.env`) | C | Large files won't accidentally be committed |
| 1.5 | Start downloading INCLUDE dataset (full) — this takes hours | A | Download started and verified progressing |
| 1.6 | Download INCLUDE-50 subset separately (smaller, finishes first) | A | INCLUDE-50 available locally |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 1.7 | Browse INCLUDE dataset contents — list all 263 word categories, count videos per word | A | Spreadsheet/CSV of word → video count |
| 1.8 | Build `vocabulary.json` — map INCLUDE words to hospital scenario; pick 50–80 words; assign integer IDs | All | JSON file with `word_to_id` and `id_to_word`, every word has ≥ 8 videos in INCLUDE |
| 1.9 | Write `config.yaml` with all hyperparameters from Section 6 | B | Config file matches all values in Section 6.1 |
| 1.10 | Create Google Cloud project, get Gemini API key, set as env variable `GEMINI_API_KEY` | C | `python -c "import google.generativeai as genai; genai.configure(api_key=...)"` works |
| 1.11 | Set up shared Google Drive folder for model checkpoints + dataset backup | A | All 3 members have access |

**Day 1 Deliverables:**
- ✅ Git repo with full skeleton structure
- ✅ Dependencies installed on all 3 machines
- ✅ INCLUDE-50 downloaded; full INCLUDE downloading overnight
- ✅ `vocabulary.json` finalized (50–80 words confirmed available in INCLUDE)
- ✅ `config.yaml` written
- ✅ Gemini API key working

**⚠️ Day 1 Risk Check:** If INCLUDE download link is dead, immediately switch to CISLR on HuggingFace (see Section 11.2, R1).

---

#### 📅 DAY 2 — Keypoint Extraction Pipeline + Model Stub
*Team splits into parallel tracks.*

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 2.1 | Write `src/data/extract_keypoints.py` — MediaPipe extraction (Section 4.4 code) | A | Script runs on a single video and outputs correct `(N, 225)` .npy file |
| 2.2 | Test extraction on 5 sample videos — verify output shapes, no NaN values | A | 5 .npy files, all shapes correct, `np.isnan().sum() == 0` |
| 2.3 | Write `src/data/dataset.py` — PyTorch Dataset class (Section 4.7.1 code) | B | `Dataset.__getitem__` returns `(tensor(30, 225), tensor(label))` |
| 2.4 | Write `src/model/lstm_model.py` — `ISLRecognizer` class (Section 4.6.1 code) | B | `model(torch.randn(2, 30, 225))` returns shape `(2, 65)` without error |
| 2.5 | Write `src/model/attention.py` — `AttentionPooling` class (Section 4.6.2 code) | B | Attention returns `(batch, 512)` from `(batch, 30, 512)` input |
| 2.6 | Start building Streamlit app skeleton — webcam feed display | C | `streamlit run app/streamlit_app.py` opens browser with live webcam |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 2.7 | Launch batch keypoint extraction on INCLUDE-50 (50 words, ~958 videos) | A | Extraction running; estimated finish time known |
| 2.8 | Add MediaPipe keypoint overlay to Streamlit UI (draw skeleton on webcam feed) | C | Can see hand/body landmarks drawn on live video |
| 2.9 | Test Gemini API with 5 sample prompts from Section 4.9 | C | All 5 return grammatically correct sentences |
| 2.10 | Write `src/utils/config.py` — config loader (reads `config.yaml`) | B | `from src.utils.config import load_config; cfg = load_config()` works |
| 2.11 | Verify LSTM model parameter count matches Section 4.6.4 (~2.85M) | B | `sum(p.numel() for p in model.parameters())` ≈ 2.85M |

**Day 2 Deliverables:**
- ✅ Keypoint extraction script working and processing INCLUDE-50
- ✅ LSTM model compiles and forward pass works
- ✅ Streamlit shows live webcam with keypoint overlay
- ✅ Gemini API produces correct sentences from sample inputs

---

#### 📅 DAY 3 — First Training Run + LLM Pipeline
*A finishes extraction. B does first training. C builds sentence pipeline.*

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 3.1 | Verify INCLUDE-50 extraction complete — check .npy file counts per word | A | All 50 words have .npy files; log any failures |
| 3.2 | Start extraction on remaining words (if vocab > 50 words from INCLUDE full) | A | Extraction running for full vocabulary |
| 3.3 | Create signer-aware train/val/test split CSVs using INCLUDE's provided split | A | `train_split.csv`, `val_split.csv`, `test_split.csv` — no signer overlap |
| 3.4 | Run first training on INCLUDE-50: 10 epochs, batch_size=32, lr=1e-3 | B | Training loop runs; loss decreases epoch-over-epoch; no NaN |
| 3.5 | Write `src/inference/gloss_buffer.py` — `GlossBuffer` class (Section 4.8 code) | C | Unit test: feed `[a,a,a,b,b,c]` → outputs `[a,b,c]` |
| 3.6 | Write `src/inference/sentence_builder.py` — `SentenceBuilder` class (Section 4.9 code) | C | Feed `[("doctor", 0.9), ("help", 0.8)]` → returns grammatical sentence |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 3.7 | Write `src/data/augmentation.py` — all 7 transforms from Section 6.2 | A | Each transform runs on `(30, 225)` array without error |
| 3.8 | Analyze first training results — plot loss curves, check train vs val accuracy | B | Training curve plot saved; know if overfitting or underfitting |
| 3.9 | Write template-based fallback sentence builder (no API) | C | Fallback produces reasonable (not perfect) sentences |
| 3.10 | Test `SentenceBuilder` with 10+ edge cases (empty gloss, single word, all low confidence) | C | All 10 cases produce output without crashing |

**Day 3 Deliverables:**
- ✅ First model trained (10 epochs, sanity check passes)
- ✅ Loss decreasing, no NaN — model is learning *something*
- ✅ Augmentation pipeline built and tested
- ✅ Sentence builder (Gemini + fallback) working end-to-end
- ✅ Train/val/test splits created

**⚠️ Day 3 Accuracy Check:** After 10 epochs on INCLUDE-50, expect train accuracy 15–30%. This is NORMAL for a sanity check. If loss isn't decreasing at all, check for bugs in data loading (wrong labels, wrong shapes).

---

#### 📅 DAY 4 — Full Dataset + Full Training Launch
*Critical day: final data prep, then launch the real training run overnight.*

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 4.1 | Complete keypoint extraction for full vocabulary (50–80 words) | A | .npy files exist for every word in `vocabulary.json` |
| 4.2 | Validate dataset integrity: no NaN, correct shapes `(N, 225)`, per-class sample counts | A | Validation report: every class has ≥ 8 samples, 0 NaN values |
| 4.3 | If any vocab words have < 8 samples: check CISLR for supplementary data OR drop the word | A | `vocabulary.json` updated — every word has ≥ 8 clean samples |
| 4.4 | Build augmented dataset: apply all 7 transforms to create 5–10× more training samples | A | Augmented file count = 5× to 10× original |
| 4.5 | Experiment with hyperparameters: try batch_size={32, 64}, seq_len={20, 30, 40} | B | Results table comparing 3–4 configurations |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 4.6 | Launch FULL training run (50–100 epochs, early stopping patience=10) — **runs overnight** | B | Training started; checkpoint saves working; TensorBoard logging active |
| 4.7 | Add class-weighted loss (Section 4.7.2) to handle class imbalance | B | Loss function uses per-class weights |
| 4.8 | Design 10 test sentences for LLM prompt — each tests a different edge case | C | 10 test cases with expected outputs documented |
| 4.9 | Iterate LLM prompt — add/remove few-shot examples until 9/10 test cases pass | C | ≥ 9/10 test cases produce correct grammar |
| 4.10 | Write `src/utils/metrics.py` — accuracy, top-k accuracy, confusion matrix functions | B | Functions work on dummy predictions |

**Day 4 Deliverables:**
- ✅ Full augmented dataset ready (all vocabulary words, 5–10× augmented)
- ✅ Full training run launched overnight
- ✅ LLM prompt handles ≥ 9/10 edge cases
- ✅ Class-weighted loss implemented
- ✅ Metrics utilities ready

---

### ═══════════════════════════════════════
### PHASE 2: CORE DEVELOPMENT (Days 5–9)
### ═══════════════════════════════════════

---

#### 📅 DAY 5 — Analyze Overnight Training + Build Real-Time Predictor

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 5.1 | Pull overnight training results: final accuracy, best epoch, loss curves | B | Know exact train acc, val acc, and test acc |
| 5.2 | Generate per-class accuracy breakdown — which words are working, which aren't | B | Sorted list: best-performing → worst-performing words |
| 5.3 | Check for overfitting: train acc >> val acc means too few samples / too little augmentation | B | Decision made: need more augmentation? Need to drop classes? |
| 5.4 | If accuracy < 40%: begin escalation (Section 11.4 Kill Switch) — reduce vocab | B + A | Decision logged |
| 5.5 | Find classes with poor keypoint extraction (too many zero-fill frames) | A | List of problematic classes with % missing data |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 5.6 | Write `src/inference/predictor.py` — connects webcam → MediaPipe → model → output | C | Script that runs live: shows webcam + prints predicted word per second |
| 5.7 | Retrain with fixes if needed (adjust LR, add more dropout, etc.) | B | Second training run launched |
| 5.8 | Fix/re-extract any problematic classes found in 5.5 | A | Re-extracted .npy files replace bad ones |
| 5.9 | Test predictor with dummy model (random predictions) just to verify pipeline works | C | Pipeline runs without crashes for 60 seconds |

**Day 5 Deliverables:**
- ✅ Know exact accuracy numbers (train/val/test)
- ✅ Per-class accuracy breakdown
- ✅ Real-time predictor script running (even with imperfect predictions)
- ✅ Kill switch decision made if accuracy < 40%

---

#### 📅 DAY 6 — Architecture Decision Day (LSTM vs GRU, Face Landmarks)

> [!WARNING]
> **DAY 6 IS THE KILL SWITCH.** If accuracy is below 40% after trying both architectures, you MUST reduce vocabulary (see Section 11.4).

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 6.1 | Write `src/model/gru_model.py` — GRU fallback model (Section 4.6.5 code) | B | GRU model compiles and trains |
| 6.2 | Train GRU on same dataset — compare accuracy to LSTM | B | Side-by-side: LSTM accuracy vs GRU accuracy |
| 6.3 | Test attention pooling vs global average pooling on the better model | B | Attention vs AvgPool comparison |
| 6.4 | **DECISION: Face landmarks?** If accuracy < 55%, try adding face (225 → 1629 dims) | B + A | Decision logged in `config.yaml` |
| 6.5 | If adding face: re-extract keypoints with face landmarks (only for top-50 words to save time) | A | .npy files with 1629 dims |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 6.6 | Augmentation ablation: train 4 runs — (no aug, scaling only, noise only, all aug) | A | Table showing accuracy for each combo |
| 6.7 | Connect real predictor to best model checkpoint — live webcam predictions | C | Webcam feed shows predicted word updating in real time |
| 6.8 | **DECISION CHECKPOINT**: Pick final architecture (LSTM or GRU, attention or AvgPool, face or no face) | All | Decision documented, `config.yaml` updated |

**Day 6 Deliverables:**
- ✅ LSTM vs GRU comparison table
- ✅ Attention vs AvgPool comparison
- ✅ Augmentation ablation results
- ✅ **Final architecture chosen and locked**
- ✅ Live predictions visible in predictor script

**🚨 Kill Switch Check:** If best accuracy < 40%, IMMEDIATELY reduce vocabulary to 20–30 words and retrain tonight.

---

#### 📅 DAY 7 — Best Model Training + LLM Hardening

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 7.1 | Launch final training run with best config: architecture + augmentation + schedule | B | Training started with cosine LR scheduler |
| 7.2 | Implement cosine annealing LR scheduler (Section 6.1) | B | LR drops from 1e-3 → 1e-4 over training |
| 7.3 | Test augmentation with speed perturbation + rotation (the two most impactful) | A | Results show ≥ 3% improvement with these augmentations |
| 7.4 | Test LLM prompt with 20 edge cases including: wrong word order, repeated words, single word, all low confidence, mixed ISL/English | C | ≥ 18/20 produce acceptable output |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 7.5 | Build complete `_template_fallback()` method (Section 4.9 code) | C | Works without internet; produces readable sentences |
| 7.6 | Test fallback on same 20 cases — compare to Gemini output | C | Fallback produces reasonable (not perfect) output for all 20 |
| 7.7 | Write `src/utils/visualize.py` — keypoint drawing helpers for Streamlit | A | Function that draws skeleton on frame |
| 7.8 | Run best model training overnight | B | Training in progress |

**Day 7 Deliverables:**
- ✅ Best model training with final architecture + cosine LR
- ✅ LLM prompt handles 18/20 edge cases
- ✅ Template fallback works offline
- ✅ Target: val accuracy ≥ 55%

---

#### 📅 DAY 8 — 🎉 Full Pipeline Integration

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 8.1 | Load best model checkpoint into `predictor.py` | C | Predictor uses trained model, not dummy |
| 8.2 | Connect predictor → gloss_buffer → sentence_builder in Streamlit app | C | Sign 3 words in sequence → sentence appears |
| 8.3 | Wire up `st.session_state` to prevent Streamlit re-runs from killing webcam | C | Clicking buttons does NOT restart webcam feed |
| 8.4 | Pull overnight training results — final model accuracy | B | Know final test accuracy |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 8.5 | **END-TO-END TEST:** Each team member signs 3 known signs → system produces sentence | All | Works for at least 1 out of 3 people |
| 8.6 | Profile latency: measure time from sign completion to sentence display | B + C | Latency < 500ms per sign (or identify bottleneck) |
| 8.7 | Fix any crashes, race conditions, or UI glitches found during E2E test | C | System runs for 5 minutes without crashing |
| 8.8 | If latency > 500ms: reduce resolution OR process every 2nd frame OR use GRU | B | Latency within target |
| 8.9 | Test on all 3 team members' machines — same setup, same model | All | Works on at least 2/3 machines |

**Day 8 Deliverables:**
- ✅ 🎉 **FIRST WORKING END-TO-END DEMO**
- ✅ Sign in front of webcam → sentence displayed
- ✅ Latency < 500ms
- ✅ Works on ≥ 2 machines

---

#### 📅 DAY 9 — Testing, Confusion Analysis, Bug Fixes

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 9.1 | Generate full confusion matrix for all classes | B | Confusion matrix heatmap saved as image |
| 9.2 | Identify top-10 most confused sign pairs (e.g., "hot" ↔ "fever") | B | List of 10 pairs with confusion rates |
| 9.3 | For each confused pair: decide — merge into one class, drop one, or add more augmentation | B + A | Decision for each pair documented |
| 9.4 | Retrain with fixes (merged/dropped classes, more targeted augmentation) | B | Retrained model improving on confused pairs |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 9.5 | Test with 2+ people OUTSIDE the team (friends, classmates) who weren't in training data | All | Accuracy on external signers measured and logged |
| 9.6 | Fix all bugs found during Day 8 integration | C | Zero known crashes |
| 9.7 | Write basic test scripts: `test_model.py`, `test_sentence_builder.py` | A | `pytest tests/` passes |
| 9.8 | If external signer accuracy < 30%: apply per-sample normalization to keypoints | A + B | Keypoints normalized relative to shoulder width |

**Day 9 Deliverables:**
- ✅ Confusion matrix with top-10 confused pairs addressed
- ✅ Tested on ≥ 2 external signers
- ✅ All integration bugs fixed
- ✅ Basic test suite passing

---

### ═══════════════════════════════════════
### PHASE 3: POLISH & DEMO (Days 10–14)
### ═══════════════════════════════════════

---

#### 📅 DAY 10 — Threshold Tuning & Error Handling

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 10.1 | Sweep confidence threshold: test 0.3, 0.4, 0.5, 0.6, 0.7 — find optimal | B | Threshold chosen that balances precision vs recall |
| 10.2 | Add "Unknown Sign ❓" display in UI for below-threshold predictions | C | Low-confidence predictions show as "Unknown" instead of wrong word |
| 10.3 | Add error handling: camera disconnected, API timeout, model load failure | C | Each failure mode shows user-friendly message, no stack traces |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 10.4 | Optimize inference speed if needed: try model quantization (`torch.quantization`) | B | Inference latency measured before and after |
| 10.5 | Final dataset cleanup: remove any corrupted files found during testing | A | Clean dataset, no bad files |
| 10.6 | Test complete pipeline for 10 minutes straight — no crashes, no memory leaks | All | 10-minute stability test passes |

---

#### 📅 DAY 11 — UI Polish & Demo Script

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 11.1 | Polish Streamlit UI: dark theme, colored confidence bars, clean layout (Section 4.10) | C | UI looks professional, not prototype-y |
| 11.2 | Add "About" section, project title with logo/emoji, team names | C | UI has branding |
| 11.3 | Select 5–8 demo sentences that the model handles reliably | All | List of sentences with ≥ 80% success rate |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 11.4 | Write `demo/demo_script.md` — exact sequence: which person signs, what they sign, what to say | All | Script timed at ≤ 12 min (leaving 3 min buffer for 15-min slot) |
| 11.5 | Practice demo 2× with the script — note any weak points | All | 2 dry runs completed |
| 11.6 | Prepare evaluation metrics charts: accuracy bar chart, confusion matrix, latency graph | B | 3–4 presentation-ready charts |
| 11.7 | Pre-cache the 5–8 demo sentences in `SentenceBuilder.cache` for zero-latency response | C | Demo sentences return instantly without API call |

---

#### 📅 DAY 12 — Feature Freeze & Backup Recording

> [!IMPORTANT]
> **🔒 FEATURE FREEZE at end of Day 12.** NO new code after today. Only bug fixes allowed Days 13–14.

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 12.1 | Code freeze: `git tag v1.0-freeze` | All | Tag pushed |
| 12.2 | Test in actual demo room/environment if accessible | All | Know exact lighting, camera angle, background |
| 12.3 | Save final model checkpoint as `checkpoints/model_final.pth` | B | Checkpoint on Google Drive + local |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 12.4 | Record backup demo video: 3 takes of complete demo, pick the best | All | `demo/backup_recording.mp4` exists and looks good |
| 12.5 | Draft presentation slides: title, problem, architecture diagram, demo, results, limitations, future work | C | ≥ 12 slides drafted |
| 12.6 | Verify backup video plays cleanly: test on projector/screen if possible | All | Video plays without issues |

---

#### 📅 DAY 13 — Presentation Prep & Dress Rehearsal

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 13.1 | Finalize presentation slides — add charts from 11.6, polish design | C | Complete deck ready |
| 13.2 | Assign speaking roles: who presents which section | All | Each person knows their slides |
| 13.3 | Full dress rehearsal #1: complete presentation + live demo (timed) | All | Run-through ≤ 15 min |

**Afternoon:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 13.4 | Fix anything that broke in rehearsal | All | Bug-fixes only |
| 13.5 | Full dress rehearsal #2: with fixes applied | All | Smoother than rehearsal #1 |
| 13.6 | Prepare demo machine: charge laptop, pre-load model into memory, test webcam, verify API key | C | Machine ready, model loads in < 10 seconds |
| 13.7 | Finalize `README.md` — clean project documentation | A | README matches actual project state |
| 13.8 | Backup everything: full repo + model + data on Google Drive | A | Complete backup exists |

---

#### 📅 DAY 14 — 🎯 DEMO DAY

**Morning:**

| # | Task | Owner | Done When |
|---|---|---|---|
| 14.1 | Light rehearsal: run through key demo signs once, no code changes | All | Confidence verified |
| 14.2 | Pre-demo checks: charge laptop to 100%, test webcam, test API, load model, test backup video | C | All systems go |
| 14.3 | Arrive early at demo venue, set up, test in actual room lighting | All | System works in venue |

**Presentation:**

| # | Task | Owner |
|---|---|---|
| 14.4 | **Present slides** (problem → approach → architecture → demo) | All |
| 14.5 | **Live demo:** Demo signer performs 5–8 pre-practiced sign sequences | Designated signer |
| 14.6 | **If live demo fails:** Immediately switch to `backup_recording.mp4` — no debugging on stage | C |
| 14.7 | **Q&A:** Handle questions about accuracy, limitations, future work | All |

**🎉 DONE.**

---

## 10. Success Criteria

| Metric | Target | Stretch Goal |
|---|---|---|
| Test accuracy (isolated signs) | ≥ 60% | ≥ 75% |
| Top-3 accuracy | ≥ 80% | ≥ 90% |
| Vocabulary size | 50 words | 80 words |
| Inference latency per sign | < 500ms | < 200ms |
| LLM sentence quality | 80% grammatically correct | 95%+ |
| Live demo | Works with backup | Works live |

---

## 11. Honest Risk Analysis & Contingency Plans

> [!CAUTION]
> **This plan is NOT foolproof.** No ML project plan is. Below is a brutally honest assessment of what can go wrong and what to do when it does. The key insight: **every risk has a degraded-but-working fallback.**

### 11.1 Risk Matrix

| # | Risk | Probability | Impact | Trigger Point |
|---|---|---|---|---|
| R1 | INCLUDE download link dead / corrupt | **MEDIUM-HIGH** | Critical | Day 1 |
| R2 | INCLUDE words don't map to our hospital vocabulary | **MEDIUM** | High | Day 1–2 |
| R3 | Accuracy stays below 50% (too few samples) | **MEDIUM** | High | Day 5–6 |
| R4 | Model fails on demo signer (signer dependence) | **HIGH** | High | Day 8–9 |
| R5 | Team member unavailable for 2+ days | Medium | High | Any day |
| R6 | Gemini API down during demo | Low | Medium | Day 14 |
| R7 | Laptop too slow for real-time inference | Low | Medium | Day 8 |
| R8 | Scope creep ("let's add one more feature") | **HIGH** | Medium | Day 10+ |

### 11.2 Contingency Plans (If X Happens, Do Y)

#### R1: INCLUDE Dataset Unavailable

**Detection:** Day 1, download fails or bit.ly link is dead.

| Action | Timeline |
|---|---|
| Try alternate sources: Kaggle mirrors, AI4Bharat GitHub, contact IIT Madras directly | 2 hours |
| If still blocked: use **CISLR from HuggingFace** as primary dataset instead (4,765 words, 7,050 videos) | Same day |
| If ALL datasets fail: use the many **Kaggle ISL/ASL hand gesture datasets** (smaller but downloadable instantly) and reduce vocabulary to 20–30 words | Same day |

#### R2: INCLUDE Words Don't Map to Hospital Vocabulary

**Detection:** Day 1–2, when mapping INCLUDE's 263 words to our list.

| Action | Timeline |
|---|---|
| **Adjust vocabulary to what INCLUDE actually has** — don't fight the data. If INCLUDE has "school" but not "hospital", use "school" | Day 1 |
| Change scenario from "hospital" to "daily conversation" if needed — the LLM can handle any scenario context | Day 1 |
| The scenario is just a demo narrative; the sign recognition works regardless of context | Immediate |

#### R3: Accuracy Below 50% (The Big One)

**Detection:** Day 5–6, after first full training run.

**This is the most likely serious problem.** ~16 samples per word across 65 classes is genuinely hard. Here's the escalation ladder:

| Step | Action | Expected Improvement |
|---|---|---|
| 1 | More aggressive augmentation (10–15× instead of 5×) | +5–10% |
| 2 | Reduce vocabulary from 65 → 50 (drop hardest classes) | +5–10% |
| 3 | Reduce to INCLUDE-50 subset only (50 well-curated words) | +5–10% |
| 4 | Drop to **20–30 words** — focus on the most distinct signs | +10–15% |
| 5 | **Nuclear option: 10–15 very distinct signs** (numbers 1–5, hello, thank_you, yes, no, eat, drink, help) — these will be nearly 90%+ accurate | Guaranteed working demo |
| 6 | Switch from LSTM to a simpler **Random Forest / SVM on frame-level features** — less accurate overall but more stable with tiny data | Last resort |

> [!IMPORTANT]
> **The nuclear option (Step 5) ALWAYS works.** 10–15 very distinct signs with ~16 samples each + augmentation = ~85%+ accuracy. The demo just shows fewer words. A demo that correctly recognizes 12 signs is infinitely better than one that randomly guesses 65 signs.

#### R4: Model Fails on Demo Signer (Signer Dependence)

**Detection:** Day 8–9, when testing live.

This is the **#1 most common failure** in sign language projects. The model learns the training signers' body proportions, not the signs.

| Action | Timeline |
|---|---|
| **The demo signer must practice** the exact signs from the training data — speed, hand position, everything | Day 10–12 |
| Apply **per-sample keypoint normalization** (normalize each person's skeleton to a standard scale) | Day 6 |
| Add **more augmentation** specifically targeting scale/translation variance | Day 6–7 |
| **Test with all 3 team members on Day 9** — if it fails for everyone, it's a model problem. If it works for 1/3, it's a signer problem — that person demos | Day 9 |
| **Last resort:** Pre-record the demo with a signer it works for, use video playback | Day 12 |

#### R6: Gemini API Down During Demo

**Detection:** Demo day.

| Action | Timeline |
|---|---|
| Template-based fallback kicks in automatically (built by Day 7) | Instant |
| Pre-cache the 5–8 demo sentences — if the glosses match a cached input, return cached output without API call | Day 11 |
| Audience won't notice the difference for simple sentences | N/A |

#### R7: Laptop Too Slow

**Detection:** Day 8, during integration.

| Action | Timeline |
|---|---|
| Reduce webcam resolution from 640×480 to 320×240 | 5 minutes |
| Process every 2nd frame instead of every frame | 5 minutes |
| Use GRU model instead of LSTM (4× smaller) | Already built |
| Skip keypoint overlay drawing (saves ~5ms/frame) | 5 minutes |

### 11.3 The Guarantee: What WILL You Have on Day 14?

Even in the absolute worst case, you **WILL** have:

| Component | Worst Case | Realistic Case | Best Case |
|---|---|---|---|
| Working pipeline | ✅ Yes (even with 10 words) | ✅ Yes (50 words) | ✅ Yes (80 words) |
| Sign recognition | 10–15 distinct signs, ~85% accuracy | 50 signs, ~60% accuracy | 65+ signs, ~75% accuracy |
| Sentence output | Template-based (no API) | Gemini API, decent quality | Gemini, excellent quality |
| Live demo | Pre-recorded backup | Works live for practiced signs | Works live for any vocabulary sign |
| Presentation | Slides + backup video | Slides + semi-live demo | Slides + impressive live demo |

**The absolute floor is: a pre-recorded demo of 10–15 signs being correctly classified and turned into sentences.** That's still a valid project deliverable.

### 11.4 Day 6 Kill Switch Decision

> [!WARNING]
> **Day 6 is your go/no-go checkpoint.** If accuracy is below 40% on Day 6 after trying both LSTM and GRU, you MUST execute the following immediately:
>
> 1. Reduce vocabulary to 20–30 most distinct signs
> 2. Retrain with aggressive augmentation
> 3. If still below 50% by Day 7 end → go to nuclear option (10–15 signs)
>
> **Do NOT spend Days 7–14 hoping accuracy will magically improve on 65 classes.** Cut scope early, deliver a polished small demo.

---

## 12. Key Dependencies (pin in requirements.txt)

```
torch>=2.0
torchvision>=0.15
mediapipe>=0.10.9
opencv-python>=4.8
numpy>=1.24
pandas>=2.0
streamlit>=1.28
google-generativeai>=0.3
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
pyyaml>=6.0
tqdm>=4.65
tensorboard>=2.14
pytest>=7.4
```

---

## 13. Daily Standup (15 min)

Every day, all 3 members:
1. **What I did yesterday**
2. **What I'm doing today**
3. **What's blocking me**

Track via shared Google Doc or Notion with the Day-by-Day checklist from Section 9.

---

## 14. Why This Plan Works (Honest Summary)

The plan works **not because nothing will go wrong**, but because:

1. **Every component has a fallback** — LSTM fails → GRU. Gemini down → template. 65 words too hard → 20 words. Live demo fails → pre-recorded.
2. **The architecture is proven** — MediaPipe + LSTM for sign recognition is the most published, most reproduced approach in the field. We're not inventing anything novel.
3. **Scope is cuttable** — the vocabulary can shrink from 80 → 50 → 20 → 10 at any checkpoint without changing architecture or code. Only data and config change.
4. **Day 6 kill switch** — forces an early honest assessment instead of discovering on Day 13 that nothing works.
5. **Pre-recorded backup** — guarantees something to show on Day 14 regardless of what happens live.
6. **The LLM layer is the easy part** — Gemini will produce perfect sentences from glosses every time. The hard part is recognition accuracy, and we've planned for degraded accuracy.

The plan does NOT guarantee 75% accuracy on 80 words. It guarantees **a working demo with a sentence output on Day 14.**
