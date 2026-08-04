"""
==============================================================================
Module: src/inference/predictor.py
Role: Real-Time Sign Prediction Engine (Webcam Stream Processor)
Reference: implementation_plan.md -> Section 4.1, 4.7.3, Section 8.3 (Bug I2), Day 5/8
==============================================================================
"""

import os
import json
import torch
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from typing import Tuple, Dict, Optional
from src.data.extract_keypoints import extract_landmarks_from_frame, get_holistic_detector
from src.model.lstm_model import ISLRecognizer
from src.model.gru_model import ISLRecognizerGRU
from src.utils.config import load_config
from src.utils.visualize import draw_landmarks_on_frame, render_prediction_overlay


class RealtimePredictor:
    """
    Real-time ISL gesture recognition engine connecting OpenCV video stream ->
    MediaPipe Holistic -> 30-frame Ring Buffer -> PyTorch Model -> Class Prediction.
    """
    def __init__(self, 
                 model_path: str = "checkpoints/best_model.pth", 
                 vocab_path: str = "data/vocabulary.json", 
                 config_path: str = "config.yaml", 
                 device: str = "cpu"):
        """
        Initializes MediaPipe Holistic, ring buffer, model weights, and vocabulary mapping.
        """
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        # 30-frame rolling window ring buffer (Section 4.5)
        self.seq_len = self.cfg['mediapipe'].get('sequence_length', 30)
        self.frame_buffer = deque(maxlen=self.seq_len)

        # MediaPipe Holistic landmark extractor helper
        model_comp = self.cfg['mediapipe'].get('model_complexity', 1)
        self.holistic = get_holistic_detector(model_complexity=model_comp)

        # Load vocabulary mapping JSON
        self.vocab_path = vocab_path
        self.id_to_word: Dict[int, str] = {}
        self.load_vocab()

        # Instantiate & load PyTorch model weights
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_vocab(self) -> None:
        """Loads id_to_word dictionary from vocabulary.json."""
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_id_to_word = data.get("id_to_word", {})
            self.id_to_word = {int(k): v for k, v in raw_id_to_word.items()}
        else:
            print(f"[WARNING] Vocabulary file '{self.vocab_path}' not found. Using default index mapping.")
            self.id_to_word = {i: f"class_{i}" for i in range(self.cfg['model']['num_classes'])}

    def load_model(self) -> None:
        """Instantiates neural network architecture and loads checkpoint state dict."""
        num_classes = self.cfg['model']['num_classes']
        arch = self.cfg['model']['architecture']

        if arch == 'bilstm':
            self.model = ISLRecognizer(
                input_dim=self.cfg['model']['input_dim'],
                hidden_dim=self.cfg['model']['hidden_dim'],
                num_layers=self.cfg['model']['num_layers'],
                num_classes=num_classes,
                dropout=self.cfg['model']['dropout'],
                classifier_dropout=self.cfg['model']['classifier_dropout']
            )
        else:
            self.model = ISLRecognizerGRU(
                input_dim=self.cfg['model']['input_dim'],
                hidden_dim=self.cfg['model']['hidden_dim'],
                num_layers=self.cfg['model']['num_layers'],
                num_classes=num_classes,
                dropout=self.cfg['model']['dropout']
            )

        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[SUCCESS] Loaded model weights from checkpoint: {self.model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint '{self.model_path}': {e}. Running with initialized weights.")
        else:
            print(f"[INFO] Model checkpoint '{self.model_path}' not found. Running with initialized weights.")

        self.model.eval().to(self.device)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Processes single webcam frame and returns annotated frame RGB image, predicted word string, and confidence.
        
        Args:
            frame_bgr (np.ndarray): Input OpenCV BGR frame (H, W, 3).
            
        Returns:
            Tuple[np.ndarray, str, float]: (annotated_frame_rgb, predicted_word_string, confidence_score)
        """
        if frame_bgr is None:
            return np.zeros((480, 640, 3), dtype=np.uint8), "UNKNOWN", 0.0

        # Step 1: Convert BGR frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Step 2: Run MediaPipe Holistic Landmark Extraction
        results = self.holistic.process(frame_rgb)

        # Check hand landmark presence to avoid classifying empty/idle zero vectors
        has_left = hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks is not None
        has_right = hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks is not None

        if not has_left and not has_right:
            self.frame_buffer.clear()
            annotated_frame = draw_landmarks_on_frame(frame_rgb, results)
            return annotated_frame, "IDLE (Raise Hands to Sign)", 0.0

        # Step 3: Extract 225-dim keypoint vector and append to ring buffer
        vector = extract_landmarks_from_frame(results)
        self.frame_buffer.append(vector)

        predicted_word = "BUFFERING"
        confidence = 0.0

        # Step 4: Perform model inference when 30-frame sequence buffer is full
        if len(self.frame_buffer) == self.seq_len:
            sequence_array = np.array(self.frame_buffer, dtype=np.float32)  # (30, 225)
            tensor_input = torch.FloatTensor(sequence_array).unsqueeze(0).to(self.device) # (1, 30, 225)

            with torch.no_grad():
                logits = self.model(tensor_input)                          # (1, num_classes)
                probs = torch.softmax(logits, dim=1)                       # (1, num_classes)
                conf_tensor, class_id_tensor = probs.max(dim=1)

            confidence = float(conf_tensor.item())
            class_id = int(class_id_tensor.item())

            conf_threshold = self.cfg['inference'].get('confidence_threshold', 0.15)

            if confidence >= conf_threshold:
                predicted_word = self.id_to_word.get(class_id, f"class_{class_id}")
            else:
                predicted_word = self.cfg['inference'].get('unknown_label', "UNKNOWN")

        # Step 5: Draw skeleton landmarks and top prediction banner overlay on frame
        annotated_frame = draw_landmarks_on_frame(frame_rgb, results)
        if predicted_word != "BUFFERING":
            annotated_frame = render_prediction_overlay(annotated_frame, predicted_word, confidence)

        return annotated_frame, predicted_word, confidence

    def reset_buffer(self) -> None:
        """Clears sequence ring buffer."""
        self.frame_buffer.clear()

    def close(self) -> None:
        """Releases MediaPipe resources."""
        if hasattr(self.holistic, 'close'):
            self.holistic.close()


if __name__ == "__main__":
    predictor = RealtimePredictor()
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_out, word, conf = predictor.process_frame(dummy_bgr)
    assert frame_out.shape == (480, 640, 3)
    assert isinstance(word, str)
    print(f"[SUCCESS] RealtimePredictor verified. Initial prediction: '{word}' ({conf*100:.1f}%)")
