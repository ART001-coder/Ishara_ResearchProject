"""
==============================================================================
Module: src/data/extract_keypoints.py
Role: MediaPipe Holistic Pose & Hand Landmark Extraction Pipeline
Reference: implementation_plan.md -> Section 4.3, 4.4 & Section 8.1 (Bug D5), Day 2
==============================================================================
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple
from tqdm import tqdm
from src.utils.config import load_config


def extract_landmarks_from_frame(results) -> np.ndarray:
    """
    Extracts and flattens MediaPipe pose, left hand, and right hand keypoints into a 225-dim vector.
    
    Layout:
        Pose: 33 landmarks * 3 (x, y, z) = 99 dims
        Left Hand: 21 landmarks * 3 (x, y, z) = 63 dims
        Right Hand: 21 landmarks * 3 (x, y, z) = 63 dims
        Total = 225 dims (Face landmarks excluded)
        
    Args:
        results: MediaPipe Holistic process results object.
        
    Returns:
        np.ndarray: 1D numpy array of shape (225,) with dtype float32.
    """
    keypoints = []

    # 1. Pose landmarks (33 * 3 = 99 dims)
    if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
        if hasattr(results.pose_landmarks, 'landmark'):
            lms = results.pose_landmarks.landmark
        elif isinstance(results.pose_landmarks, list) and len(results.pose_landmarks) > 0:
            lms = results.pose_landmarks[0]
        else:
            lms = []

        for lm in lms:
            keypoints.extend([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)])

    if len(keypoints) < 99:
        keypoints.extend([0.0] * (99 - len(keypoints)))
    keypoints = keypoints[:99]

    # 2. Left Hand landmarks (21 * 3 = 63 dims)
    left_start = len(keypoints)
    if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
        if hasattr(results.left_hand_landmarks, 'landmark'):
            lms = results.left_hand_landmarks.landmark
        elif isinstance(results.left_hand_landmarks, list) and len(results.left_hand_landmarks) > 0:
            lms = results.left_hand_landmarks[0]
        else:
            lms = []

        for lm in lms:
            keypoints.extend([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)])

    if len(keypoints) < left_start + 63:
        keypoints.extend([0.0] * (left_start + 63 - len(keypoints)))
    keypoints = keypoints[:162]

    # 3. Right Hand landmarks (21 * 3 = 63 dims)
    right_start = len(keypoints)
    if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
        if hasattr(results.right_hand_landmarks, 'landmark'):
            lms = results.right_hand_landmarks.landmark
        elif isinstance(results.right_hand_landmarks, list) and len(results.right_hand_landmarks) > 0:
            lms = results.right_hand_landmarks[0]
        else:
            lms = []

        for lm in lms:
            keypoints.extend([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)])

    if len(keypoints) < 225:
        keypoints.extend([0.0] * (225 - len(keypoints)))

    return np.array(keypoints[:225], dtype=np.float32)


def get_holistic_detector(model_complexity: int = 1):
    """
    Instantiates MediaPipe Holistic detector via the legacy solutions API.
    Raises loudly on failure instead of silently falling back to a dummy
    detector - a silent fallback here previously caused every video to be
    processed with zero landmarks with no visible error (see Bug D5 postmortem:
    root cause was an incompatible mediapipe pip version, not the videos).
    """
    if not (hasattr(mp, 'solutions') and hasattr(mp.solutions, 'holistic')):
        raise RuntimeError(
            "mediapipe.solutions.holistic is not available in this installed mediapipe "
            "version. Recent mediapipe releases (0.10.31+) dropped the legacy solutions "
            "API on some platforms. Fix: pip install \"mediapipe==0.10.21\" --force-reinstall "
            "--no-deps, then restart the runtime."
        )
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )


def process_video_to_keypoints(video_path: str, output_npy_path: str, model_complexity: int = 1) -> bool:
    """
    Processes a single MP4 video file into an (N, 225) numpy keypoint sequence file.
    
    Args:
        video_path (str): Input video filepath.
        output_npy_path (str): Destination .npy file path.
        model_complexity (int): MediaPipe complexity level (0, 1, or 2).
        
    Returns:
        bool: True if extraction succeeded with < 30% missing hand frames, False otherwise.
    """
    if os.path.exists(output_npy_path):
        try:
            arr = np.load(output_npy_path)
            if arr.ndim >= 2 and arr.shape[1] == 225 and len(arr) > 0:
                return True
        except Exception:
            pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Unable to open video stream: {video_path}")
        return False

    holistic = get_holistic_detector(model_complexity=model_complexity)

    sequence_list = []
    missing_hand_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # Track missing hand frames (Bug D5 mitigation)
        has_left = hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks is not None
        has_right = hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks is not None
        if not has_left and not has_right:
            missing_hand_frames += 1

        vector = extract_landmarks_from_frame(results)
        sequence_list.append(vector)

    cap.release()
    if hasattr(holistic, 'close'):
        holistic.close()

    if total_frames == 0:
        print(f"[SKIP] Video contained 0 readable frames: {video_path}")
        return False

    missing_pct = (missing_hand_frames / total_frames) * 100.0

    # Bug D5 rule: If > 30% of frames lack hand detection, quarantine/skip video clip
    if missing_pct > 30.0:
        print(f"[SKIP] Video '{os.path.basename(video_path)}' has {missing_pct:.1f}% missing hand landmarks (>30% threshold - Bug D5).")
        return False

    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    keypoint_array = np.array(sequence_list, dtype=np.float32)  # Shape: (N, 225)
    np.save(output_npy_path, keypoint_array)

    return True


def batch_extract_dataset(split_csv_path: str, output_processed_dir: str) -> None:
    """
    Batch extracts keypoints for all videos listed in a dataset split CSV.
    """
    import pandas as pd

    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"Split index CSV file not found: {split_csv_path}")

    df = pd.read_csv(split_csv_path)
    print(f"[INFO] Batch keypoint extraction starting for {len(df)} video records in '{split_csv_path}'...")

    success_count = 0
    skip_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Keypoints"):
        video_path = row["video_path"]
        word_label = row["word_label"]
        vid_basename = os.path.splitext(os.path.basename(video_path))[0]
        
        npy_path = os.path.join(output_processed_dir, word_label, f"{vid_basename}.npy")

        if process_video_to_keypoints(video_path, npy_path):
            success_count += 1
        else:
            skip_count += 1

    print(f"[SUCCESS] Keypoint extraction finished. Successfully extracted: {success_count}, Skipped/Quarantined: {skip_count}")


if __name__ == "__main__":
    import argparse
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Batch-extract MediaPipe keypoints for a dataset split.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = parser.parse_args()

    paths = cfg["paths"]
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        batch_extract_dataset(
            split_csv_path=paths[f"{split}_split_csv"],
            output_processed_dir=paths[f"{split}_processed"],
        )
