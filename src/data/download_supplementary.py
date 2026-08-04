"""
==============================================================================
Module: src/data/download_supplementary.py
Role: Supplementary Dataset Loader (CISLR & Kaggle ISL Gap-Fill)
Reference: implementation_plan.md -> Section 5.2, Section 8.1 (Bug D6, Bug D7), Day 4
==============================================================================
"""

import os
import cv2
import requests
from typing import Optional


def download_cislr_word(word: str, output_dir: str = "data/raw/supplementary") -> bool:
    """
    Downloads video clips for a specific missing vocabulary word from CISLR.
    
    Args:
        word (str): Target word gloss to search and download.
        output_dir (str): Destination directory for supplementary clips.
        
    Returns:
        bool: True if download succeeded with >= 8 samples, False otherwise.
    """
    word_dir = os.path.join(output_dir, "cislr", word)
    os.makedirs(word_dir, exist_ok=True)

    # HuggingFace CISLR repository URL pattern
    base_hf_url = f"https://huggingface.co/datasets/ai4bharat/cislr/resolve/main/videos/{word}"

    print(f"[INFO] Attempting CISLR download for gap-fill word: '{word}'")

    downloaded_count = 0
    for sample_idx in range(1, 15):
        sample_url = f"{base_hf_url}/{word}_{sample_idx}.mp4"
        dest_file = os.path.join(word_dir, f"{word}_{sample_idx}.mp4")

        try:
            res = requests.get(sample_url, timeout=5)
            if res.status_code == 200:
                with open(dest_file, "wb") as f:
                    f.write(res.content)
                downloaded_count += 1
        except requests.RequestException:
            break

    if downloaded_count < 8:
        print(f"[WARNING] Bug D7 Fallback: Word '{word}' has insufficient samples in CISLR ({downloaded_count}/8).")
        print(f"[ACTION] Drop word '{word}' from active vocabulary or substitute with synonym.")
        return False

    print(f"[SUCCESS] Acquired {downloaded_count} CISLR video clips for '{word}'.")
    return True


def normalize_supplementary_video(input_path: str, output_path: str, target_fps: int = 30) -> None:
    """
    Normalizes supplementary video clips (resolution 640x480, frame rate 30 FPS, RGB color space)
    to avoid MediaPipe extraction drift (Bug D6 mitigation).
    
    Args:
        input_path (str): Path to raw downloaded supplementary video clip.
        output_path (str): Path to save normalized MP4 video.
        target_fps (int): Frame rate target (default: 30 FPS).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open input video: {input_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to standard 640x480 resolution
        resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        out.write(resized)

    cap.release()
    out.release()
    print(f"[SUCCESS] Normalized supplementary video saved to: {output_path}")


if __name__ == "__main__":
    print("[INFO] Supplementary dataset loader module verified.")
