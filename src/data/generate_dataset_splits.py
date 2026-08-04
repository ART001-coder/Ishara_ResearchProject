"""
==============================================================================
Module: src/data/generate_dataset_splits.py
Role: Synthetic Dataset Generator & Split Index Builder
Reference: implementation_plan.md -> Section 5.5, Day 1 Task 1.6 & Day 2 Task 2.8
==============================================================================
"""

import os
import json
import numpy as np
import pandas as pd


def create_synthetic_keypoints_and_csv(vocab_path: str = "data/vocabulary.json", 
                                       samples_per_class: int = 10) -> None:
    """
    Generates synthetic landmark keypoint files (.npy) and dataset index CSVs
    (train_split.csv, val_split.csv, test_split.csv) across all 65 vocabulary words.
    
    Args:
        vocab_path (str): Path to vocabulary.json mapping file.
        samples_per_class (int): Number of synthetic keypoint sequences per word.
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file '{vocab_path}' not found!")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data.get("word_to_id", {})

    train_records = []
    val_records = []
    test_records = []

    train_dir = "data/processed/train"
    val_dir = "data/processed/val"
    test_dir = "data/processed/test"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    np.random.seed(42)

    for word, label_id in word_to_id.items():
        word_train_dir = os.path.join(train_dir, word)
        word_val_dir = os.path.join(val_dir, word)
        word_test_dir = os.path.join(test_dir, word)

        os.makedirs(word_train_dir, exist_ok=True)
        os.makedirs(word_val_dir, exist_ok=True)
        os.makedirs(word_test_dir, exist_ok=True)

        for i in range(samples_per_class):
            num_frames = np.random.randint(25, 40)
            base_pattern = np.sin(np.linspace(0, np.pi * 2, num_frames))[:, None] * (label_id + 1) * 0.05
            noise = np.random.normal(0, 0.02, size=(num_frames, 225)).astype(np.float32)
            pattern_expanded = np.tile(base_pattern, (1, 225))
            kp_data = (pattern_expanded + noise).astype(np.float32)

            vid_name = f"{word}_sample_{i+1}.mp4"
            vid_path = f"data/raw/include/{word}/{vid_name}"

            # 70% Train, 15% Val, 15% Test split allocation
            if i < 7:
                npy_path = os.path.join(word_train_dir, f"{word}_sample_{i+1}.npy")
                np.save(npy_path, kp_data)
                train_records.append({
                    "video_path": vid_path,
                    "word_label": word,
                    "label_id": label_id,
                    "signer_id": f"Signer_{(i % 3) + 1}",
                    "dataset_source": "include",
                    "duration_frames": num_frames
                })
            elif i < 9:
                npy_path = os.path.join(word_val_dir, f"{word}_sample_{i+1}.npy")
                np.save(npy_path, kp_data)
                val_records.append({
                    "video_path": vid_path,
                    "word_label": word,
                    "label_id": label_id,
                    "signer_id": "Signer_4",
                    "dataset_source": "include",
                    "duration_frames": num_frames
                })
            else:
                npy_path = os.path.join(word_test_dir, f"{word}_sample_{i+1}.npy")
                np.save(npy_path, kp_data)
                test_records.append({
                    "video_path": vid_path,
                    "word_label": word,
                    "label_id": label_id,
                    "signer_id": "Signer_5",
                    "dataset_source": "include",
                    "duration_frames": num_frames
                })

    # Save CSV Index Files
    df_train = pd.DataFrame(train_records)
    df_val = pd.DataFrame(val_records)
    df_test = pd.DataFrame(test_records)

    df_train.to_csv("data/train_split.csv", index=False)
    df_val.to_csv("data/val_split.csv", index=False)
    df_test.to_csv("data/test_split.csv", index=False)

    print(f"[SUCCESS] Dataset index generated: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test records.")
    print(f"[SUCCESS] Synthetic .npy keypoints created for {len(word_to_id)} vocabulary classes.")


if __name__ == "__main__":
    create_synthetic_keypoints_and_csv()
