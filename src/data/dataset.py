"""
==============================================================================
Module: src/data/dataset.py
Role: PyTorch Dataset & DataLoader Implementation for Keypoint Sequences
Reference: implementation_plan.md -> Section 4.7.1, Section 8.1/8.2, Day 2/3
==============================================================================
"""

import os
import random
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader
from src.data.augmentation import KeypointAugmentor
from src.utils.config import load_config


class ISLDataset(Dataset):
    """
    PyTorch Dataset handling loading, sequence alignment (padding/cropping),
    and augmentation for ISL keypoint sequence tensors.
    """
    def __init__(self, file_paths: List[str], labels: List[int], seq_len: int = 30, augment: bool = False):
        """
        Args:
            file_paths (list): List of .npy keypoint filepath strings.
            labels (list): List of integer class IDs.
            seq_len (int): Fixed temporal window length (default: 30 frames).
            augment (bool): Whether to apply spatial/temporal data augmentation.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.seq_len = seq_len
        self.augment = augment
        self.augmentor = KeypointAugmentor() if augment else None

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads, aligns sequence length, augments, and converts keypoints to PyTorch tensors.
        
        Returns:
            x (torch.FloatTensor): Keypoint sequence tensor of shape (seq_len=30, 225)
            y (torch.LongTensor): Integer class label scalar tensor
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        if os.path.exists(file_path):
            keypoints = np.load(file_path)  # Shape: (N, 225)
        else:
            # Fallback zero tensor if file missing
            keypoints = np.zeros((self.seq_len, 225), dtype=np.float32)

        # Ensure array is 2D float32
        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(-1, 225)
        keypoints = keypoints.astype(np.float32)

        num_frames = len(keypoints)

        # ----------------------------------------------------------------------
        # Sequence Length Alignment (Bug M6 Mitigation: Padding vs Cropping)
        # ----------------------------------------------------------------------
        if num_frames >= self.seq_len:
            if self.augment:
                # Random temporal crop during training
                start = random.randint(0, num_frames - self.seq_len)
            else:
                # Center temporal crop during evaluation
                start = (num_frames - self.seq_len) // 2
            keypoints = keypoints[start : start + self.seq_len]
        else:
            # Zero-padding at the end for short sequences
            pad_length = self.seq_len - num_frames
            pad = np.zeros((pad_length, 225), dtype=np.float32)
            keypoints = np.concatenate([keypoints, pad], axis=0)

        # ----------------------------------------------------------------------
        # Apply Spatial & Temporal Data Augmentation
        # ----------------------------------------------------------------------
        if self.augment and self.augmentor:
            keypoints = self.augmentor.apply_all(keypoints)

        x_tensor = torch.FloatTensor(keypoints)                    # (30, 225)
        y_tensor = torch.tensor(label, dtype=torch.long)          # Scalar

        return x_tensor, y_tensor


def get_dataloaders(config_path: str = "config.yaml") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoader instances for train, val, and test splits based on CSV indexes.
    
    Args:
        config_path (str): Path to config.yaml file.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    cfg = load_config(config_path)
    paths = cfg['paths']
    batch_size = cfg['training']['batch_size']
    seq_len = cfg['mediapipe']['sequence_length']

    # Load CSV split files
    train_csv = paths['train_split_csv']
    val_csv = paths['val_split_csv']
    test_csv = paths['test_split_csv']

    def load_split_data(csv_path: str, processed_dir: str) -> Tuple[List[str], List[int]]:
        if not os.path.exists(csv_path):
            return [], []

        df = pd.read_csv(csv_path)
        file_paths = []
        labels = []

        for _, row in df.iterrows():
            vid_path = row['video_path']
            word = row['word_label']
            label_id = int(row['label_id'])
            vid_basename = os.path.splitext(os.path.basename(vid_path))[0]
            npy_path = os.path.join(processed_dir, word, f"{vid_basename}.npy")

            file_paths.append(npy_path)
            labels.append(label_id)

        return file_paths, labels

    train_files, train_labels = load_split_data(train_csv, paths['train_processed'])
    val_files, val_labels = load_split_data(val_csv, paths['val_processed'])
    test_files, test_labels = load_split_data(test_csv, paths['test_processed'])

    train_dataset = ISLDataset(train_files, train_labels, seq_len=seq_len, augment=True)
    val_dataset = ISLDataset(val_files, val_labels, seq_len=seq_len, augment=False)
    test_dataset = ISLDataset(test_files, test_labels, seq_len=seq_len, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dummy_file = "data/processed/train/dummy.npy"
    os.makedirs(os.path.dirname(dummy_file), exist_ok=True)
    np.save(dummy_file, np.ones((20, 225), dtype=np.float32))

    dataset = ISLDataset([dummy_file], [5], seq_len=30, augment=False)
    x, y = dataset[0]
    assert x.shape == (30, 225)
    assert y.item() == 5
    print("[SUCCESS] ISLDataset verified successfully.")
