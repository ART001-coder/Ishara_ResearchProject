"""
==============================================================================
Module: src/data/augmentation.py
Role: Keypoint Sequence Data Augmentation Pipeline (7 Transforms)
Reference: implementation_plan.md -> Section 6.2 & Section 8.2 (Bug M1), Day 3
==============================================================================
"""

import random
import numpy as np
from typing import Tuple


class KeypointAugmentor:
    """
    Applies spatial and temporal augmentations to keypoint sequences.
    
    IMPORTANT: Horizontal flip is strictly DISABLED to prevent sign meaning distortion.
    """
    def __init__(self, 
                 scale_range: Tuple[float, float] = (0.9, 1.1), 
                 shift_range: Tuple[float, float] = (-0.05, 0.05), 
                 rotation_deg: float = 15.0, 
                 noise_std: float = 0.01, 
                 dropout_max: int = 3):
        """
        Initializes augmentation parameters.
        """
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.rotation_deg = rotation_deg
        self.noise_std = noise_std
        self.dropout_max = dropout_max

    def random_scale(self, kp: np.ndarray) -> np.ndarray:
        """
        Scales keypoint coordinates by a random factor between scale_range[0] and scale_range[1].
        Shape: (seq_len, 225)
        """
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        return kp * scale_factor

    def random_translate(self, kp: np.ndarray) -> np.ndarray:
        """
        Shifts x,y coordinates by random offset between shift_range[0] and shift_range[1].
        Applies offset strictly to x and y coordinates (indices 0 and 1 of 3D triplets).
        """
        kp_copy = kp.copy()
        seq_len, total_dim = kp_copy.shape
        num_landmarks = total_dim // 3

        shift_x = random.uniform(self.shift_range[0], self.shift_range[1])
        shift_y = random.uniform(self.shift_range[0], self.shift_range[1])

        # Reshape to (seq_len, num_landmarks, 3) to apply translation cleanly
        kp_reshaped = kp_copy.reshape(seq_len, num_landmarks, 3)
        kp_reshaped[:, :, 0] += shift_x  # x offset
        kp_reshaped[:, :, 1] += shift_y  # y offset

        return kp_reshaped.reshape(seq_len, total_dim)

    def random_rotate(self, kp: np.ndarray) -> np.ndarray:
        """
        Rotates 2D (x,y) keypoints around center origin (0.5, 0.5) by random angle in [-rotation_deg, +rotation_deg].
        """
        angle_rad = np.radians(random.uniform(-self.rotation_deg, self.rotation_deg))
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        kp_copy = kp.copy()
        seq_len, total_dim = kp_copy.shape
        num_landmarks = total_dim // 3

        kp_reshaped = kp_copy.reshape(seq_len, num_landmarks, 3)

        # Center rotation around normalized frame center (0.5, 0.5)
        x = kp_reshaped[:, :, 0] - 0.5
        y = kp_reshaped[:, :, 1] - 0.5

        x_rot = x * cos_a - y * sin_a + 0.5
        y_rot = x * sin_a + y * cos_a + 0.5

        kp_reshaped[:, :, 0] = x_rot
        kp_reshaped[:, :, 1] = y_rot

        return kp_reshaped.reshape(seq_len, total_dim)

    def add_gaussian_noise(self, kp: np.ndarray) -> np.ndarray:
        """
        Adds zero-mean Gaussian jitter to keypoints to simulate MediaPipe extraction noise.
        """
        noise = np.random.normal(0, self.noise_std, size=kp.shape).astype(np.float32)
        return kp + noise

    def apply_frame_dropout(self, kp: np.ndarray) -> np.ndarray:
        """
        Randomly zeroes out 1 to dropout_max frames in the sequence to simulate missing detection frames.
        """
        seq_len = len(kp)
        if seq_len <= 1:
            return kp

        n_drop = random.randint(1, min(self.dropout_max, seq_len - 1))
        drop_indices = random.sample(range(seq_len), n_drop)

        kp_copy = kp.copy()
        kp_copy[drop_indices] = 0.0
        return kp_copy

    def apply_all(self, kp: np.ndarray) -> np.ndarray:
        """
        Chains random spatial and temporal augmentations with 50% probability each.
        
        Args:
            kp (np.ndarray): Keypoint sequence tensor of shape (seq_len, 225).
            
        Returns:
            np.ndarray: Augmented keypoint sequence tensor of shape (seq_len, 225).
        """
        augmented = kp.copy()

        if random.random() > 0.5:
            augmented = self.random_scale(augmented)
        if random.random() > 0.5:
            augmented = self.random_translate(augmented)
        if random.random() > 0.5:
            augmented = self.random_rotate(augmented)
        if random.random() > 0.5:
            augmented = self.add_gaussian_noise(augmented)
        if random.random() > 0.5:
            augmented = self.apply_frame_dropout(augmented)

        return augmented.astype(np.float32)


if __name__ == "__main__":
    dummy_seq = np.ones((30, 225), dtype=np.float32)
    augmentor = KeypointAugmentor()
    aug_seq = augmentor.apply_all(dummy_seq)
    assert aug_seq.shape == (30, 225)
    assert not np.isnan(aug_seq).any()
    print("[SUCCESS] KeypointAugmentor verified successfully.")
