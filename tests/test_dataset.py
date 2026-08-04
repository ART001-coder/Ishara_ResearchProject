"""
==============================================================================
Module: tests/test_dataset.py
Role: Automated Unit Test Suite for Dataset & DataLoader Modules
Reference: implementation_plan.md -> Section 4.7.1, Section 6.2, Section 8.2 (Bug M6), Day 9
==============================================================================
"""

import os
import pytest
import torch
import numpy as np
from src.data.dataset import ISLDataset
from src.data.augmentation import KeypointAugmentor


def test_dataset_item_shape_padding(tmp_path):
    """
    Test 1: Verifies zero-padding for sequence lengths < 30 frames.
    """
    npy_file = os.path.join(tmp_path, "sample_short.npy")
    np.save(npy_file, np.ones((20, 225), dtype=np.float32))

    dataset = ISLDataset([str(npy_file)], [5], seq_len=30, augment=False)
    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert x.shape == (30, 225)
    assert (x[20:] == 0.0).all()  # Zero padding slice
    assert y.item() == 5


def test_dataset_item_shape_cropping(tmp_path):
    """
    Test 2: Verifies temporal cropping for sequence lengths > 30 frames.
    """
    npy_file = os.path.join(tmp_path, "sample_long.npy")
    np.save(npy_file, np.ones((45, 225), dtype=np.float32))

    dataset = ISLDataset([str(npy_file)], [8], seq_len=30, augment=False)
    x, y = dataset[0]

    assert x.shape == (30, 225)
    assert y.item() == 8


def test_augmentation_transforms():
    """
    Test 3: Verifies keypoint augmentation output shape and valid array bounds.
    """
    augmentor = KeypointAugmentor()
    input_kp = np.ones((30, 225), dtype=np.float32)
    aug_kp = augmentor.apply_all(input_kp)

    assert aug_kp.shape == (30, 225)
    assert not np.isnan(aug_kp).any()


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dataset_item_shape_padding(tmp_dir)
        test_dataset_item_shape_cropping(tmp_dir)
    test_augmentation_transforms()
    print("[SUCCESS] All test_dataset tests passed.")
