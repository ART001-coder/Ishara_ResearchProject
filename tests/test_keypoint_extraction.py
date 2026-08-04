"""
==============================================================================
Module: tests/test_keypoint_extraction.py
Role: Automated Unit Test Suite for Keypoint Extraction Pipeline
Reference: implementation_plan.md -> Section 4.3, 4.4, Section 8.1 (Bug D5), Day 9
==============================================================================
"""

import pytest
import numpy as np
from src.data.extract_keypoints import extract_landmarks_from_frame


class MockLandmark:
    def __init__(self, x=0.5, y=0.5, z=0.5):
        self.x = x
        self.y = y
        self.z = z


class MockLandmarkList:
    def __init__(self, count):
        self.landmark = [MockLandmark() for _ in range(count)]


class MockResults:
    def __init__(self, has_pose=True, has_left=True, has_right=True):
        self.pose_landmarks = MockLandmarkList(33) if has_pose else None
        self.left_hand_landmarks = MockLandmarkList(21) if has_left else None
        self.right_hand_landmarks = MockLandmarkList(21) if has_right else None


def test_feature_vector_dimension():
    """
    Test 1: Asserts keypoint extraction output vector is 225 dimensions.
    """
    results = MockResults(has_pose=True, has_left=True, has_right=True)
    vector = extract_landmarks_from_frame(results)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (225,)
    assert vector.dtype == np.float32


def test_zero_fill_for_missing_hands():
    """
    Test 2: Verifies missing hand landmarks produce 0.0 zero-fills without NaNs.
    """
    results = MockResults(has_pose=True, has_left=False, has_right=False)
    vector = extract_landmarks_from_frame(results)

    assert vector.shape == (225,)
    assert not np.isnan(vector).any()

    # Pose (indices 0..98) should be non-zero
    assert (vector[:99] == 0.5).all()

    # Left hand (99..161) and Right hand (162..224) should be 0.0
    assert (vector[99:162] == 0.0).all()
    assert (vector[162:225] == 0.0).all()


if __name__ == "__main__":
    test_feature_vector_dimension()
    test_zero_fill_for_missing_hands()
    print("[SUCCESS] All test_keypoint_extraction tests passed.")
