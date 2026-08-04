"""
==============================================================================
Module: src/utils/config.py
Role: Centralized Configuration Loader & Validation Utility
Reference: implementation_plan.md -> Section 7 & Section 9 (Day 1 Task 1.9 & Day 2 Task 2.10)
==============================================================================
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads and validates system settings from PyYAML configuration file.
    
    Args:
        config_path (str): Relative or absolute path to config.yaml.
        
    Returns:
        dict: Parsed and validated configuration dictionary.
    """
    if not os.path.exists(config_path):
        # Fallback to search relative to project root if running from subdirectory
        alt_path = os.path.join(os.path.dirname(__file__), "..", "..", config_path)
        if os.path.exists(alt_path):
            config_path = os.path.abspath(alt_path)
        else:
            raise FileNotFoundError(f"Configuration file '{config_path}' not found!")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    validate_config(config_dict)
    return config_dict


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validates structural integrity and landmark dimension constraints in config dict.
    
    Args:
        config (dict): Configuration dictionary loaded from YAML.
        
    Returns:
        bool: True if configuration passes all assertions.
    """
    required_sections = ['project', 'paths', 'mediapipe', 'model', 'training', 'inference', 'llm']
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required configuration section: '{section}'")

    # Verify landmark dimensions math (99 pose + 63 left hand + 63 right hand = 225)
    dims = config['mediapipe']['landmark_dims']
    pose = dims.get('pose_landmarks', 99)
    left = dims.get('left_hand_landmarks', 63)
    right = dims.get('right_hand_landmarks', 63)
    total = dims.get('total_feature_dim', 225)

    if pose + left + right != total:
        raise ValueError(
            f"Landmark dimension mismatch: pose({pose}) + left({left}) + right({right}) = {pose+left+right} != total({total})"
        )

    # Check model class count matches target vocabulary size
    num_classes = config['model'].get('num_classes', 65)
    vocab_size = config['project'].get('target_vocab_size', 65)
    if num_classes != vocab_size:
        raise ValueError(f"Model num_classes ({num_classes}) must match project target_vocab_size ({vocab_size})")

    return True


if __name__ == "__main__":
    cfg = load_config()
    print("[SUCCESS] config.yaml successfully loaded and validated.")
    print(f"Project Name: {cfg['project']['name']}")
    print(f"Feature Dimension: {cfg['mediapipe']['landmark_dims']['total_feature_dim']}")
    print(f"Classes: {cfg['model']['num_classes']}")
