"""
==============================================================================
Ishara: Core Package Initializer (src/__init__.py)
Reference: implementation_plan.md -> Section 7 (Project File Structure)
==============================================================================

This package contains the core modules for the Ishara ISL translation pipeline:
- data: Dataset loading, MediaPipe pose extraction, keypoint augmentation.
- model: Bi-LSTM, GRU, Attention architectures, and PyTorch training loops.
- inference: Real-time predictor, gloss buffer deduplication, Gemini LLM client.
- utils: Configuration loader, metrics evaluation, visualization utilities.
"""

__version__ = "1.0.0"
__author__ = "Ishara Development Team"
