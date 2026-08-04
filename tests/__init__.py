"""
==============================================================================
Ishara Automated Test Suite Package (tests/__init__.py)
Reference: implementation_plan.md -> Section 7 & Section 9 (Day 9 Task 9.7)
==============================================================================

Test Modules:
- test_keypoint_extraction: Tests MediaPipe output dimension shapes (225 dims) and zero-fill handling.
- test_dataset: Tests ISLDataset padding, cropping, augmentation transforms, and DataLoader shapes.
- test_model: Tests Bi-LSTM and GRU model compilation, parameter counts, and forward tensor shapes.
- test_sentence_builder: Tests Gemini API prompt formatting, cache lookup, and offline template fallback.
"""
