"""
==============================================================================
Ishara Data Subpackage Initializer (src/data/__init__.py)
Reference: implementation_plan.md -> Section 5 & Section 7
==============================================================================

Modules:
- download_include: Automated downloader and filter for INCLUDE dataset.
- download_supplementary: Dataset loader for CISLR and Kaggle supplementary sources.
- extract_keypoints: MediaPipe Holistic pose and hand landmark extractor.
- augmentation: Spatial and temporal keypoint sequence transforms (7 augmentations).
- dataset: PyTorch Dataset and DataLoader class handling padding, cropping, and batching.
"""
