"""
==============================================================================
Ishara Inference Subpackage Initializer (src/inference/__init__.py)
Reference: implementation_plan.md -> Section 4.7.3, 4.8, 4.9 & Section 7
==============================================================================

Modules:
- predictor: Real-time sign prediction engine connecting MediaPipe -> Model -> Output.
- gloss_buffer: Prediction deduplication, confidence filtering, and temporal window buffer.
- sentence_builder: LLM API sentence reconstruction (Gemini 2.0 Flash) with rule-based fallback.
"""
