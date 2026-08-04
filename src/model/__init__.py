"""
==============================================================================
Ishara Model Subpackage Initializer (src/model/__init__.py)
Reference: implementation_plan.md -> Section 4.6 & Section 7
==============================================================================

Modules:
- lstm_model: Primary Bidirectional LSTM architecture (ISLRecognizer).
- gru_model: Fallback Gated Recurrent Unit architecture (ISLRecognizerGRU).
- attention: Learned Attention Pooling layer (AttentionPooling).
- train: Complete PyTorch training loop, validation, gradient clipping, and logging.
"""
