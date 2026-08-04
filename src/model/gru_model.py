"""
==============================================================================
Module: src/model/gru_model.py
Role: Fallback Gated Recurrent Unit Classifier Architecture (ISLRecognizerGRU)
Reference: implementation_plan.md -> Section 4.6.5 (~400K params fallback)
==============================================================================
"""

import torch
import torch.nn as nn


class ISLRecognizerGRU(nn.Module):
    """
    Simpler GRU model with Global Average Pooling (~400K params vs ~2.85M for LSTM).
    Use if Bi-LSTM is too slow, overfits, or experiences vanishing gradients.
    
    Input:  (batch, seq_len=30, input_dim=225)
    Output: (batch, num_classes=65) — Raw classification logits
    """
    def __init__(self, input_dim: int = 225, hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 65, dropout: float = 0.3):
        """
        Initializes GRU model parameters and layers.
        """
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass for GRU model.
        
        Args:
            x (torch.Tensor): Input shape (batch, 30, 225)
            
        Returns:
            logits (torch.Tensor): Raw classification logits shape (batch, 65)
        """
        batch, seq_len, features = x.shape
        x_flat = x.reshape(batch * seq_len, features)
        x_norm = self.input_norm(x_flat)
        x_reshaped = x_norm.reshape(batch, seq_len, features)

        # Step 1: GRU processing
        gru_out, _ = self.gru(x_reshaped)                 # (batch, 30, 128)

        # Step 2: Global Average Pooling across temporal timesteps
        pooled = gru_out.mean(dim=1)                      # (batch, 128)

        # Step 3: Classifier head
        logits = self.classifier(pooled)                  # (batch, 65)

        return logits


if __name__ == "__main__":
    dummy_input = torch.randn(4, 30, 225)
    model = ISLRecognizerGRU(num_classes=65)
    logits = model(dummy_input)
    assert logits.shape == (4, 65)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SUCCESS] ISLRecognizerGRU verified. Parameters: {param_count:,}")
