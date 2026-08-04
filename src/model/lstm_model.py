"""
==============================================================================
Module: src/model/lstm_model.py
Role: Primary Bidirectional LSTM Classifier Architecture (ISLRecognizer)
Reference: implementation_plan.md -> Section 4.6.1, Section 4.6.3, Section 4.6.4 (~2.85M params)
==============================================================================
"""

import torch
import torch.nn as nn
from src.model.attention import AttentionPooling


class ISLRecognizer(nn.Module):
    """
    Bidirectional LSTM with Attention for Isolated Sign Recognition.
    
    Input:  (batch, seq_len=30, input_dim=225)
    Output: (batch, num_classes=65) — Raw classification logits
    """
    def __init__(self, input_dim: int = 225, hidden_dim: int = 256, num_layers: int = 2,
                 num_classes: int = 65, dropout: float = 0.3, classifier_dropout: float = 0.4):
        """
        Initializes model parameters and neural network layers.
        """
        super().__init__()
        
        # Layer 1: Input Normalization across batch
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Layer 2-3: Stacked Bidirectional LSTM (256 forward + 256 backward = 512)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Layer 4: Attention Pooling Layer
        self.attention = AttentionPooling(hidden_dim * 2)  # 512
        
        # Layer 5: Classification Head with Dropout Regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),     # 512 -> 512
            nn.ReLU(),
            nn.Dropout(classifier_dropout),                 # 0.4
            nn.Linear(hidden_dim * 2, num_classes)          # 512 -> 65
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass pipeline across keypoint sequence input tensor.
        
        Args:
            x (torch.Tensor): Keypoint sequence tensor of shape (batch, 30, 225)
            
        Returns:
            logits (torch.Tensor): Raw classification logits of shape (batch, num_classes=65)
        """
        batch, seq_len, features = x.shape

        # Step 1: Normalize inputs via BatchNorm1d
        x_flat = x.reshape(batch * seq_len, features)     # (batch*30, 225)
        x_norm = self.input_norm(x_flat)                   # (batch*30, 225)
        x_reshaped = x_norm.reshape(batch, seq_len, features) # (batch, 30, 225)

        # Step 2: Bi-LSTM sequence processing
        lstm_out, _ = self.lstm(x_reshaped)                # (batch, 30, 512)

        # Step 3: Attention pooling over sequence timesteps
        context = self.attention(lstm_out)                 # (batch, 512)

        # Step 4: Classification head
        logits = self.classifier(context)                  # (batch, 65)

        return logits


if __name__ == "__main__":
    dummy_input = torch.randn(4, 30, 225)
    model = ISLRecognizer(num_classes=65)
    logits = model(dummy_input)
    assert logits.shape == (4, 65)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SUCCESS] ISLRecognizer Bi-LSTM verified. Parameters: {param_count:,}")
