"""
==============================================================================
Module: src/model/attention.py
Role: Learned Attention Pooling Layer Implementation
Reference: implementation_plan.md -> Section 4.6.2, Day 2 Task 2.5
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Learns which timesteps (frames) in a 30-frame keypoint sequence are most
    discriminative for gesture classification.
    
    Reference: implementation_plan.md -> Section 4.6.2
    """
    def __init__(self, hidden_dim: int = 512):
        """
        Args:
            hidden_dim (int): Dimensionality of incoming LSTM hidden states (default: 512 for Bi-LSTM).
        """
        super().__init__()
        self.attention_weight = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Calculates frame attention weights and outputs weighted context vector representation.
        
        Args:
            lstm_output (torch.Tensor): Output tensor from Bi-LSTM of shape (batch, seq_len=30, hidden_dim=512)
            
        Returns:
            context (torch.Tensor): Weighted output vector of shape (batch, hidden_dim=512)
        """
        # Step 1: Project each timestep to a scalar attention score
        scores = self.attention_weight(lstm_output)        # Shape: (batch, 30, 1)
        scores = scores.squeeze(-1)                        # Shape: (batch, 30)

        # Step 2: Normalize scores along temporal dimension using Softmax
        alpha = F.softmax(scores, dim=1)                   # Shape: (batch, 30)

        # Step 3: Compute weighted sum across sequence timesteps
        alpha_expanded = alpha.unsqueeze(-1)               # Shape: (batch, 30, 1)
        context = (alpha_expanded * lstm_output).sum(dim=1)# Shape: (batch, 512)

        return context


if __name__ == "__main__":
    dummy_lstm_out = torch.randn(4, 30, 512)
    attn_layer = AttentionPooling(hidden_dim=512)
    context = attn_layer(dummy_lstm_out)
    assert context.shape == (4, 512)
    print("[SUCCESS] AttentionPooling module verified successfully.")
