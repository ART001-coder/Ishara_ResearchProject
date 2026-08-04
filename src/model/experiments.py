"""
==============================================================================
Module: src/model/experiments.py
Role: Automated Model Ablation & Experimentation Runner (Bi-LSTM vs GRU)
Reference: implementation_plan.md -> Section 4.6, Section 9 (Day 5 & 6 Tasks 5.1–5.3, 6.1–6.4)
==============================================================================
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Any
from src.data.dataset import get_dataloaders
from src.model.lstm_model import ISLRecognizer
from src.model.gru_model import ISLRecognizerGRU
from src.model.train import train_one_epoch, evaluate
from src.utils.config import load_config


def run_ablation_experiments(config_path: str = "config.yaml", num_epochs: int = 5) -> pd.DataFrame:
    """
    Executes controlled ablation experiments comparing model architectures (Bi-LSTM vs GRU)
    and logs performance metrics for the Day 6 Architecture Decision Checkpoint.
    
    Args:
        config_path (str): Path to configuration YAML file.
        num_epochs (int): Number of epochs per ablation experiment run.
        
    Returns:
        pd.DataFrame: Comparison DataFrame detailing model architectures, parameter counts, train acc, and val acc.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(config_path)
    criterion = nn.CrossEntropyLoss()

    experiments = [
        {"name": "Bi-LSTM + Attention (Primary)", "arch": "bilstm", "hidden_dim": 256},
        {"name": "GRU + AvgPool (Fallback)", "arch": "gru", "hidden_dim": 128}
    ]

    results = []

    for exp in experiments:
        print(f"[INFO] Running ablation experiment: {exp['name']}...")

        if exp["arch"] == "bilstm":
            model = ISLRecognizer(
                input_dim=225,
                hidden_dim=exp["hidden_dim"],
                num_layers=2,
                num_classes=65
            ).to(device)
        else:
            model = ISLRecognizerGRU(
                input_dim=225,
                hidden_dim=exp["hidden_dim"],
                num_layers=2,
                num_classes=65
            ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        best_val_acc = 0.0
        final_train_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            final_train_acc = train_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        results.append({
            "Experiment": exp["name"],
            "Architecture": exp["arch"].upper(),
            "Parameters": param_count,
            "Train Accuracy (%)": round(final_train_acc * 100, 2),
            "Best Val Accuracy (%)": round(best_val_acc * 100, 2),
            "Status": "Passed" if best_val_acc > 0.05 else "Flagged"
        })

    df_results = pd.DataFrame(results)
    print("\n=================== ABLATION EXPERIMENT SUMMARY ===================")
    print(df_results.to_string(index=False))
    print("===================================================================\n")

    return df_results


if __name__ == "__main__":
    df_exp = run_ablation_experiments(num_epochs=5)
