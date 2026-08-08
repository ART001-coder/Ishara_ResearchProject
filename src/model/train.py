"""
==============================================================================
Module: src/model/train.py
Role: Model Training Loop, Evaluation, Optimization & Checkpointing
Reference: implementation_plan.md -> Section 4.7.2, Section 6.1, Section 8.2 (M2/M3/M4)
==============================================================================
"""

import os
import math
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.data.dataset import get_dataloaders
from src.model.lstm_model import ISLRecognizer
from src.model.gru_model import ISLRecognizerGRU
from src.utils.config import load_config


def train_one_epoch(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    criterion: nn.Module, 
                    device: torch.device, 
                    clip_norm: float = 1.0) -> Tuple[float, float]:
    """
    Executes training optimization over a single epoch.
    
    Args:
        model (nn.Module): PyTorch neural network model instance.
        dataloader (DataLoader): Training PyTorch DataLoader.
        optimizer (Optimizer): PyTorch Optimizer instance (AdamW).
        criterion (nn.Module): Loss function (Weighted CrossEntropyLoss).
        device (torch.device): CUDA GPU or CPU computation device.
        clip_norm (float): Gradient clipping maximum norm threshold (default: 1.0).
        
    Returns:
        Tuple[float, float]: (epoch_mean_loss, epoch_accuracy_proportion)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        if batch_x.size(0) == 0:
            continue

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)                         # (batch, num_classes)
        loss = criterion(logits, batch_y)

        # Bug M3 Check: NaN loss detection
        if torch.isnan(loss):
            raise ValueError("[FATAL] NaN loss detected during training batch! Check keypoint normalization.")

        loss.backward()

        # Bug M2 Mitigation: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = logits.max(dim=1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    if total == 0:
        return 0.0, 0.0

    return total_loss / total, correct / total


def evaluate(model: nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             criterion: nn.Module, 
             device: torch.device) -> Tuple[float, float]:
    """
    Evaluates model loss and accuracy performance on validation or test dataset.
    
    Args:
        model (nn.Module): Model instance.
        dataloader (DataLoader): Validation or test DataLoader.
        criterion (nn.Module): Loss criterion.
        device (torch.device): Computation device.
        
    Returns:
        Tuple[float, float]: (validation_mean_loss, validation_accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            if batch_x.size(0) == 0:
                continue

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = logits.max(dim=1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

    if total == 0:
        return 0.0, 0.0

    return total_loss / total, correct / total


def calculate_class_weights(dataloader: torch.utils.data.DataLoader, num_classes: int = 65) -> torch.Tensor:
    """
    Computes inverse class frequency weights to handle class imbalance (Bug M4 mitigation).
    """
    class_counts = [0] * num_classes

    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'labels'):
        for label in dataloader.dataset.labels:
            if 0 <= label < num_classes:
                class_counts[label] += 1
    else:
        for _, batch_y in dataloader:
            for label in batch_y.numpy():
                if 0 <= label < num_classes:
                    class_counts[label] += 1

    counts_arr = np.array(class_counts, dtype=np.float32)
    # Avoid zero division
    counts_arr[counts_arr == 0] = 1.0

    weights = 1.0 / counts_arr
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


def run_training(config_path: str = "config.yaml") -> None:
    """
    Main execution pipeline for model training, optimization, evaluation, and checkpointing.
    """
    cfg = load_config(config_path)
    train_cfg = cfg['training']
    model_cfg = cfg['model']
    paths = cfg['paths']

    # Set random seeds for reproducibility
    seed = train_cfg.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using training device: {device}")

    # Load DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(config_path)

    # Instantiate Model Architecture
    num_classes = model_cfg['num_classes']
    if model_cfg['architecture'] == 'bilstm':
        model = ISLRecognizer(
            input_dim=model_cfg['input_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            num_layers=model_cfg['num_layers'],
            num_classes=num_classes,
            dropout=model_cfg['dropout'],
            classifier_dropout=model_cfg['classifier_dropout']
        ).to(device)
    else:
        model = ISLRecognizerGRU(
            input_dim=model_cfg['input_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            num_layers=model_cfg['num_layers'],
            num_classes=num_classes,
            dropout=model_cfg['dropout']
        ).to(device)

    # Loss Criterion with Class Weighting (Bug M4) & Label Smoothing
    if train_cfg.get('use_class_weights', True) and len(train_loader.dataset) > 0:
        class_weights = calculate_class_weights(train_loader, num_classes).to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, 
        label_smoothing=train_cfg.get('label_smoothing', 0.1)
    )

    # Optimizer & Scheduler Setup
    optimizer = AdamW(
        model.parameters(), 
        lr=train_cfg['learning_rate'], 
        weight_decay=train_cfg.get('weight_decay', 0.01)
    )

    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=train_cfg['num_epochs'], 
        eta_min=train_cfg.get('min_learning_rate', 1e-4)
    )

    # Early Stopping Tracking
    best_val_acc = 0.0
    patience = train_cfg.get('early_stopping_patience', 10)
    patience_counter = 0

    best_model_path = paths['best_model_path']
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    print(f"[INFO] Starting training pipeline for {train_cfg['num_epochs']} epochs...")

    for epoch in range(1, train_cfg['num_epochs'] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            clip_norm=train_cfg.get('gradient_clipping', 1.0)
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{train_cfg['num_epochs']:03d} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f}")

        # Checkpoint Saving & Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model checkpoint (Val Acc: {best_val_acc*100:.2f}%) to '{best_model_path}'")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered after {epoch} epochs (patience={patience}).")
                break

    print(f"[SUCCESS] Training complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the Ishara sign-recognition model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_training(args.config)
