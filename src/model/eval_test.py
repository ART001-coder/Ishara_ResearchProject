"""
==============================================================================
Module: src/model/eval_test.py
Role: Test Split Evaluation & Confusion Matrix Rendering Engine
Reference: implementation_plan.md -> Section 10, Section 8.2 (Bug M5), Section 9 (Day 9)
==============================================================================
"""

import os
import json
import torch
import numpy as np
from src.data.dataset import get_dataloaders
from src.model.lstm_model import ISLRecognizer
from src.utils.metrics import calculate_topk_accuracy, generate_confusion_matrix, find_top_confused_pairs
from src.utils.config import load_config


def run_test_evaluation(config_path: str = "config.yaml") -> None:
    """
    Evaluates best model checkpoint on held-out test split, generates confusion matrix plot,
    and extracts top-10 confused sign pairs.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(config_path)

    model = ISLRecognizer(num_classes=65).to(device)
    best_model_path = cfg['paths']['best_model_path']

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"[SUCCESS] Loaded test checkpoint: {best_model_path}")
    else:
        print(f"[WARNING] Checkpoint '{best_model_path}' not found. Running with initialized weights.")

    model.eval()

    y_true = []
    y_logits = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            if batch_x.size(0) == 0:
                continue
            logits = model(batch_x.to(device))
            y_true.extend(batch_y.numpy())
            y_logits.extend(logits.cpu().numpy())

    if len(y_true) == 0:
        print("[WARNING] Test evaluation split is empty. Skipping evaluation.")
        return

    y_true_arr = np.array(y_true)
    y_logits_arr = np.array(y_logits)
    y_pred_arr = np.argmax(y_logits_arr, axis=1)

    top1_acc = calculate_topk_accuracy(y_true_arr, y_logits_arr, k=1)
    top3_acc = calculate_topk_accuracy(y_true_arr, y_logits_arr, k=3)

    print("\n=================== HELD-OUT TEST EVALUATION ===================")
    print(f"Top-1 Test Accuracy: {top1_acc*100:.2f}%")
    print(f"Top-3 Test Accuracy: {top3_acc*100:.2f}%")
    print("=================================================================\n")

    # Load class names
    vocab_path = cfg['paths'].get('vocabulary_file', 'data/vocabulary.json')
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    id_to_word = vocab_data.get("id_to_word", {})
    class_names = [id_to_word.get(str(i), f"class_{i}") for i in range(65)]

    # Generate Confusion Matrix plot
    cm_path = "logs/confusion_matrix.png"
    cm = generate_confusion_matrix(y_true_arr, y_pred_arr, class_names, output_path=cm_path)

    # Extract Top-10 Confused Pairs (Bug M5 analysis)
    top_confused = find_top_confused_pairs(cm, class_names, top_n=10)
    print("Top 10 Confused Sign Pairs:")
    for true_w, pred_w, count in top_confused:
        print(f"  - True: '{true_w}' --> Predicted: '{pred_w}' ({count} instances)")


if __name__ == "__main__":
    run_test_evaluation()
