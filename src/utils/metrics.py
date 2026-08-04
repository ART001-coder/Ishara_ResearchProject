"""
==============================================================================
Module: src/utils/metrics.py
Role: Evaluation Metrics & Confusion Matrix Generation Utilities
Reference: implementation_plan.md -> Section 8.2 (Bug M5), Section 9, Section 10
==============================================================================
"""

import os
import numpy as np
from typing import List, Tuple, Union


def calculate_topk_accuracy(y_true: Union[np.ndarray, list], 
                            y_pred_logits: Union[np.ndarray, list], 
                            k: int = 3) -> float:
    """
    Calculates Top-k accuracy given true class labels and raw output logits/probabilities.
    
    Args:
        y_true (array-like): Ground truth integer class indices, shape (N,).
        y_pred_logits (array-like): Model output raw logits or probabilities, shape (N, num_classes).
        k (int): Number of top predictions to consider (default: 3).
        
    Returns:
        float: Accuracy score between 0.0 and 1.0.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred_logits)

    if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
        return 0.0

    # Get top-k predicted indices for each sample
    # np.argpartition / np.argsort along axis 1
    num_classes = y_pred_arr.shape[1]
    k_actual = min(k, num_classes)
    
    topk_preds = np.argsort(y_pred_arr, axis=1)[:, -k_actual:]

    # Check if ground truth label is within top-k predictions
    correct = 0
    for idx, true_label in enumerate(y_true_arr):
        if true_label in topk_preds[idx]:
            correct += 1

    return float(correct / len(y_true_arr))


def generate_confusion_matrix(y_true: Union[np.ndarray, list], 
                              y_pred: Union[np.ndarray, list], 
                              class_names: List[str], 
                              output_path: str = "logs/confusion_matrix.png") -> np.ndarray:
    """
    Computes confusion matrix and saves an annotated heatmap image plot.
    
    Args:
        y_true (array-like): Ground truth class indices.
        y_pred (array-like): Predicted class indices.
        class_names (list): List of class string names ordered by ID.
        output_path (str): Filepath destination to save heatmap plot.
        
    Returns:
        np.ndarray: Computed confusion matrix array of shape (num_classes, num_classes).
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    num_classes = len(class_names)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, 
                cmap="Blues", fmt="d", cbar=True)
    plt.title("Ishara ISL Sign Recognition - Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[SUCCESS] Confusion matrix saved to: {output_path}")
    return cm


def find_top_confused_pairs(cm: np.ndarray, 
                            class_names: List[str], 
                            top_n: int = 10) -> List[Tuple[str, str, int]]:
    """
    Extracts the top-N most confused sign word pairs from off-diagonal confusion matrix elements.
    
    Args:
        cm (np.ndarray): Confusion matrix array of shape (N, N).
        class_names (list): Class name strings.
        top_n (int): Number of top confused pairs to return.
        
    Returns:
        List[Tuple[str, str, int]]: List of (True_Word, Predicted_Word, Misclassification_Count).
    """
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)  # Ignore correct diagonal predictions

    num_classes = cm_off.shape[0]
    pairs = []

    for i in range(num_classes):
        for j in range(num_classes):
            count = int(cm_off[i, j])
            if count > 0:
                true_word = class_names[i] if i < len(class_names) else f"Class_{i}"
                pred_word = class_names[j] if j < len(class_names) else f"Class_{j}"
                pairs.append((true_word, pred_word, count))

    # Sort descending by misclassification count
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


if __name__ == "__main__":
    dummy_y_true = [0, 1, 2, 0, 1, 2]
    dummy_logits = [
        [0.8, 0.1, 0.1],  # 0 -> 0 (correct)
        [0.2, 0.7, 0.1],  # 1 -> 1 (correct)
        [0.3, 0.6, 0.1],  # 2 -> 1 (misclassified)
        [0.9, 0.05, 0.05],# 0 -> 0 (correct)
        [0.1, 0.8, 0.1],  # 1 -> 1 (correct)
        [0.1, 0.2, 0.7]   # 2 -> 2 (correct)
    ]
    top1 = calculate_topk_accuracy(dummy_y_true, dummy_logits, k=1)
    top3 = calculate_topk_accuracy(dummy_y_true, dummy_logits, k=3)
    print(f"[SUCCESS] Top-1 Acc: {top1*100:.1f}%, Top-3 Acc: {top3*100:.1f}%")

    dummy_cm = np.array([
        [10, 2, 0],
        [1, 15, 4],
        [0, 3, 12]
    ])
    classes = ["hello", "doctor", "help"]
    top_pairs = find_top_confused_pairs(dummy_cm, classes, top_n=3)
    print(f"[SUCCESS] Top confused pairs: {top_pairs}")
