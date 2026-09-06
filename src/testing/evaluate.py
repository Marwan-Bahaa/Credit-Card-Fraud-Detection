import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve(y_true, y_probs, model_name="Model", target_threshold=None):
    # 1. Compute Precision, Recall, and Thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # 2. Compute Average Precision (PR-AUC)
    ap_score = average_precision_score(y_true, y_probs)
    
    # 3. Plot Curve
    plt.figure(figsize=(8, 5))
    plt.plot(recalls, precisions, color='purple', lw=2, label=f'{model_name} (AP = {ap_score:.4f})')
    
    # 4. Highlight operating threshold if provided
    if target_threshold is not None:
        # Find index closest to target threshold
        idx = np.argmin(np.abs(thresholds - target_threshold))
        plt.scatter(
            recalls[idx], precisions[idx], 
            color='red', s=100, zorder=5, 
            label=f'Threshold = {target_threshold:.4f} (P={precisions[idx]:.2f}, R={recalls[idx]:.2f})'
        )

    # Baseline (no-skill model = proportion of positive class)
    no_skill = np.sum(y_true) / len(y_true)
    plt.axhline(y=no_skill, color='navy', linestyle='--', label=f'Baseline ({no_skill:.4f})')

    plt.xlabel('Recall (Sensitivity)', fontsize=11)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=11)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=12)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return ap_score