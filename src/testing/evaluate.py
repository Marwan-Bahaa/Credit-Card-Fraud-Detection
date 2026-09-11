import numpy as np
import matplotlib.pyplot as plt
import json 
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report, auc  

def cls_report(model, train, val, test, log_file="metrics_history.log"):
    """
    Evaluates classification reports across train, validation, and test splits,
    prints text reports, and appends structured JSON logs to a file.
    """
    splits = {"Train": train, "Validation": val, "Test": test}

    print("=== Classification Reports (Train -> Validation -> Test) ===")

    with open(log_file, "a") as f:
        for split_name, (y_true, y_pred) in splits.items():
            # Generate text report for stdout
            text_report = classification_report(y_true, y_pred)
            print(f"\n--- {split_name} Set ---")
            print(text_report)

            # Generate dictionary report for JSON logging
            dict_report = classification_report(y_true, y_pred, output_dict=True)

            # Add context metadata to log entry
            log_entry = {
                "split": split_name,
                "model": model.__class__.__name__,
                "metrics": dict_report,
            }

            # Append to file
            f.write(json.dumps(log_entry, indent=4))
            f.write("\n" + "=" * 40 + "\n")


def plot_pr_and_threshold_curves(y_true, y_probs, best_threshold=None, metric_name="AP"):
    """
    Plots the Precision-Recall curve alongside Precision/Recall vs. Threshold curves.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth binary labels.
    y_probs : array-like
        Target scores/probabilities.
    best_threshold : float, optional
        Selected decision threshold to mark on the plot.
    metric_name : str, optional
        Name of the optimization metric to display in the title.
        
    Returns:
    --------
    tuple : (pr_auc, ap_score)
    """
    # Compute Precision, Recall, and Thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recalls, precisions)
    ap_score = average_precision_score(y_true, y_probs)
    baseline = np.sum(y_true) / len(y_true)

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- Plot 1: Precision-Recall Curve ---
    axes[0].plot(recalls, precisions, color='b', lw=2, label=f'PR Curve (AUC = {pr_auc:.2f})')
    axes[0].plot([0, 1], [baseline, baseline], color='r', linestyle='--', label=f'Baseline ({baseline:.2f})')
    axes[0].set_xlabel('Recall (True Positive Rate)')
    axes[0].set_ylabel('Precision (Positive Predictive Value)')
    axes[0].set_title('Precision-Recall Curve')
    axes[0].legend(loc='lower left')
    axes[0].grid(True)

    # --- Plot 2: Precision & Recall vs. Threshold ---
    axes[1].plot(thresholds, precisions[:-1], 'b--', label='Precision')
    axes[1].plot(thresholds, recalls[:-1], 'g-', label='Recall')
    
    if best_threshold is not None:
        axes[1].axvline(
            best_threshold, 
            color='r', 
            linestyle=':', 
            label=f'Selected Threshold ({best_threshold:.2f})'
        )
        
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'Threshold Optimization ({metric_name.upper()})')
    axes[1].legend(loc='lower left')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return pr_auc, ap_score