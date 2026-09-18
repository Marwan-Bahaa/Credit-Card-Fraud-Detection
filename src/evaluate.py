import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns 
from numpy import argmax



def eval_classification_report_confusion_matrix(y_pred, y_true, title="" ,save_png=False, path="", digits=5 ):

    print(f'{title} Classification Report')
    print(classification_report(y_pred=y_pred, y_true=y_true, digits=digits))   
    report_stats = classification_report(y_pred=y_pred, y_true=y_true, digits=digits, output_dict=True)

    labels = ['True Negative', 'False Positive' , 'False Negative', 'True Positive'] # order of confusion matrix labels
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_flat = cm.flatten()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    for i, txt in enumerate(cm_flat):
        plt.text(i % 2 + 0.5, i // 2 + 0.5, f"{labels[i]}\n{txt}", ha='center', va='center', color='black')
    plt.title(f'Confusion Matrix of {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    if save_png: 
        plt.savefig(f'{path}/{title} Confusion Matrix.png')
    else: 
        plt.show()

    return report_stats




def eval_best_threshold(y_pred,y_true , with_repect_to="f1_score"): 
    """
    Get best threshold from precision recall curve with respect to f1_score, precision or recall

    parameters:
    y_pred: predicted values
    y_true: true values
    with_repect_to: "f1_score" , "precision" or "recall"

    returns:
    optimal threshold and f1 scores
    """
    precision, recall, thresholds = precision_recall_curve(y_score=y_pred,y_true=y_true)
    f1_scores = ((2 * precision * recall) / (precision + recall))

    if with_repect_to == "f1_score":
        optimal_threshold_index = argmax(f1_scores)
    elif with_repect_to == "precision":
        optimal_threshold_index = argmax(precision)
    elif with_repect_to == "recall":
        optimal_threshold_index = argmax(recall)
    else:
        raise ValueError("Invalid value for with_repect_to. Please choose 'f1_score', 'precision' or 'recall'.")        

    optimal_threshold = thresholds[optimal_threshold_index]
    print("Optimal Threshold:", optimal_threshold , "F1 Score:", f1_scores[optimal_threshold_index])
    return optimal_threshold , f1_scores



def eval_auc_precision_recall_curve(y_pred_prob, y_true):
     """
     Get Area under curve of precision recal of precision recall curve

     Uasge:
     Auc of precision recall curve give good indicator of over all model peformance.
     """
     precision, recall, _ = precision_recall_curve(y_score=y_pred_prob,y_true=y_true)
     
     return float(auc(x=recall, y=precision))


def plot_pr_and_threshold_curves(y_true, y_probs, best_threshold=None, metric_name="AP", model_name='None'):
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
    # Replace plt.show() on line 94 with:
    plt.savefig(f"models_figures/{model_name}_evaluation_plot.png", bbox_inches="tight")
    plt.close()  # Free up memory

    return pr_auc, ap_score