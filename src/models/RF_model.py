import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, balanced_accuracy_score

class RandomForest_model(BaseEstimator, ClassifierMixin):
    def __init__(self, metric='f1', n_estimators=100, max_depth=None, min_samples_split=2, 
                 class_weights={0: 1, 1: 3}, n_jobs=-1, random_state=None):
        self.metric = metric
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weights = class_weights
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Internal Random Forest instance
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weights,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.best_threshold_ = 0.5

    def fit(self, X, y):
        # 1. Fit the underlying Random Forest model
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        
        # 2. Get predicted probabilities for the positive class
        y_probs = self.model.predict_proba(X)[:, 1]
        
        # 3. Find the best threshold according to the requested metric
        self.best_threshold_ = self._find_best_threshold(y, y_probs)
        return self

    def _find_best_threshold(self, y_true, y_probs):
        if self.metric == 'f1':
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
            # Prevent division by zero
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores)
            return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        elif self.metric == 'youden':
            # Maximizes (True Positive Rate - False Positive Rate)
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            return thresholds[best_idx]

        elif self.metric == 'balanced_accuracy':
            thresholds = np.linspace(0.01, 0.99, 100)
            scores = [balanced_accuracy_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]
            best_idx = np.argmax(scores)
            return thresholds[best_idx]

        else:
            raise ValueError(f"Unknown metric '{self.metric}'. Choose 'f1', 'youden', or 'balanced_accuracy'.")

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        # Predicts using the optimal threshold discovered during .fit()
        y_probs = self.predict_proba(X)[:, 1]
        return (y_probs >= self.best_threshold_).astype(int)

    def plot_threshold_curve(self, X, y):
        """Visualizes Precision and Recall across thresholds alongside the selected best threshold."""
        y_probs = self.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_probs)
        
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.axvline(self.best_threshold_, color="r", linestyle=":", label=f"Selected Threshold ({self.best_threshold_:.2f})")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title(f"Random Forest Threshold Optimization ({self.metric.upper()})")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    import sys
    import os
    from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score

    # Gets the parent directory of your current notebook folder
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

    # Adds it to the search path if it's not already there
    if parent_dir not in sys.path:
        sys.path.append(parent_dir) 

    from Enum.PathEnum import PathEnum
    from data.data_helper import load_data
    from Preprocessing.preprocess import Preprocessing

    preprocessor = Preprocessing(target_col='Class', factor=1.5)
        
    # 1. Read Data
    train_df = load_data(PathEnum.TRAIN_PATH.value)
    val_df = load_data(PathEnum.VAL_PATH.value)
    test_df = load_data(PathEnum.TEST_PATH.value)

    # 2. Process Train set (Fit + Transform)
    X_train, y_train, clip_bounds, scaler = preprocessor.fit_transform(train_df)

    # 3. Process Validation set (Transform ONLY)
    X_val, y_val = preprocessor.transform(val_df, clip_bounds, scaler)

    # 4. Instantiate and fit custom Random Forest model
    clf = RandomForest_model(
        metric='f1', 
        n_estimators=100, 
        max_depth=10, 
        class_weights={0: 1, 1: 3}, 
        random_state=42
    )
    clf.fit(X_train, y_train)

    print(f"Optimal Threshold Selected: {clf.best_threshold_:.4f}\n")

    # Predict using the automatically calculated optimal threshold
    y_pred = clf.predict(X_val)
    print("--- Classification Report (Validation Set) ---")
    print(classification_report(y_val, y_pred))

    # Plot the threshold curve
    clf.plot_threshold_curve(X_val, y_val)

    # Get predicted probabilities for the positive class
    y_probs = clf.predict_proba(X_val)[:, 1]

    # Compute Precision, Recall, and Thresholds
    precision, recall, thresholds = precision_recall_curve(y_val, y_probs)

    # Calculate Area Under the PR Curve (PR-AUC)
    ap_score = average_precision_score(y_val, y_probs)

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label=f'Random Forest PR Curve (AP = {ap_score:.2f})', color='purple')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall Curve (Validation Set)')
    plt.legend()
    plt.grid(True)
    plt.show() 
    print(clf.best_threshold_)