import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, balanced_accuracy_score

class LogisticRegression_model(BaseEstimator, ClassifierMixin):
    def __init__(self, metric='f1', C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=None, class_wights={0:1, 1:3}):
        self.metric = metric
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state 
        self.class_wights = class_wights
        
        # Internal model instance
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_wights
        )
        self.best_threshold_ = 0.5

    def fit(self, X, y):
        # 1. Fit the underlying logistic regression model
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
        plt.title(f"Threshold Optimization ({self.metric.upper()})")
        plt.legend()
        plt.grid(True)
        plt.show()
  

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import sys
    import os

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

    # 3. Process Train set (Fit + Transform)
    X_train, y_train, clip_bounds, scaler = preprocessor.fit_transform(train_df)

    # 4. Process Validation and Test sets (Transform ONLY)
    X_test, y_test = preprocessor.transform(val_df, clip_bounds, scaler)


    # Instantiate and fit custom model
    clf = LogisticRegression_model(metric='f1', random_state=42)
    clf.fit(X_train, y_train)

    print(f"Optimal Threshold Selected: {clf.best_threshold_:.4f}\n")

    # Predict using the automatically calculated optimal threshold
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Plot the threshold curve
    clf.plot_threshold_curve(X_test, y_test)

    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    # 1. Get predicted probabilities for the positive class
    y_probs = clf.model.predict_proba(X_test)[:, 1]

    # 2. Compute Precision, Recall, and Thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

    # 3. Calculate Area Under the PR Curve (PR-AUC)
    ap_score = average_precision_score(y_test, y_probs)

    # 4. Plot
    plt.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.2f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
