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
    import os
    import sys
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        precision_recall_curve,
    )
    from sklearn.model_selection import GridSearchCV

    # 1. Setup path imports
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from Enum.PathEnum import PathEnum
    from data.data_helper import load_data
    from Preprocessing.preprocess import Preprocessing

    # 2. Load and Preprocess Data
    preprocessor = Preprocessing(target_col='Class', factor=1.5)

    train_df = load_data(PathEnum.TRAIN_PATH.value)
    val_df = load_data(PathEnum.VAL_PATH.value)
    test_df = load_data(PathEnum.TEST_PATH.value)

    # Prevent data leakage: fit_transform ONLY on train set
    X_train, y_train, clip_bounds, scaler = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df, clip_bounds, scaler)
    X_test, y_test = preprocessor.transform(test_df, clip_bounds, scaler)

    # 3. Define Hyperparameter Grid Search
    print("--- Running Grid Search for Hyperparameters ---")
    # param_grid = {
    #     'n_estimators': [50],  # [5, 10, 15, ..., 50]
    #     'max_depth': [9],         # [3, 4, 5, ..., 10]
    # }

    # Instantiate base estimator for grid search
    # (Extract base model if Random Forest_model wraps RandomForestClassifier)
    from sklearn.ensemble import RandomForestClassifier
    # base_rf = RandomForestClassifier(
    #     n_estimators=50,
    #     max_depth=9,
    #     class_weight={0: 1, 1: 3},
    #     random_state=42,
    #     n_jobs=-1
    # )

    # grid_search = GridSearchCV(
    #     estimator=base_rf,
    #     param_grid=param_grid,
    #     scoring='average_precision',  # PR-AUC optimal for imbalanced fraud data
    #     cv=5,
    #     n_jobs=-1,
    #     verbose=1
    # )
    # grid_search.fit(X_train, y_train)

    # best_n_estimators = grid_search.best_params_['n_estimators']
    # best_max_depth = grid_search.best_params_['max_depth']

    # print(f"\nBest Parameters Found -> n_estimators: {best_n_estimators}, max_depth: {best_max_depth}")
    # print(f"Best CV PR-AUC Score: {grid_search.best_score_:.4f}\n")

    # 4. Instantiate and fit custom Random Forest model with tuned parameters
    clf = RandomForest_model(
        metric='f1',
        n_estimators=50,
        max_depth=9,
        class_weights={0: 1, 1: 3},
        random_state=42
    )
    clf.fit(X_train, y_train)

    print(f"Optimal Threshold Selected: {clf.best_threshold_:.4f}\n")

    # 5. Evaluate Predictions using Optimal Threshold
    print("--- Classification Report (Train Set) ---")
    y_pred_train = clf.predict(X_train)
    print(classification_report(y_train, y_pred_train))

    print("--- Classification Report (Validation Set) ---")
    y_pred_val = clf.predict(X_val)
    print(classification_report(y_val, y_pred_val))

    print("--- Classification Report (Test Set) ---")
    y_pred_test = clf.predict(X_test)
    print(classification_report(y_test, y_pred_test))

    # # 6. Visual Diagnostics
    # if hasattr(clf, 'plot_threshold_curve'):
    #     clf.plot_threshold_curve(X_val, y_val)

    # # Compute PR-AUC on Validation Set
    # y_probs = clf.predict_proba(X_val)[:, 1]
    # precision, recall, thresholds = precision_recall_curve(y_val, y_probs)
    # ap_score = average_precision_score(y_val, y_probs)

    # plt.figure(figsize=(8, 5))
    # plt.plot(recall, precision, label=f'Random Forest (AP = {ap_score:.4f})', color='purple', lw=2)
    
    # # Highlight optimal operating threshold on PR curve
    # idx = np.argmin(np.abs(thresholds - clf.best_threshold_))
    # if idx < len(recalls):
    #     plt.scatter(
    #         recall[idx], precision[idx], 
    #         color='red', s=80, zorder=5, 
    #         label=f'Threshold = {clf.best_threshold_:.4f}'
    #     )

    # plt.xlabel('Recall (Sensitivity)')
    # plt.ylabel('Precision (Positive Predictive Value)')
    # plt.title(f'Precision-Recall Curve (Validation Set) | depth={best_max_depth}, n_est={best_n_estimators}')
    # plt.legend(loc='lower left')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    print(f"Final Optimal Threshold Saved: {clf.best_threshold_:.4f}")