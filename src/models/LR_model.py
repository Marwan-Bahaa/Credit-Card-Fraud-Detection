import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, balanced_accuracy_score, average_precision_score

class LogisticRegression_model(BaseEstimator, ClassifierMixin):
    def __init__(self, metric='f1', C=1.0, max_iter=1000, random_state=42, class_weight={0:1, 1:3}, penalty='l2', solver='lbfgs', best_threshould=0.5):
        self.metric = metric
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state 
        self.class_wights = class_weight 
        self.penalty = penalty
        self.best_threshold_ = best_threshould
        self.model = LogisticRegression(
            C=self.C, 
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_wights, 
            penalty=self.penalty,             
        )
        

    def fit(self, X_scaled, y):
        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_
    
        y_probs = self.model.predict_proba(X_scaled)[:, 1]
        
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



    def predict_proba(self, X_scaled):
        return self.model.predict_proba(X_scaled)


    def predict(self, X_scaled):
        # Predicts using the optimal threshold discovered during .fit()
        y_probs = self.predict_proba(X_scaled)[:, 1]
        return (y_probs >= self.best_threshold_).astype(int)



  



if __name__ == '__main__':
    import sys
    import os

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir) 


    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from Enum.PathEnum import PathEnum
    from data.data_helper import load_data
    from Preprocessing.preprocess import Preprocessing
    from testing.evaluate import plot_pr_and_threshold_curves

    preprocessor = Preprocessing(target_col='Class', factor=1.5)
        
    train_df = load_data(PathEnum.TRAIN_PATH.value)
    val_df = load_data(PathEnum.VAL_PATH.value)

    X_train, y_train, clip_bounds, scaler = preprocessor.fit_transform(train_df)

    X_test, y_test = preprocessor.transform(val_df, clip_bounds, scaler)

    clf = LogisticRegression_model(metric='f1', random_state=42, max_iter=5000)
    clf.fit(X_train, y_train)

    print(f"Optimal Threshold Selected: {clf.best_threshold_:.4f}\n")

    y_pred = clf.predict(X_test)
    y_prop = clf.predict_proba(X_test)[:, 1]
    print(y_prop.shape)

    print(classification_report(y_test, y_pred)) 
    plot_pr_and_threshold_curves(y_test, y_prop, clf.best_threshold_)
