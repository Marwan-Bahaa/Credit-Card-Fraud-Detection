import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, balanced_accuracy_score

class RandomForest_model(BaseEstimator, ClassifierMixin):
    def __init__(self, metric='f1', n_estimators=100, max_depth=9, min_samples_split=5, 
                 class_weight={0: 1, 1: 3}, n_jobs=-1, random_state=42, min_samples_leaf=2):
        self.metric = metric
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weights = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state 
        self.min_samples_leaf=min_samples_leaf
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weights,
            n_jobs=self.n_jobs,
            random_state=self.random_state, 
            min_samples_leaf=self.min_samples_leaf, 
            max_features='sqrt'
        )
        self.best_threshold_ = 0.5

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_ 

        y_probs = self.model.predict_proba(X)[:, 1] 

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
        y_probs = self.predict_proba(X)[:, 1]
        return (y_probs >= self.best_threshold_).astype(int)




if __name__ == '__main__':
    import os
    import sys
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import GridSearchCV


    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from Enum.PathEnum import PathEnum
    from data.data_helper import load_data
    from Preprocessing.preprocess import Preprocessing 
    from testing.evaluate import plot_pr_and_threshold_curves, cls_report
    from sklearn.metrics import classification_report 

    preprocessor = Preprocessing(target_col='Class', factor=1.5)

    train_df = load_data(PathEnum.TRAIN_PATH.value)
    val_df = load_data(PathEnum.VAL_PATH.value)

    # Prevent data leakage: fit_transform ONLY on train set
    X_train, y_train, clip_bounds, scaler = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df, clip_bounds, scaler)

    model = RandomForest_model() 

    model.fit(X_train, y_train)  
    y_pred = model.predict(X_val)
    y_prop = model.predict_proba(X_val)[:,1]

    print(classification_report(y_val, y_pred))  
    plot_pr_and_threshold_curves(y_val, y_prop, model.best_threshold_)  
