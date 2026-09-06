import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score

# 1. Path setup
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 2. Imports from your custom modules
from Enum.PathEnum import PathEnum
from data.data_helper import load_data
from Preprocessing.preprocess import Preprocessing


def train_vc_models():
    # Instantiate preprocessor
    preprocessor = Preprocessing(target_col='Class', factor=1.5)
        
    # Read Data
    train_df = load_data(PathEnum.TRAIN_PATH.value)
    val_df = load_data(PathEnum.VAL_PATH.value)
    test_df = load_data(PathEnum.TEST_PATH.value)

    # Process Data
    # Train: Fit + Transform
    X_train, y_train, clip_bounds, scaler = preprocessor.fit_transform(train_df)
    
    # Val & Test: Transform ONLY
    X_val, y_val = preprocessor.transform(val_df, clip_bounds, scaler)
    X_test, y_test = preprocessor.transform(test_df, clip_bounds, scaler)

    print("--- 1. Training Logistic Regression ---")
    lr_model = LogisticRegression(
        C=1.0,
        class_weight={0: 1, 1: 3},
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    print("--- 2. Training Random Forest ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight={0: 1, 1: 3},
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    print("--- 3. Training Soft Voting Ensemble ---")
    # Soft voting averages probabilities from base models
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('rf', rf_model)
        ],
        voting='soft',
        weights=[1, 2]  # Give Random Forest 2x weight
    )
    ensemble_model.fit(X_train, y_train)

    # Evaluate Ensemble on Validation Set
    y_val_pred = ensemble_model.predict(X_val)
    print("\n--- Validation Classification Report (Ensemble) ---")
    print(classification_report(y_val, y_val_pred))

    # Evaluate Ensemble on Test Set
    y_test_pred = ensemble_model.predict(X_test)
    print("--- Test Classification Report (Ensemble) ---")
    print(classification_report(y_test, y_test_pred))

    # Evaluate PR Curve & AP Score on Test Set
    y_probs = ensemble_model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    ap_score = average_precision_score(y_test, y_probs)

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, color='purple', lw=2, label=f'Voting Ensemble (AP = {ap_score:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save Preprocessor Artifacts & Models
    joblib.dump(ensemble_model,'Ensemble.pkl')
    
    print(f"\nAll models successfully saved")


if __name__ == '__main__':
    train_vc_models()