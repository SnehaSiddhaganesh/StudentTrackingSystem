"""
Synthetic Feature Generator & ML Attention Model Trainer.
Trains Random Forest / Gradient Boosting / XGBoost Classifier for Classroom Attention Scoring.

⚠️ TRANSPARENCY & SCIENTIFIC INTEGRITY NOTICE:
--------------------------------------------------
The training dataset in this module is 100% SYNTHETIC and RULE-GENERATED.
The dataset simulates facial landmark metrics (head yaw/pitch, EAR, MAR, gaze)
under controlled statistical distributions.

ACCURACY LIMITATION:
Validation metrics reported below (e.g. ~95% accuracy) indicate the model's
fidelity in capturing the underlying synthetic generator rules. THEY DO NOT
REFLECT REAL-WORLD FIELD ACCURACY on live, diverse human student populations.
Real deployment requires an ethically collected, manually annotated classroom
benchmark dataset with ground-truth attention annotations.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def generate_synthetic_dataset(n_samples=3600):
    """
    Generate synthetic classroom facial geometry feature vectors for training.
    
    Classes:
    0: Attentive (facing forward, eyes open, minimal head tilt)
    1: Distracted (head turned sideways/down, gaze shifted away)
    2: Drowsy (eyes partially/fully closed, low EAR, possible yawn)
    """
    np.random.seed(42)
    samples_per_class = n_samples // 3

    # --- Class 0: Attentive ---
    yaw_0 = np.random.normal(0.0, 7.0, samples_per_class)
    pitch_0 = np.random.normal(-2.0, 5.0, samples_per_class)
    roll_0 = np.random.normal(0.0, 4.0, samples_per_class)
    gaze_x_0 = np.random.normal(0.0, 0.12, samples_per_class)
    gaze_y_0 = np.random.normal(0.0, 0.12, samples_per_class)
    ear_0 = np.random.uniform(0.24, 0.38, samples_per_class)
    mar_0 = np.random.uniform(0.05, 0.25, samples_per_class)
    pose_score_0 = np.random.uniform(0.80, 1.0, samples_per_class)
    gaze_score_0 = np.random.uniform(0.80, 1.0, samples_per_class)
    drowsiness_score_0 = np.random.uniform(0.85, 1.0, samples_per_class)
    y_0 = np.zeros(samples_per_class, dtype=int)

    # --- Class 1: Distracted ---
    yaw_1 = np.random.choice([-1, 1], samples_per_class) * np.random.uniform(25.0, 65.0, samples_per_class)
    pitch_1 = np.random.choice([-1, 1], samples_per_class) * np.random.uniform(18.0, 45.0, samples_per_class)
    roll_1 = np.random.normal(0.0, 12.0, samples_per_class)
    gaze_x_1 = np.random.choice([-1, 1], samples_per_class) * np.random.uniform(0.4, 0.9, samples_per_class)
    gaze_y_1 = np.random.choice([-1, 1], samples_per_class) * np.random.uniform(0.3, 0.8, samples_per_class)
    ear_1 = np.random.uniform(0.18, 0.36, samples_per_class)
    mar_1 = np.random.uniform(0.05, 0.30, samples_per_class)
    pose_score_1 = np.random.uniform(0.1, 0.55, samples_per_class)
    gaze_score_1 = np.random.uniform(0.1, 0.55, samples_per_class)
    drowsiness_score_1 = np.random.uniform(0.70, 0.95, samples_per_class)
    y_1 = np.ones(samples_per_class, dtype=int)

    # --- Class 2: Drowsy ---
    yaw_2 = np.random.normal(0.0, 15.0, samples_per_class)
    pitch_2 = np.random.uniform(15.0, 45.0, samples_per_class)
    roll_2 = np.random.normal(0.0, 10.0, samples_per_class)
    gaze_x_2 = np.random.normal(0.0, 0.3, samples_per_class)
    gaze_y_2 = np.random.uniform(0.3, 0.8, samples_per_class)
    ear_2 = np.random.uniform(0.04, 0.16, samples_per_class)
    mar_2 = np.random.uniform(0.35, 0.85, samples_per_class)
    pose_score_2 = np.random.uniform(0.4, 0.7, samples_per_class)
    gaze_score_2 = np.random.uniform(0.2, 0.6, samples_per_class)
    drowsiness_score_2 = np.random.uniform(0.0, 0.40, samples_per_class)
    y_2 = np.full(samples_per_class, 2, dtype=int)

    X = np.vstack([
        np.column_stack([yaw_0, pitch_0, roll_0, gaze_x_0, gaze_y_0, ear_0, mar_0, pose_score_0, gaze_score_0, drowsiness_score_0]),
        np.column_stack([yaw_1, pitch_1, roll_1, gaze_x_1, gaze_y_1, ear_1, mar_1, pose_score_1, gaze_score_1, drowsiness_score_1]),
        np.column_stack([yaw_2, pitch_2, roll_2, gaze_x_2, gaze_y_2, ear_2, mar_2, pose_score_2, gaze_score_2, drowsiness_score_2])
    ])
    y = np.concatenate([y_0, y_1, y_2])
    return X, y


def train_xgboost_model(output_path=None):
    """
    Trains ML Classifier on synthetic dataset and saves pickle file.
    Evaluates Precision, Recall, F1, and Confusion Matrix.
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, 'models', 'xgboost_attention.pkl')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("[INFO] Generating synthetic classroom feature dataset (Rule-based simulation)...")
    X, y = generate_synthetic_dataset(n_samples=3600)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_names = ['Attentive', 'Distracted', 'Drowsy']

    if HAS_XGBOOST:
        print("[INFO] Training XGBoost Classifier...")
        model = xgb.XGBClassifier(
            n_estimators=120, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'
        )
    else:
        print("[INFO] Training GradientBoosting Classifier fallback...")
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=False)

    print("\n=======================================================")
    print("      SYNTHETIC VALIDATION EVALUATION REPORT")
    print("=======================================================")
    print(f"Synthetic Validation Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in class_names], columns=[f"Pred {c}" for c in class_names])
    print(cm_df)
    print("\nDetailed Precision / Recall / F1-Score Breakdown:")
    print(report)
    print("[LIMITATION NOTE] These metrics only measure fit to the synthetic generator.")
    print("=======================================================\n")

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[INFO] Successfully saved model artifact to: {output_path}")
    return model


if __name__ == '__main__':
    train_xgboost_model()
