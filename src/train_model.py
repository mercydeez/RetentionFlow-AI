from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"


def train_and_save_model() -> dict[str, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    feature_columns = [
        "purchases_last_month",
        "days_since_last_login",
        "avg_spend",
        "complaints_count",
    ]
    required_columns = feature_columns + ["churn"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[feature_columns]
    y = df["churn"]

    model = LogisticRegression(max_iter=1000, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": conf_matrix.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }


def main() -> None:
    metrics = train_and_save_model()
    print("Model Evaluation Results")
    print("------------------------")
    print(f"Accuracy (overall correct predictions): {metrics['accuracy']:.3f}")
    print(f"Precision (quality of positive predictions): {metrics['precision']:.3f}")
    print(f"Recall (coverage of actual churners): {metrics['recall']:.3f}")
    print(f"F1 Score (balance of precision and recall): {metrics['f1_score']:.3f}")
    print(f"ROC-AUC (ranking quality across thresholds): {metrics['roc_auc']:.3f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
