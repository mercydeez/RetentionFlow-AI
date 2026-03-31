import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
FEATURE_COLUMNS = [
    "purchases_last_month",
    "days_since_last_login",
    "avg_spend",
    "complaints_count",
]


def categorize_churn_segment(churn_probability: float) -> str:
    if churn_probability < 0.4:
        return "low"
    if churn_probability < 0.7:
        return "medium"
    return "high"


def load_model(model_path: Path = MODEL_PATH) -> Any:
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Model file missing or empty: {model_path}. Train the model first."
        )
    return joblib.load(model_path)


def predict_churn(customer_dict: dict[str, Any]) -> tuple[float, str]:
    missing = [key for key in FEATURE_COLUMNS if key not in customer_dict]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    model = load_model()
    input_df = pd.DataFrame([{key: customer_dict[key] for key in FEATURE_COLUMNS}])

    churn_probability = float(model.predict_proba(input_df)[0, 1])
    churn_segment = categorize_churn_segment(churn_probability)

    return churn_probability, churn_segment


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict customer churn.")
    parser.add_argument(
        "--input-json",
        type=str,
        default="",
        help="JSON string with customer fields.",
    )
    args = parser.parse_args()

    if args.input_json:
        payload = json.loads(args.input_json)
    else:
        payload = {
            "purchases_last_month": 2,
            "days_since_last_login": 40,
            "avg_spend": 31.4,
            "complaints_count": 3,
        }

    probability, churn_segment = predict_churn(payload)
    print("Input:", payload)
    print(f"Churn Probability: {probability:.4f}")
    print("Churn Segment:", churn_segment)


if __name__ == "__main__":
    main()
