from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"


def generate_synthetic_data() -> pd.DataFrame:
    data = [
        {"purchases_last_month": 1, "days_since_last_login": 45, "avg_spend": 28.5, "complaints_count": 4, "churn": 1},
        {"purchases_last_month": 8, "days_since_last_login": 5, "avg_spend": 92.0, "complaints_count": 0, "churn": 0},
        {"purchases_last_month": 3, "days_since_last_login": 30, "avg_spend": 41.8, "complaints_count": 2, "churn": 1},
        {"purchases_last_month": 11, "days_since_last_login": 2, "avg_spend": 135.4, "complaints_count": 0, "churn": 0},
        {"purchases_last_month": 6, "days_since_last_login": 10, "avg_spend": 76.2, "complaints_count": 1, "churn": 0},
        {"purchases_last_month": 0, "days_since_last_login": 60, "avg_spend": 19.9, "complaints_count": 5, "churn": 1},
        {"purchases_last_month": 2, "days_since_last_login": 37, "avg_spend": 33.7, "complaints_count": 3, "churn": 1},
        {"purchases_last_month": 9, "days_since_last_login": 6, "avg_spend": 101.6, "complaints_count": 0, "churn": 0},
        {"purchases_last_month": 4, "days_since_last_login": 22, "avg_spend": 58.1, "complaints_count": 2, "churn": 0},
        {"purchases_last_month": 1, "days_since_last_login": 50, "avg_spend": 24.4, "complaints_count": 4, "churn": 1},
        {"purchases_last_month": 12, "days_since_last_login": 1, "avg_spend": 168.0, "complaints_count": 0, "churn": 0},
        {"purchases_last_month": 5, "days_since_last_login": 18, "avg_spend": 69.5, "complaints_count": 1, "churn": 0},
        {"purchases_last_month": 2, "days_since_last_login": 33, "avg_spend": 39.2, "complaints_count": 3, "churn": 1},
        {"purchases_last_month": 7, "days_since_last_login": 9, "avg_spend": 84.8, "complaints_count": 1, "churn": 0},
        {"purchases_last_month": 3, "days_since_last_login": 27, "avg_spend": 47.3, "complaints_count": 2, "churn": 1},
        {"purchases_last_month": 10, "days_since_last_login": 4, "avg_spend": 122.9, "complaints_count": 0, "churn": 0},
        {"purchases_last_month": 6, "days_since_last_login": 14, "avg_spend": 79.0, "complaints_count": 1, "churn": 0},
        {"purchases_last_month": 1, "days_since_last_login": 42, "avg_spend": 26.8, "complaints_count": 5, "churn": 1},
        {"purchases_last_month": 8, "days_since_last_login": 7, "avg_spend": 96.7, "complaints_count": 0, "churn": 0},
        {"purchases_last_month": 4, "days_since_last_login": 24, "avg_spend": 55.4, "complaints_count": 2, "churn": 0},
        {"purchases_last_month": 0, "days_since_last_login": 55, "avg_spend": 17.5, "complaints_count": 6, "churn": 1},
        {"purchases_last_month": 9, "days_since_last_login": 3, "avg_spend": 110.1, "complaints_count": 0, "churn": 0},
        {"purchases_last_month": 2, "days_since_last_login": 35, "avg_spend": 36.0, "complaints_count": 3, "churn": 1},
        {"purchases_last_month": 7, "days_since_last_login": 11, "avg_spend": 88.9, "complaints_count": 1, "churn": 0},
        {"purchases_last_month": 5, "days_since_last_login": 16, "avg_spend": 72.4, "complaints_count": 1, "churn": 0},
        {"purchases_last_month": 1, "days_since_last_login": 47, "avg_spend": 22.6, "complaints_count": 4, "churn": 1},
        {"purchases_last_month": 11, "days_since_last_login": 2, "avg_spend": 142.3, "complaints_count": 0, "churn": 0},
        {"purchases_last_month": 3, "days_since_last_login": 29, "avg_spend": 44.7, "complaints_count": 2, "churn": 1},
        {"purchases_last_month": 6, "days_since_last_login": 12, "avg_spend": 81.5, "complaints_count": 1, "churn": 0},
        {"purchases_last_month": 2, "days_since_last_login": 40, "avg_spend": 31.4, "complaints_count": 3, "churn": 1},
    ]

    df = pd.DataFrame(data)
    return df


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved dataset to: {DATA_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
