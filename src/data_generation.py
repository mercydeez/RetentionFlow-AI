from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"


def generate_synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 600

    purchases_last_month = rng.poisson(lam=4.5, size=n_rows)
    purchases_last_month = np.clip(purchases_last_month, 0, 18)

    days_since_last_login = rng.gamma(shape=2.0, scale=8.0, size=n_rows)
    days_since_last_login = np.clip(days_since_last_login, 0, 90)

    avg_spend = rng.normal(loc=72.0, scale=28.0, size=n_rows)
    avg_spend = np.clip(avg_spend, 8.0, 260.0)

    complaints_count = rng.poisson(lam=1.1, size=n_rows)
    complaints_count = np.clip(complaints_count, 0, 8)

    logit = (
        1.2
        - 0.30 * purchases_last_month
        + 0.05 * days_since_last_login
        - 0.012 * avg_spend
        + 0.42 * complaints_count
        + rng.normal(0.0, 0.9, size=n_rows)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, churn_prob, size=n_rows)

    # Flip a small portion of labels to keep the task realistic and non-separable.
    noise_mask = rng.random(n_rows) < 0.06
    churn = np.where(noise_mask, 1 - churn, churn)

    df = pd.DataFrame(
        {
            "purchases_last_month": purchases_last_month.astype(int),
            "days_since_last_login": np.round(days_since_last_login, 1),
            "avg_spend": np.round(avg_spend, 2),
            "complaints_count": complaints_count.astype(int),
            "churn": churn.astype(int),
        }
    )
    return df


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved dataset to: {DATA_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
