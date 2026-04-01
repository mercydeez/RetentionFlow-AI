import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from generate_content import generate_marketing_message
from predict import FEATURE_COLUMNS, MODEL_PATH, predict_churn
from train_model import train_and_save_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"
DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "results.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "summary.json"


SAMPLE_CUSTOMERS = [
    {
        "purchases_last_month": 1,
        "days_since_last_login": 44,
        "avg_spend": 29.8,
        "complaints_count": 4,
    },
    {
        "purchases_last_month": 2,
        "days_since_last_login": 10,
        "avg_spend": 45.0,
        "complaints_count": 2,
    },
    {
        "purchases_last_month": 11,
        "days_since_last_login": 3,
        "avg_spend": 141.2,
        "complaints_count": 0,
    },
]


def _ensure_model_exists() -> None:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return

    print("No trained model found. Training model first...")
    metrics = train_and_save_model()
    print(f"Model trained. Accuracy: {metrics['accuracy']:.3f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline")
    if logger.handlers:
        return logger

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def _print_customer_report(
    customer_number: int,
    churn_probability: float,
    churn_segment: str,
    recommended_action: str,
    generated_message: dict[str, Any],
) -> None:
    print(f"Customer {customer_number}:")
    print(f"  churn_probability: {churn_probability:.4f}")
    print(f"  churn_segment: {churn_segment}")
    print(f"  recommended_action: {recommended_action}")
    print(f"  subject: {generated_message['subject']}")
    print(f"  email_body: {generated_message['email_body']}")
    print()


def _normalize_customer(customer_data: dict[str, Any]) -> dict[str, float | int]:
    return {
        "purchases_last_month": int(customer_data["purchases_last_month"]),
        "days_since_last_login": int(customer_data["days_since_last_login"]),
        "avg_spend": float(customer_data["avg_spend"]),
        "complaints_count": int(customer_data["complaints_count"]),
    }


def _load_customers(limit: int | None, logger: logging.Logger) -> tuple[list[dict[str, float | int]], str]:
    def _sample_customers() -> list[dict[str, float | int]]:
        customers = [_normalize_customer(customer) for customer in SAMPLE_CUSTOMERS]
        return customers[:limit] if limit is not None else customers

    if not DATA_PATH.exists():
        logger.warning("Dataset not found at %s. Using sample customers.", DATA_PATH)
        return _sample_customers(), "fallback-samples"

    df = pd.read_csv(DATA_PATH)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        logger.warning(
            "Dataset missing required columns %s. Using sample customers.",
            missing_columns,
        )
        return _sample_customers(), "fallback-samples"

    selected_df = df[FEATURE_COLUMNS]
    if limit is not None:
        selected_df = selected_df.head(limit)

    customers = [
        _normalize_customer(record) for record in selected_df.to_dict(orient="records")
    ]
    return customers, "csv"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be a positive integer.")
    return parsed


def _get_recommended_action(churn_segment: str) -> str:
    mapping = {
        "low": "Upsell premium products",
        "medium": "Send reminder / small discount",
        "high": "Send aggressive retention offer",
    }
    return mapping.get(churn_segment.lower(), "Send reminder / small discount")


def _save_results(results: list[dict[str, Any]], logger: logging.Logger) -> None:
    columns = [
        "purchases_last_month",
        "days_since_last_login",
        "avg_spend",
        "complaints_count",
        "churn_probability",
        "churn_segment",
        "recommended_action",
        "subject",
        "email_body",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved pipeline results to %s", OUTPUT_PATH)
    print(f"Saved results to: {OUTPUT_PATH}")


def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_customers = len(results)
    segment_counts = {"low": 0, "medium": 0, "high": 0}

    for row in results:
        segment = str(row.get("churn_segment", "")).lower()
        if segment in segment_counts:
            segment_counts[segment] += 1

    avg_churn_probability = 0.0
    if total_customers > 0:
        avg_churn_probability = sum(
            float(row["churn_probability"]) for row in results
        ) / total_customers

    return {
        "total_customers_processed": total_customers,
        "low_risk": segment_counts["low"],
        "medium_risk": segment_counts["medium"],
        "high_risk": segment_counts["high"],
        "average_churn_probability": round(avg_churn_probability, 4),
    }


def _save_summary(summary: dict[str, Any], logger: logging.Logger) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved pipeline summary to %s", SUMMARY_PATH)


def _print_summary(summary: dict[str, Any]) -> None:
    print("Summary:")
    print(f"Total Customers: {summary['total_customers_processed']}")
    print(f"Low Risk: {summary['low_risk']}")
    print(f"Medium Risk: {summary['medium_risk']}")
    print(f"High Risk: {summary['high_risk']}")
    print(f"Avg Churn Probability: {summary['average_churn_probability']:.2f}")
    print(f"Saved summary to: {SUMMARY_PATH}")


def run_pipeline(limit: int | None = None) -> None:
    _ensure_model_exists()
    logger = _get_logger()
    customers, source = _load_customers(limit=limit, logger=logger)

    if not customers:
        print("No customers found to process.")
        logger.info("Pipeline run aborted: no customers to process")
        return

    print(
        f"Running integrated churn-to-marketing pipeline for {len(customers)} customers "
        f"(source: {source})..."
    )
    print()
    logger.info(
        "Pipeline run started source=%s total_customers=%s limit=%s",
        source,
        len(customers),
        limit,
    )

    results: list[dict[str, Any]] = []

    for index, customer in enumerate(customers, start=1):
        logger.info(
            "Customer %s input_data=%s",
            index,
            json.dumps(customer, separators=(",", ":")),
        )

        churn_probability, churn_segment = predict_churn(customer)
        logger.info(
            "Customer %s prediction_probability=%.4f churn_segment=%s",
            index,
            churn_probability,
            churn_segment,
        )

        recommended_action = _get_recommended_action(churn_segment)
        logger.info(
            "Customer %s recommended_action=%s",
            index,
            recommended_action,
        )

        generated_message = generate_marketing_message(
            customer, churn_segment, churn_probability
        )

        generation_source = generated_message.get("generation_source", "unknown")
        llm_success = generation_source == "llm"
        fallback_used = generation_source == "fallback"
        logger.info(
            "Customer %s llm_success=%s fallback_used=%s generation_source=%s",
            index,
            llm_success,
            fallback_used,
            generation_source,
        )
        if fallback_used:
            logger.info(
                "Customer %s fallback_reason=%s",
                index,
                generated_message.get("fallback_reason", "n/a"),
            )

        _print_customer_report(
            customer_number=index,
            churn_probability=churn_probability,
            churn_segment=churn_segment,
            recommended_action=recommended_action,
            generated_message=generated_message,
        )

        results.append(
            {
                "purchases_last_month": customer["purchases_last_month"],
                "days_since_last_login": customer["days_since_last_login"],
                "avg_spend": customer["avg_spend"],
                "complaints_count": customer["complaints_count"],
                "churn_probability": round(churn_probability, 4),
                "churn_segment": churn_segment,
                "recommended_action": recommended_action,
                "subject": generated_message["subject"],
                "email_body": generated_message["email_body"],
            }
        )

    _save_results(results=results, logger=logger)
    summary = _compute_summary(results)
    _save_summary(summary=summary, logger=logger)
    _print_summary(summary)
    logger.info("Pipeline run completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end retention pipeline.")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Optional number of customers to process from data/customers.csv",
    )
    args = parser.parse_args()
    run_pipeline(limit=args.limit)
