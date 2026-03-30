import json
import logging
from pathlib import Path

from generate_content import generate_marketing_message
from predict import MODEL_PATH, predict_churn
from train_model import train_and_save_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"


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
    customer_data: dict[str, float | int],
    churn_probability: float,
    churn_segment: str,
    generated_message: dict[str, str],
) -> None:
    print("=" * 70)
    print(f"Customer {customer_number}")
    print("-" * 70)
    print("Customer Data:")
    for key, value in customer_data.items():
        print(f"  {key}: {value}")

    print("Prediction:")
    print(f"  churn_probability: {churn_probability:.4f}")
    print(f"  churn_segment: {churn_segment}")

    print("Generated Message:")
    print(f"Subject: {generated_message['subject']}")
    print(generated_message["email_body"])
    print()


def run_pipeline() -> None:
    _ensure_model_exists()
    logger = _get_logger()

    print("Running integrated churn-to-marketing pipeline for 3 sample customers...")
    print()
    logger.info("Pipeline run started")

    for index, customer in enumerate(SAMPLE_CUSTOMERS, start=1):
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
            customer_data=customer,
            churn_probability=churn_probability,
            churn_segment=churn_segment,
            generated_message=generated_message,
        )

    logger.info("Pipeline run completed")


if __name__ == "__main__":
    run_pipeline()
