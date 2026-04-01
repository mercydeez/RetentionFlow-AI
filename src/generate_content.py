import json
import os
import re
from typing import Any

from openai import NotFoundError, OpenAI
from dotenv import load_dotenv


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

load_dotenv()


def _get_client_and_model(model: str | None) -> tuple[OpenAI, str, str]:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if openai_api_key:
        selected_model = model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        return OpenAI(api_key=openai_api_key), selected_model, "openai"

    if groq_api_key:
        selected_model = (
            model or os.getenv("GROQ_MODEL") or os.getenv("MODEL") or DEFAULT_GROQ_MODEL
        )
        return (
            OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1"),
            selected_model,
            "groq",
        )

    raise EnvironmentError(
        "Set OPENAI_API_KEY for OpenAI or GROQ_API_KEY for Groq in your .env file."
    )


def generate_marketing_message(
    customer_data: dict[str, Any],
    churn_risk: str,
    churn_probability: float | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    required_keys = ["avg_spend", "purchases_last_month"]
    missing_keys = [key for key in required_keys if key not in customer_data]
    if missing_keys:
        raise ValueError(f"Missing required customer_data keys: {missing_keys}")

    normalized_risk = churn_risk.strip().lower()
    if normalized_risk not in {"low", "medium", "high"}:
        raise ValueError("churn_risk must be one of: 'low', 'medium', 'high'.")

    avg_spend = float(customer_data["avg_spend"])
    purchases_last_month = int(customer_data["purchases_last_month"])

    if normalized_risk == "high":
        campaign_type = "discount email"
        objective = (
            "retain this customer with a strong limited-time discount and urgency."
        )
    elif normalized_risk == "medium":
        campaign_type = "value reinforcement email"
        objective = (
            "reduce churn risk with a moderate incentive and clear product value reminder."
        )
    else:
        campaign_type = "upsell / premium product email"
        objective = "promote a premium offer that feels valuable and relevant."

    probability_text = ""
    if churn_probability is not None:
        probability_text = f" The churn probability is {churn_probability:.2f}."

    prompt = (
        "Role: You are an e-commerce marketing assistant.\n"
        "Tone: Friendly, persuasive, and not spammy.\n"
        "Constraints:\n"
        "- Maximum 120 words in the email body\n"
        "- Include a clear call-to-action\n"
        "- No exaggerated or unrealistic claims\n"
        "- Return valid JSON only with keys: subject, email_body\n"
        "\n"
        f"Customer context: avg_spend=${avg_spend:.2f}, "
        f"purchases_last_month={purchases_last_month}, risk={normalized_risk}."
        f"{probability_text}\n"
        f"Campaign type: {campaign_type}. Goal: {objective}"
    )

    fallback_templates = {
        "low": {
            "subject": "Upgrade Your Experience with Our Premium Picks",
            "email_body": (
                "Thanks for being a loyal customer. Based on your recent activity, "
                "we think you would love our premium collection with added benefits and value. "
                "Explore the premium options today and discover your next favorite product."
            ),
            "generation_source": "fallback",
        },
        "medium": {
            "subject": "A Special Offer to Keep You Moving Forward",
            "email_body": (
                "We appreciate your recent purchases and want to make your next one even better. "
                "Here is a limited-time value offer designed for you. "
                "Use it now and continue enjoying the products you already trust."
            ),
            "generation_source": "fallback",
        },
        "high": {
            "subject": "Exclusive Limited-Time Discount Just for You",
            "email_body": (
                "We value you and do not want you to miss out. "
                "Enjoy a personalized limited-time discount on your next order. "
                "Claim your offer now and let us make your next purchase even more rewarding."
            ),
            "generation_source": "fallback",
        },
    }

    client, selected_model, provider = _get_client_and_model(model)

    def _create_chat_completion(selected_model_name: str) -> Any:
        request_payload = {
            "model": selected_model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert e-commerce marketing copywriter.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }

        # Prefer structured JSON output when provider/model supports it.
        try:
            return client.chat.completions.create(
                **request_payload,
                response_format={"type": "json_object"},
            )
        except Exception:
            return client.chat.completions.create(**request_payload)

    def _extract_json_text(raw_text: str) -> str:
        text = raw_text.strip()

        # Handle fenced markdown output like ```json ... ```.
        fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1).strip()

        # Handle extra prose around a JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].strip()

        return text

    def _parse_message_json(raw_text: str) -> dict[str, Any]:
        json_text = _extract_json_text(raw_text)
        parsed = json.loads(json_text)
        subject = str(parsed.get("subject", "")).strip()
        email_body = str(parsed.get("email_body", "")).strip()
        if not subject or not email_body:
            raise ValueError("LLM JSON output is missing subject or email_body.")
        return {
            "subject": subject,
            "email_body": email_body,
            "generation_source": "llm",
        }

    try:
        try:
            response = _create_chat_completion(selected_model)
        except NotFoundError:
            if provider == "groq" and model is None and selected_model != DEFAULT_GROQ_MODEL:
                response = _create_chat_completion(DEFAULT_GROQ_MODEL)
            else:
                raise

        generated_text = ""
        if response.choices and response.choices[0].message:
            generated_text = (response.choices[0].message.content or "").strip()

        if not generated_text:
            raise RuntimeError("The language model returned an empty response.")

        return _parse_message_json(generated_text)
    except Exception as exc:
        fallback_message = dict(fallback_templates[normalized_risk])
        fallback_message["fallback_reason"] = str(exc)
        return fallback_message


