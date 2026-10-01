"""Small shared options for OpenAI-compatible model endpoints."""
import json
import os
from urllib.parse import urlsplit


def validate_base_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Model base URL must be an HTTP(S) endpoint without embedded credentials")
    if parsed.path.rstrip("/").endswith(("/chat/completions", "/embeddings", "/responses")):
        raise ValueError("Use the API base URL ending in /v1, not /chat/completions or /embeddings")
    if parsed.query or parsed.fragment:
        raise ValueError("Model base URL must not contain query parameters or fragments")
    return value.rstrip("/")


def llm_extra_body() -> dict:
    try:
        value = json.loads(os.getenv("LLM_EXTRA_BODY") or "{}")
    except json.JSONDecodeError:
        raise ValueError("LLM_EXTRA_BODY must be a JSON object") from None
    if not isinstance(value, dict):
        raise ValueError("LLM_EXTRA_BODY must be a JSON object")
    reserved = {"model", "messages", "max_tokens", "temperature", "stream", "response_format",
                "api_key", "base_url", "tools", "tool_choice"}
    if reserved.intersection(value):
        raise ValueError("LLM_EXTRA_BODY must not override protocol, credentials, or tools")
    return value


def answer_extra_options() -> dict:
    validate_base_url(os.getenv("OPENAI_BASE_URL"))
    extra = llm_extra_body()
    return {"extra_body": extra} if extra else {}
