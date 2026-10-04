"""A-Mem model boundaries; inject substitutes for offline algorithm tests.

The structured-output policy belongs to A-Mem. Embedding clients live in
langmem_eval.embeddings; no model is loaded at import time.
"""
import json
import os

from ...model_api import validate_base_url
from .config import AMemSettings


def _validate(value, schema):
    """Validate the small, fixed upstream JSON schemas even in prompt-only mode."""
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": type(value) is bool,
             "integer": type(value) is int}
    if kind in valid and not valid[kind]:
        raise ValueError(f"A-Mem response must have type {kind}")
    if kind == "object":
        if any(k not in value for k in schema.get("required", [])):
            raise ValueError("A-Mem response is missing required fields")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError("A-Mem response contains unknown fields")
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key])
    elif kind == "array":
        for item in value:
            _validate(item, schema["items"])


class OpenAIController:
    def __init__(self, model: str, settings: AMemSettings, *, client=None):
        from openai import OpenAI
        validate_base_url(os.getenv("OPENAI_BASE_URL"))
        self.client = client if client is not None else OpenAI()
        self.model, self.settings = model, settings

    def complete(self, prompt: str, schema: dict) -> dict:
        mode = self.settings.response_format
        kwargs = {}
        if mode == "json_schema":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "response", "schema": schema, "strict": True}}
        elif mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.extra_body:
            kwargs["extra_body"] = self.settings.extra_body
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": "You must respond with a JSON object."},
                                        {"role": "user", "content": prompt}],
            temperature=self.settings.temperature, max_tokens=self.settings.max_output_tokens, **kwargs)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("A-Mem structured response was truncated; increase AMEM_MAX_OUTPUT_TOKENS")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("A-Mem received an empty structured response")
        content = content.strip()
        if content.startswith("```") and content.endswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("A-Mem received invalid JSON") from None
        _validate(result, schema)
        return result
