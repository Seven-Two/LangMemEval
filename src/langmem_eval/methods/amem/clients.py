"""A-Mem model boundaries; inject substitutes for offline algorithm tests.

The structured-output policy and embedding defaults belong to A-Mem. Shared
endpoint validation lives in model_api; no model is loaded at import time.
"""
from functools import lru_cache
import json
import os

import numpy as np

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


@lru_cache(maxsize=2)
def _load_local_model(name, revision, device, local_files_only):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Local A-Mem embeddings need: uv sync --locked --extra dev --extra amem") from None
    return SentenceTransformer(name, revision=revision, device=device, local_files_only=local_files_only)


class LocalEmbedder:
    def __init__(self, settings):
        self.settings = settings
        self.model = _load_local_model(settings.embedding_model, settings.embedding_revision,
                                       settings.device, settings.local_files_only)

    def encode(self, texts):
        from agents_memory.usage import record_external_usage
        failed = True
        try:
            result = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            failed = False
            return result
        finally:
            record_external_usage(provider="sentence_transformers_local", model=self.settings.embedding_model,
                                  kind="embedding", failed=failed)


class APIEmbedder:
    def __init__(self, settings: AMemSettings, *, client=None):
        from openai import OpenAI
        self.settings = settings
        if client is None:
            if not settings.embedding_base_url or not settings.embedding_api_key:
                raise ValueError("Set AMEM_EMBEDDING_BASE_URL and AMEM_EMBEDDING_API_KEY for API embeddings")
            client = OpenAI(api_key=settings.embedding_api_key, base_url=settings.embedding_base_url)
        self.client = client

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.settings.embedding_dims or 0))
        vectors = []
        for offset in range(0, len(texts), self.settings.embedding_batch_size):
            chunk = texts[offset:offset + self.settings.embedding_batch_size]
            kwargs = {"dimensions": self.settings.embedding_dims} if self.settings.embedding_dims else {}
            result = self.client.embeddings.create(model=self.settings.embedding_model, input=chunk,
                                                    encoding_format="float", **kwargs)
            ordered = sorted(result.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(chunk))):
                raise ValueError("Embedding service returned missing or duplicate vector indices")
            vectors.extend(item.embedding for item in ordered)
        return np.asarray(vectors, dtype=float)
