"""A-Mem model boundaries; inject substitutes for offline algorithm tests.

The structured-output policy belongs to A-Mem. Embedding clients live in
langmem_eval.embeddings; no model is loaded at import time.
"""
import os
from agents_memory.diagnostics import event
from ...llm_diagnostics import DiagnosticHttpClient, chat_completion
from ...model_api import validate_base_url
from .config import AMemSettings
from .responses import ResponsePurpose, decode_response, prepare_response


class OpenAIController:
    def __init__(self, model: str, settings: AMemSettings, *, client=None):
        from openai import OpenAI
        validate_base_url(os.getenv("OPENAI_BASE_URL"))
        self._owns_client = client is None
        self.client = client if client is not None else OpenAI(http_client=DiagnosticHttpClient())
        self.model, self.settings = model, settings
        self.last_response_metadata = {}

    def complete(self, prompt: str, schema: dict, *, purpose: ResponsePurpose = "generic") -> dict:
        self.last_response_metadata = {}
        mode = self.settings.response_format
        kwargs = {}
        if mode == "json_schema":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "response", "schema": schema, "strict": True}}
        elif mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.extra_body:
            kwargs["extra_body"] = self.settings.extra_body
        response = chat_completion(self.client, operation="amem.structured",
            diagnostic_metadata=self.last_response_metadata,
            model=self.model, messages=[{"role": "system", "content": "You must respond with a JSON object."},
                                        {"role": "user", "content": prompt}],
            temperature=self.settings.temperature, max_tokens=self.settings.max_output_tokens, **kwargs)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            event("amem.response.truncated", **self.last_response_metadata,
                  policy="parse_and_validate",
                  max_output_tokens=self.settings.max_output_tokens)
        parsed = decode_response(choice.message.content, purpose=purpose)
        prepared = prepare_response(parsed, schema, purpose=purpose)
        if prepared.ignored_extra_fields:
            event("amem.response.extra_fields_ignored", count=prepared.ignored_extra_fields,
                  **self.last_response_metadata)
        return prepared.data

    def close(self):
        if self._owns_client:
            self.client.close()
