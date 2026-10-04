"""Metadata-only diagnostics for synchronous, non-streaming chat requests.

Keep SDK retry policy intact. Observe each SDK HTTP send, including connection
failures, without enabling HTTP/SDK debug logs that could expose request bodies.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from openai import DefaultHttpxClient

from agents_memory.diagnostics import event, stage


@dataclass
class _Request:
    attempts: int = 0
    last_finished: float | None = None
    last_status: int | None = None
    last_error: str | None = None


_request = ContextVar("chat_diagnostic_request", default=None)


class DiagnosticHttpClient(DefaultHttpxClient):
    """Retain OpenAI's HTTP defaults while observing actual SDK attempts."""

    def __del__(self):
        # Match the SDK's default owned-client cleanup for persistent backends.
        try:
            if not self.is_closed:
                self.close()
        except Exception:
            pass

    def send(self, request, **kwargs):
        trace = _request.get()
        if trace is None:
            return super().send(request, **kwargs)
        started = perf_counter()
        trace.attempts += 1
        attempt = trace.attempts
        if attempt > 1:
            event("llm.retry", attempt=attempt, previous_status_code=trace.last_status,
                  previous_error_type=trace.last_error,
                  gap_since_previous_attempt_s=round(started - trace.last_finished, 3))
        event("llm.http.start", attempt=attempt)
        try:
            response = super().send(request, **kwargs)
        except Exception as exc:
            trace.last_status, trace.last_error = None, type(exc).__name__
            event("llm.http.error", attempt=attempt, error_type=trace.last_error,
                  elapsed_s=round(perf_counter() - started, 3))
            raise
        else:
            trace.last_status, trace.last_error = response.status_code, None
            event("llm.http.response", attempt=attempt, status_code=response.status_code,
                  elapsed_s=round(perf_counter() - started, 3),
                  provider_request_id=response.headers.get("x-request-id"),
                  retry_after=response.headers.get("retry-after"),
                  retry_after_ms=response.headers.get("retry-after-ms"))
            return response
        finally:
            trace.last_finished = perf_counter()


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _usage_dict(usage):
    # model_dump preserves provider extensions and nested reasoning/cache counts.
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return {key: _usage_dict(value) for key, value in usage.items()}
    if isinstance(usage, (list, tuple)):
        return [_usage_dict(value) for value in usage]
    if isinstance(usage, (str, int, float, bool)):
        return usage
    if hasattr(usage, "__dict__"):
        return _usage_dict(vars(usage))
    return None


def _text_length(value):
    """Unicode character count; unknown/opaque data is not zero text."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        lengths = [_text_length(item) for item in value]
        return sum(lengths) if all(n is not None for n in lengths) else None
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning"):
            if key in value:
                return _text_length(value[key])
    return None


def _choice_metadata(choice):
    message = _field(choice, "message")
    # Do not sum aliases: providers may return the same reasoning in two fields.
    reasoning = {name: _text_length(_field(message, name))
                 for name in ("reasoning_content", "reasoning", "reasoning_details")}
    selected = next((name for name, length in reasoning.items() if length), None)
    if selected is None:
        selected = next((name for name, length in reasoning.items() if length is not None), None)
    return {
        "index": _field(choice, "index", 0),
        "finish_reason": _field(choice, "finish_reason"),
        "visible_output_chars": _text_length(_field(message, "content")),
        "reasoning_chars": reasoning[selected] if selected else None,
        "reasoning_field": selected,
        "reasoning_chars_by_field": reasoning,
    }


def _thinking_options(extra):
    result = {}
    if not isinstance(extra, dict):
        return result
    if type(extra.get("enable_thinking")) is bool:
        result["enable_thinking"] = extra["enable_thinking"]
    template = extra.get("chat_template_kwargs")
    if isinstance(template, dict) and type(template.get("enable_thinking")) is bool:
        result["chat_template_kwargs.enable_thinking"] = template["enable_thinking"]
    return result


def chat_completion(client, *, operation, diagnostic_metadata=None, **kwargs):
    """Return the original response; trace lengths before callers parse/strip it.

    Injected clients without DiagnosticHttpClient still get output diagnostics,
    but their HTTP attempt/retry counts are explicitly unknown.
    Optional diagnostic_metadata receives correlation IDs and finish_reason for
    downstream parsing logs; it is never forwarded to the provider.
    """
    if kwargs.get("stream"):
        raise ValueError("chat_completion diagnostics require non-streaming responses")
    trace = _Request()
    request_id = uuid4().hex
    if diagnostic_metadata is not None:
        diagnostic_metadata.clear()
        diagnostic_metadata["request_id"] = request_id
    observed = isinstance(getattr(client, "_client", None), DiagnosticHttpClient)
    token = _request.set(trace)
    started = perf_counter()

    def counts():
        return {"http_attempts": trace.attempts if observed else None,
                "retry_count": max(0, trace.attempts - 1) if observed else None,
                "retry_observation": "sdk_http_send" if observed else "unavailable"}

    try:
        with stage("llm.api", request_id=request_id, operation=operation, model=kwargs.get("model")):
            event("llm.request", max_output_tokens=kwargs.get("max_completion_tokens", kwargs.get("max_tokens")),
                  response_format=_field(kwargs.get("response_format"), "type"),
                  thinking_options=_thinking_options(kwargs.get("extra_body")),
                  max_retries=getattr(client, "max_retries", None))
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                event("llm.error", error_type=type(exc).__name__,
                      cause_type=type(exc.__cause__).__name__ if exc.__cause__ else None,
                      status_code=getattr(exc, "status_code", None),
                      elapsed_s=round(perf_counter() - started, 3), **counts())
                raise
            choices = [_choice_metadata(choice) for choice in _field(response, "choices", [])]
            if diagnostic_metadata is not None:
                diagnostic_metadata.update(response_id=_field(response, "id"),
                    finish_reason=choices[0]["finish_reason"] if choices else None)
            usage = _usage_dict(_field(response, "usage"))
            event("llm.response", elapsed_s=round(perf_counter() - started, 3),
                  provider_request_id=_field(response, "_request_id"),
                  response_id=_field(response, "id"), response_model=_field(response, "model"),
                  usage=usage, prompt_tokens=_field(usage, "prompt_tokens"),
                  completion_tokens=_field(usage, "completion_tokens"),
                  choices=choices, **(choices[0] if choices else {}), **counts())
            return response
    finally:
        _request.reset(token)
