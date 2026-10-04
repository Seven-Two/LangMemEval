"""Offline HTTP attempts, payload privacy and provider usage diagnostics."""
import json
from types import SimpleNamespace as NS

import httpx2 as httpx
import openai
import pytest

from agents_memory import diagnostics as log, usage as accounting
from langmem_eval.llm_diagnostics import DiagnosticHttpClient, chat_completion


@pytest.fixture
def captured(tmp_path, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic tests must never use network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    records = []
    original = log._emit

    def emit(status, fields, **kwargs):
        records.append({"status": status, **fields})
        original(status, fields, **kwargs)

    monkeypatch.setattr(log, "_emit", emit)
    with log.run_logging(tmp_path / "run.log"):
        yield records


def completion(message=None, usage=None):
    body = {"id": "response-1", "object": "chat.completion", "created": 0,
            "model": "offline", "choices": [{"index": 0, "finish_reason": "stop",
            "message": message or {"role": "assistant", "content": "OK"}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body, headers={"x-request-id": "provider-1"})


def client(handler, retries=2):
    return openai.OpenAI(api_key="private-key-marker", base_url="https://offline.invalid/v1",
        max_retries=retries, http_client=DiagnosticHttpClient(
            transport=httpx.MockTransport(handler), trust_env=False))


def call(obj):
    return chat_completion(obj, operation="amem.structured", model="offline",
        messages=[{"role": "user", "content": "private-prompt-marker"}], max_tokens=1000,
        extra_body={"chat_template_kwargs": {"enable_thinking": False},
                    "private-option": "private-option-marker"})


def test_retry_success_links_lengths_usage_and_http_attempts(captured, tmp_path):
    visible, reasoning = "private-answer-marker 上海", "private-reasoning-marker" * 10
    provider_usage = {"prompt_tokens": 100, "completion_tokens": 3000, "total_tokens": 3100,
        "completion_tokens_details": {"reasoning_tokens": 2990},
        "prompt_tokens_details": {"cached_tokens": 90}, "provider_counter": 17}
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"message": "busy"}},
                                  headers={"retry-after": "0.001"})
        return completion({"role": "assistant", "content": visible,
                           "reasoning_content": reasoning}, provider_usage)

    accounting.start()
    accounting.reset()
    try:
        with client(handler) as obj, accounting.phase("write"), log.stage("amem.analyze", message_index=1):
            response = call(obj)
        assert response.choices[0].message.content == visible
        # HTTP retries must not become additional logical usage/cost events.
        assert accounting.get_report()["method"]["calls"] == 1
    finally:
        accounting.reset()
    events = [r for r in captured if r["status"].startswith("llm.")]
    assert len({r["request_id"] for r in events}) == 1
    assert all(r["parent_stage"] == "amem.analyze" and r["message_index"] == 1 for r in events)
    result = next(r for r in events if r["status"] == "llm.response")
    assert result["visible_output_chars"] == len(visible)
    assert result["reasoning_chars"] == len(reasoning)
    assert result["usage"]["provider_counter"] == 17
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2990
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 90
    assert result["http_attempts"] == 2 and result["retry_count"] == 1
    assert result["provider_request_id"] == "provider-1"
    retry = next(r for r in events if r["status"] == "llm.retry")
    assert retry["previous_status_code"] == 429
    assert retry["gap_since_previous_attempt_s"] >= 0
    request_event = next(r for r in events if r["status"] == "llm.request")
    assert request_event["thinking_options"] == {"chat_template_kwargs.enable_thinking": False}
    assert requests[0] == requests[1]
    text = (tmp_path / "run.log").read_text(encoding="utf-8")
    for marker in ("private-answer-marker", "private-reasoning-marker", "private-prompt-marker",
                   "private-key-marker", "private-option-marker"):
        assert marker not in text


def test_timeout_then_success_and_request_context_reset(captured):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("offline timeout", request=request)
        return completion()

    with client(handler) as obj:
        call(obj)
        call(obj)
    retry = next(r for r in captured if r["status"] == "llm.retry")
    assert retry["previous_error_type"] == "ReadTimeout"
    responses = [r for r in captured if r["status"] == "llm.response"]
    assert [r["retry_count"] for r in responses] == [1, 0]
    assert responses[0]["request_id"] != responses[1]["request_id"]
    assert responses[0]["reasoning_chars"] is None
    assert responses[0]["usage"] is None


@pytest.mark.parametrize("failure", ["timeout", "http500", "http400"])
def test_exhausted_or_nonretryable_request_records_error_and_attempt_count(captured, failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ConnectTimeout("offline timeout", request=request)
        return httpx.Response(500 if failure == "http500" else 400,
                              json={"error": {"message": "offline failure"}})

    expected = openai.APITimeoutError if failure == "timeout" else openai.APIStatusError
    with client(handler, retries=1) as obj, pytest.raises(expected):
        call(obj)
    result = next(r for r in captured if r["status"] == "llm.error")
    count = 1 if failure == "http400" else 2
    assert result["http_attempts"] == count and result["retry_count"] == count - 1
    assert not any(r["status"] == "llm.response" for r in captured)
    assert len([r for r in captured if r["status"] == "llm.http.start"]) == count


@pytest.mark.parametrize("fields, expected", [
    ({}, None), ({"reasoning_content": ""}, 0),
    ({"reasoning": "think"}, 5),
    ({"reasoning_details": [{"type": "reasoning.text", "text": "think"}]}, 5),
    ({"reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}]}, None),
    ({"reasoning_content": "think", "reasoning": "think"}, 5),
    ({"reasoning_content": "", "reasoning": "think"}, 5),
])
def test_reasoning_missing_empty_alias_and_opaque_are_distinct(captured, fields, expected):
    with client(lambda request: completion({"role": "assistant", "content": "OK", **fields})) as obj:
        call(obj)
    result = next(r for r in captured if r["status"] == "llm.response")
    assert result["visible_output_chars"] == 2
    assert result["reasoning_chars"] == expected


def test_injected_client_marks_retry_counts_unknown(captured):
    obj = NS(chat=NS(completions=NS(create=lambda **kwargs: NS(
        choices=[NS(message=NS(content="OK"), finish_reason="stop")]))))
    call(obj)
    result = next(r for r in captured if r["status"] == "llm.response")
    assert result["retry_count"] is None and result["http_attempts"] is None
    assert result["retry_observation"] == "unavailable"
