"""Offline regression coverage for the pinned upstream JSON fallback."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from agents_memory import diagnostics as log
from langmem_eval.methods.amem import prompts
from langmem_eval.methods.amem.backend import AMemBackend
from langmem_eval.methods.amem.clients import InvalidJSONResponse, OpenAIController
from langmem_eval.methods.amem.config import AMemSettings
from test_amem_contract import Encoder, analysis, decision, session


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Fallback tests must never use network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")


def scripted_backend(replies, *, mode="json_schema", threshold=100):
    pending, calls = iter(replies), []

    def create(**kwargs):
        calls.append(kwargs)
        value = next(pending)
        if isinstance(value, Exception):
            raise value
        content, finish = value if isinstance(value, tuple) else (json.dumps(value), "stop")
        return NS(id=f"response-{len(calls)}", choices=[NS(
            message=NS(content=content), finish_reason=finish)])

    settings = AMemSettings(response_format=mode, evolution_threshold=threshold)
    controller = OpenAIController("offline", settings,
        client=NS(chat=NS(completions=NS(create=create))))
    obj = AMemBackend("offline", settings=settings, controller=controller, embedder=Encoder())
    return obj, calls


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "prompt"])
@pytest.mark.parametrize("finish", ["length", "stop"])
def test_invalid_evolution_preserves_notes_and_continues_without_retry(mode, finish, tmp_path):
    obj, calls = scripted_backend([analysis(), decision(), analysis("new context"),
        ('{"should_evolve":true,"actions":["update_neighbor"],"private-output-marker":', finish),
        analysis(), decision(), {"keywords": "Berlin"}], mode=mode)
    obj.ingest(session("Berlin"))
    before, vectors = obj.snapshot(), obj.vectors.copy()
    path = tmp_path / "fallback.log"
    with log.run_logging(path):
        obj.ingest(session("Shanghai", name="session_2"))
    assert obj.snapshot()["notes"][0] == before["notes"][0]
    assert np.array_equal(obj.vectors[:1], vectors)
    assert len(obj.notes) == 2 and obj.vectors.shape == (2, 2)
    assert obj.notes[-1].context == "new context"
    assert obj.notes[-1].tags == ["personal"] and obj.notes[-1].links == []
    assert obj.evolution_count == 0
    assert obj.describe()["evolution_stats"] == {
        "attempts": 2, "truncated": int(finish == "length"), "skipped_invalid_json": 1}
    assert obj.describe()["evolution_failure_policy"] == "upstream_skip_invalid_json"
    text = path.read_text(encoding="utf-8")
    assert 'status="amem.evolution.skipped"' in text and 'reason="invalid_json"' in text
    assert 'response_id="response-4"' in text and 'status="failed"' not in text
    request_id = obj.controller.last_response_metadata["request_id"]
    assert text.count(f'request_id="{request_id}"') >= 3
    assert "private-output-marker" not in text
    obj.ingest(session("Paris", name="session_3"))
    assert obj.retrieve("Where?", 1)
    assert len(obj.notes) == 3 and len(calls) == 7
    assert obj.describe()["evolution_stats"] == {
        "attempts": 3, "truncated": int(finish == "length"), "skipped_invalid_json": 1}
    assert all(call["max_tokens"] == 1000 for call in calls)
    assert all("diagnostic_metadata" not in call for call in calls)


def test_length_with_complete_json_applies_evolution_and_rebuilds(tmp_path):
    evolution = decision(True, actions=["strengthen", "update_neighbor"], suggested_connections=[0],
        tags_to_update=["move"], new_context_neighborhood=["former residence"],
        new_tags_neighborhood=[["past"]])
    obj, calls = scripted_backend([analysis(), decision(), analysis(),
        ("Here is the JSON:\n" + json.dumps(evolution) + "\nDone", "length")], threshold=1)
    with log.run_logging(tmp_path / "valid.log"):
        obj.ingest(session("Berlin", "Shanghai"))
    assert obj.evolution_count == 1 and len(calls) == 4
    assert obj.notes[0].context == "former residence"
    assert obj.notes[1].links == [obj.notes[0].id]
    assert any("former residence" in text for text in obj.embedder.calls[-1])
    assert obj.describe()["evolution_stats"] == {"attempts": 2, "truncated": 1, "skipped_invalid_json": 0}


@pytest.mark.parametrize("bad", [RuntimeError("transport failure"), {"should_evolve": "yes"},
    ('{"should_evolve": true}', "length"),
    decision(True, actions=["strengthen"], suggested_connections=[99]),
    decision(True, actions=["unknown"]), (None, "stop")])
def test_other_evolution_errors_remain_fatal_and_atomic(bad):
    obj, _ = scripted_backend([analysis(), decision(), analysis(), bad])
    obj.ingest(session("Berlin"))
    before, vectors = obj.snapshot(), obj.vectors.copy()
    with pytest.raises((ValueError, RuntimeError)):
        obj.ingest(session("Shanghai"))
    assert obj.snapshot() == before and np.array_equal(obj.vectors, vectors)
    assert obj.evolution_stats["skipped_invalid_json"] == 0


def test_empty_evolution_text_uses_json_fallback():
    obj, calls = scripted_backend([analysis(), ("", "length")], threshold=1)
    obj.ingest(session("Berlin"))
    assert len(obj.notes) == 1 and obj.evolution_count == 0 and len(calls) == 2
    assert obj.evolution_stats["skipped_invalid_json"] == 1
    assert len(obj.embedder.calls) == 1  # normal insertion, no evolution rebuild


def test_embedding_error_after_fallback_does_not_commit_memory():
    obj, _ = scripted_backend([analysis(), decision(), analysis(), ("broken JSON", "length")])
    obj.ingest(session("Berlin"))
    before, vectors = obj.snapshot(), obj.vectors.copy()
    obj._search = lambda *args: [0]
    obj.embedder.encode = lambda texts: np.array([[np.nan, 0.]])
    with pytest.raises(ValueError, match="finite"):
        obj.ingest(session("Shanghai"))
    assert obj.snapshot() == before and np.array_equal(obj.vectors, vectors)


@pytest.mark.parametrize("schema", [prompts.ANALYSIS_SCHEMA, prompts.QUERY_SCHEMA])
def test_analysis_and_query_still_fail_on_invalid_or_truncated_json(schema):
    obj, calls = scripted_backend([("broken", "stop"), ("{}", "length")])
    with pytest.raises(InvalidJSONResponse):
        obj.controller.complete("test", schema)
    with pytest.raises(ValueError, match="truncated"):
        obj.controller.complete("test", schema)
    assert len(calls) == 2


def test_fallback_counts_reach_answer_results(monkeypatch):
    import openai
    from langmem_eval import registry
    from langmem_eval.adapter import run_method

    obj, _ = scripted_backend([analysis(), ("broken", "length"), {"keywords": "Berlin"}])
    monkeypatch.setattr(registry, "create_backend", lambda *args: obj)
    monkeypatch.setenv("LLM_EXTRA_BODY", "{}")

    class AnswerClient:
        def __init__(self, **kwargs):
            self.chat = NS(completions=self)

        def create(self, **kwargs):
            return NS(choices=[NS(message=NS(content="Berlin"), finish_reason="stop")])

    monkeypatch.setattr(openai, "OpenAI", AnswerClient)
    conv = {"conversation": {"session_1_date_time": "2025-01-01", "session_1": [
        {"speaker": "Alice", "text": "Berlin"}]}, "qa": [
        {"question": "Where?", "answer": "Berlin", "category": 1}]}
    row = run_method("amem", conv, "offline", False)[0]
    assert row["answer_status"] == "ok"
    config = row["answer_trace"]["method_config"]
    assert config["implementation"] == "amem_original_json_unified_v3"
    assert config["evolution_stats"] == {"attempts": 1, "truncated": 1, "skipped_invalid_json": 1}
