"""Fixed, offline A-Mem acceptance contract established before implementation.

The score measures integration behavior, not benchmark accuracy. Model outputs
and embeddings are scripted so no credentials, weights or network are needed.
"""
import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from langmem_eval.interfaces import Session


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([[1., 0.] if "Berlin" in t else [0., 1.] for t in texts])


class Controller:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, prompt, schema):
        self.calls.append((prompt, schema))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def analysis(context="residence"):
    return {"keywords": ["residence"], "context": context, "tags": ["personal"]}


def decision(evolve=False, **overrides):
    return dict(should_evolve=evolve, actions=[], suggested_connections=[],
                tags_to_update=[], new_context_neighborhood=[], new_tags_neighborhood=[],
                **{}) | overrides


def session(*texts, date="2025-01-01", name="session_1"):
    return Session(name, date, [{"role": "user", "content": json.dumps(
        {"speaker": "Alice", "text": t, "source_id": f"{name}:{i}", "date": date})}
        for i, t in enumerate(texts)])


def backend(responses, **kwargs):
    from langmem_eval.methods.amem.backend import AMemBackend
    from langmem_eval.methods.amem.config import AMemSettings
    controller, encoder = Controller(responses), Encoder()
    settings = AMemSettings(**kwargs)
    obj = AMemBackend("offline-model", settings=settings, controller=controller, embedder=encoder)
    return obj, controller, encoder


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError("Acceptance tests must never use network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def test_discovery_is_lazy():
    from langmem_eval.registry import discover_methods
    from agents_memory.systems import SYSTEMS
    assert "amem" in discover_methods()
    assert "amem" in SYSTEMS


def test_turn_granularity_dates_duplicate_content_and_no_future_input():
    obj, llm, encoder = backend([analysis(), decision(), analysis(), decision()])
    obj.ingest(session("Berlin", "Berlin"))
    notes = obj.snapshot()["notes"]
    assert len(notes) == 2 and len({n["id"] for n in notes}) == 2
    assert [n["source_id"] for n in notes] == ["session_1:0", "session_1:1"]
    assert all(n["timestamp"] == "2025-01-01" for n in notes)
    assert all("Alice" in n["content"] and "Berlin" in n["content"] for n in notes)
    assert len(llm.calls) == 4
    assert "Berlin" in encoder.calls[0][0]


def test_linking_and_neighbor_evolution_preserve_original_evidence():
    obj, _, _ = backend([analysis(), decision(), analysis(), decision(True,
        actions=["strengthen", "update_neighbor"], suggested_connections=[0],
        tags_to_update=["move"], new_context_neighborhood=["former residence"],
        new_tags_neighborhood=[["past"]])])
    obj.ingest(session("Berlin", "Shanghai"))
    first, second = obj.snapshot()["notes"]
    assert first["context"] == "former residence" and first["tags"] == ["past"]
    assert "Berlin" in first["content"]
    assert second["links"] == [first["id"]] and second["tags"] == ["move"]


def test_index_rebuild_obeys_upstream_evolution_threshold():
    obj, _, enc = backend([analysis(), decision(), analysis(), decision(True,
        actions=["update_neighbor"], new_context_neighborhood=["updated residence"],
        new_tags_neighborhood=[["updated"]])], evolution_threshold=1)
    obj.ingest(session("Berlin", "Shanghai"))
    assert any("updated residence" in text for text in enc.calls[-1])
    assert obj.snapshot()["evolution_count"] == 1


def test_keyword_retrieval_expands_linked_notes_within_complete_groups():
    obj, llm, _ = backend([analysis(), decision(), analysis(), decision(True,
        actions=["strengthen"], suggested_connections=[0], tags_to_update=["move"]),
        {"keywords": "Shanghai"}])
    obj.ingest(session("Berlin", "Shanghai"))
    before = obj.snapshot()
    records = obj.retrieve("Where did Alice move?", 1)
    assert len(records) == 1 and "Shanghai" in records[0] and "Berlin" in records[0]
    assert "Where did Alice move?" in llm.calls[-1][0]
    assert obj.snapshot() == before  # searches must not evolve notes
    assert obj.last_retrieval["query"] == "Shanghai"


def test_instances_are_isolated_and_empty_search_does_not_call_llm():
    first, _, _ = backend([analysis(), decision()])
    second, llm, _ = backend([])
    first.ingest(session("Berlin"))
    assert second.retrieve("Where?", 3) == [] and llm.calls == []
    assert second.snapshot()["notes"] == []


def test_failed_evolution_or_invalid_links_cannot_commit_partial_note():
    for bad in [RuntimeError("offline error"), decision(True, actions=["strengthen"],
                    suggested_connections=[-1], tags_to_update=["bad"])]:
        obj, _, _ = backend([analysis(), decision(), analysis(), bad])
        obj.ingest(session("Berlin"))
        before = obj.snapshot()
        with pytest.raises((ValueError, RuntimeError)):
            obj.ingest(session("Shanghai", name="session_2"))
        assert obj.snapshot() == before


def test_invalid_metadata_is_not_silently_replaced_by_generic_memory():
    obj, _, _ = backend([{"keywords": "not a list", "context": "x", "tags": []}])
    with pytest.raises(ValueError):
        obj.ingest(session("Berlin"))
    assert obj.snapshot()["notes"] == []


def test_api_llm_and_embedding_have_independent_configuration(monkeypatch):
    import httpx
    from openai import OpenAI
    from langmem_eval.methods.amem.config import AMemSettings
    from langmem_eval.methods.amem.clients import OpenAIController
    from langmem_eval.embeddings import APIEmbedder
    monkeypatch.setenv("OPENAI_BASE_URL", "https://chat.invalid/v1")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("EMBEDDING_DIMS", "2")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.invalid/v1")
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"enable_thinking": false}')
    settings = AMemSettings.from_env()
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        if request.url.path.endswith("embeddings"):
            assert all(isinstance(t, str) for t in body["input"])
            return httpx.Response(200, json={"object": "list", "model": "test-embedding",
                "data": [{"object": "embedding", "index": i, "embedding": [1., 0.]}
                         for i, _ in enumerate(body["input"])],
                "usage": {"prompt_tokens": 2, "total_tokens": 2}})
        return httpx.Response(200, json={"id": "offline", "object": "chat.completion",
            "created": 0, "model": "offline-model", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": '{"keywords":"Berlin"}'},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}})
    def client(url):
        return OpenAI(api_key="offline", base_url=url,
                      http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    from agents_memory.usage import start, reset, phase, get_report
    start(); reset()
    with phase("retrieve"):
        ctrl = OpenAIController("offline-model", settings, client=client("https://chat.invalid/v1"))
        assert ctrl.complete("keywords?", {"type": "object", "properties": {
            "keywords": {"type": "string"}}, "required": ["keywords"]}) == {"keywords": "Berlin"}
        emb = APIEmbedder(settings, client=client(settings.embedding_base_url))
        assert np.asarray(emb.encode(["Berlin"] * 11)).shape == (11, 2)
    assert requests[0][1]["enable_thinking"] is False
    assert len(requests) == 3 and all("embedding.invalid" in r[0] for r in requests[1:])
    assert get_report()["method"]["total_tokens"] > 0
    assert "offline" not in json.dumps(settings.public_config())


def test_unified_adapter_records_provenance_and_keeps_gold_out_of_requests(monkeypatch):
    import openai
    from langmem_eval import registry
    from langmem_eval.adapter import run_method
    obj, llm, _ = backend([analysis(), decision(), {"keywords": "Berlin"}])
    monkeypatch.setattr(registry, "create_backend", lambda *args: obj)
    sent = []
    class Client:
        def __init__(self, **kwargs):
            self.chat = NS(completions=self)
        def create(self, **kwargs):
            sent.append(kwargs)
            return NS(choices=[NS(message=NS(content="Berlin"), finish_reason="stop")])
    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"enable_thinking":false}')
    conv = {"conversation": {"session_1_date_time": "2025-01-01", "session_1": [
        {"speaker": "Alice", "text": "Berlin"}]}, "qa": [
        {"question": "Where?", "answer": "GOLD_MUST_NOT_LEAK", "category": 1}]}
    row = run_method("amem", conv, "offline-model", False)[0]
    trace = row["answer_trace"]
    assert trace["method_config"]["upstream_commit"] == "0c8039f28fdcc08189a23c07a3437d9d2482f9c2"
    assert trace["retrieval_trace"]["query"] == "Berlin"
    assert trace["context"] and row["answer_status"] == "ok"
    assert "GOLD_MUST_NOT_LEAK" not in json.dumps([llm.calls, sent])
    assert sent[0]["extra_body"] == {"enable_thinking": False}
