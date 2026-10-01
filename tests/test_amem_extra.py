"""Edge cases beyond the immutable A-Mem acceptance metric; never use real APIs."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import zipfile

import numpy as np
import pytest

from langmem_eval.amem import AMemSettings, OpenAIController, LocalEmbedder
from langmem_eval.model_api import validate_base_url, llm_extra_body
from test_amem_contract import backend, analysis, decision, session


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline boundary tests must not use network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")


@pytest.mark.parametrize("url", ["https://host/v1/chat/completions/",
    "https://host/v1/embeddings", "https://key@host/v1", "https://host/v1?key=secret"])
def test_reject_misconfigured_endpoints(url):
    with pytest.raises(ValueError):
        validate_base_url(url)


def test_extra_payload_cannot_override_answer_protocol(monkeypatch):
    for value in ['[]', '{"model":"other"}', '{"messages":[]}', 'broken']:
        monkeypatch.setenv("LLM_EXTRA_BODY", value)
        with pytest.raises(ValueError):
            llm_extra_body()


def test_configuration_redacts_private_options():
    config = AMemSettings(embedding_api_key="sensitive-token",
        extra_body={"enable_thinking": False, "private-service-option": "sensitive-option"})
    public = json.dumps(config.public_config())
    assert "sensitive" not in public and "sensitive" not in repr(config)
    assert config.public_config()["extra_body"] == {"enable_thinking": False}


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "prompt"])
def test_chat_modes_validate_output_and_reject_truncation(mode):
    sent = []
    choice = NS(message=NS(content='```json\n{"keywords":"Berlin"}\n```'), finish_reason="stop")
    def create(**kwargs):
        sent.append(kwargs)
        return NS(choices=[choice])
    client = NS(chat=NS(completions=NS(create=create)))
    ctrl = OpenAIController("offline", AMemSettings(response_format=mode), client=client)
    schema = {"type": "object", "required": ["keywords"], "properties": {"keywords": {"type": "string"}}}
    assert ctrl.complete("Produce JSON", schema) == {"keywords": "Berlin"}
    if mode == "prompt":
        assert "response_format" not in sent[0]
    else:
        assert sent[0]["response_format"]["type"] == mode
    choice.finish_reason = "length"
    with pytest.raises(ValueError, match="truncated"):
        ctrl.complete("Produce JSON", schema)
    choice.finish_reason = "stop"
    choice.message.content = '{"keywords":[]}'
    with pytest.raises(ValueError):
        ctrl.complete("Produce JSON", schema)


@pytest.mark.parametrize("bad", [np.array([[np.nan, 0.]]), np.array([[0., 0.]]), np.array([[1., 2., 3.]])])
def test_bad_embedding_rolls_back_neighbor_updates_and_new_note(bad):
    obj, _, encoder = backend([analysis(), decision(), analysis(), decision(True,
        actions=["update_neighbor"], new_context_neighborhood=["changed"],
        new_tags_neighborhood=[["changed"]])])
    obj.ingest(session("Berlin"))
    before, vectors = obj.snapshot(), obj.vectors.copy()
    original = encoder.encode
    calls = 0
    def encode(texts):
        nonlocal calls
        calls += 1
        return original(texts) if calls == 1 else bad  # fail after evolution, at insertion
    encoder.encode = encode
    with pytest.raises(ValueError):
        obj.ingest(session("Shanghai"))
    assert obj.snapshot() == before
    assert np.array_equal(obj.vectors, vectors)


def test_local_embedding_records_unknown_cost_without_loading_weights(monkeypatch):
    from langmem_eval import amem
    from agents_memory.usage import reset, phase, get_report
    calls = []
    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.array([[1., 0.] for _ in texts])
    monkeypatch.setattr(amem, "_load_local_model", lambda *args: FakeModel())
    reset()
    with phase("write"):
        vectors = LocalEmbedder(AMemSettings()).encode(["test"])
    assert vectors.shape == (1, 2) and calls[0][1]["convert_to_numpy"] is True
    report = get_report()
    assert report["method"]["calls"] == 1
    assert report["method"]["calls_without_price"] == 1


def test_distribution_contains_amem_and_license():
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("*.whl"))
    if not wheels:
        pytest.skip("Build the distribution before testing wheel contents")
    with zipfile.ZipFile(max(wheels, key=lambda p: p.stat().st_mtime)) as archive:
        for name in ("langmem_eval/amem.py", "langmem_eval/_amem_prompts.py",
                     "langmem_eval/methods/amem.py", "langmem_eval/model_api.py"):
            assert name in archive.namelist()
        assert archive.read("langmem_eval/AMEM_LICENSE") == (root / "third_party/amem/LICENSE").read_bytes()


def test_sync_evaluation_failure_trace_and_judge_accounting(monkeypatch):
    from langmem_eval.evaluation import evaluate_questions
    from agents_memory.systems import _helpers as scoring
    from agents_memory.usage import reset, get_report, record_external_usage
    conv = {"sample_id": "test", "qa": [
        {"question": "fail", "answer": "", "category": 1},
        {"question": "pass", "answer": "Berlin", "category": 1}]}
    def answer(question):
        answer.trace = {"failure_stage": "retrieve" if question == "fail" else None}
        if question == "fail":
            raise ValueError("offline error")
        return "Berlin"
    def judge(*args, **kwargs):
        record_external_usage(provider="offline", model="judge", prompt_tokens=1, completion_tokens=1)
        raise RuntimeError("offline judge error")
    monkeypatch.setattr(scoring, "evaluate_longmemeval", judge)
    reset()
    rows = evaluate_questions(conv, answer, True, judge_fn="longmemeval")
    assert rows[0]["f1"] == 0 and rows[0]["error"]["stage"] == "retrieve"
    assert rows[0]["judge_status"] == "not_run"
    assert rows[1]["answer_status"] == "ok" and rows[1]["judge_status"] == "error"
    assert rows[1]["status"] == "error" and rows[1]["f1"] == 1
    assert rows[0]["answer_trace"]["failure_stage"] == "retrieve"  # copied per question
    assert get_report()["judge"]["calls"] == 1
    assert get_report()["method"]["calls"] == 0
