"""Offline regression tests for research protocol and fair denominators."""
import json
from types import SimpleNamespace as NS

import pytest

from agents_memory.experiment import compute_summary, failed_results, freeze_manifest, normalize_results
from agents_memory.systems._helpers import _qa_results
from langmem_eval.benchmark import select_context, token_encoding
from langmem_eval.protocol import AnswerProtocol


def conversation():
    return {"sample_id": "original", "conversation": {
        "session_1": [{"speaker": "Alice", "text": "I live in Berlin"}]},
        "qa": [{"question": "Where?", "answer": "Berlin", "category": 1},
               {"question": "Unknown?", "answer": "", "category": 1}]}


@pytest.fixture(autouse=True)
def clean_protocol_env(monkeypatch):
    import os
    for key in list(os.environ):
        if key.startswith("EVAL_") or key in {"LANGMEM_TOP_K", "LANGMEM_MAX_CONTEXT_CHARS"}:
            monkeypatch.delenv(key)


def test_budget_full_serialization_unicode_and_oversize_skip():
    enc = token_encoding("cl100k_base")
    records = ["很长的记录" * 100, "上海，2025年。", "hello <|endoftext|>", ""]
    expected = "\n\n".join(records[1:3])
    budget = len(enc.encode(expected, disallowed_special=()))
    result = select_context(records, max_tokens=budget)
    assert result.context == expected
    assert result.token_count == budget
    assert result.selected_indices == [1, 2]
    assert result.dropped_indices == [0, 3]
    assert select_context(records, max_tokens=0).context == ""
    assert select_context([], max_tokens=1).token_count == 0
    with pytest.raises(ValueError):
        select_context(records, max_tokens=-1)


def test_protocol_defaults_overrides_and_validation(monkeypatch):
    assert AnswerProtocol.from_env().max_output_tokens == 256
    assert AnswerProtocol.from_env("longmemeval").max_output_tokens == 512
    monkeypatch.setenv("EVAL_MAX_OUTPUT_TOKENS", "900")
    monkeypatch.setenv("EVAL_EMPTY_CONTEXT", "answer")
    protocol = AnswerProtocol.from_env()
    assert protocol.max_output_tokens == 900 and protocol.empty_context == "answer"
    assert "1-5" not in protocol.messages("q", "c")[0]["content"]
    monkeypatch.setenv("EVAL_CONTEXT_TOKENS", "-1")
    with pytest.raises(ValueError):
        AnswerProtocol.from_env()


def test_adapter_traces_exact_request_and_empty_policy(monkeypatch):
    import openai
    import langmem_eval.registry as registry
    from langmem_eval.adapter import run_method
    records, requests = ["Berlin"], []
    class Backend:
        def ingest(self, session):
            pass
        def retrieve(self, question, limit):
            return list(records)
    class Client:
        def __init__(self, **kwargs):
            self.chat = NS(completions=self)
        def create(self, **kwargs):
            requests.append(kwargs)
            return NS(choices=[NS(message=NS(content="Berlin"), finish_reason="stop")])
    monkeypatch.setattr(registry, "create_backend", lambda *a: Backend())
    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setenv("EVAL_MAX_OUTPUT_TOKENS", "321")
    conv = conversation()
    conv["qa"] = conv["qa"][:1]
    row = run_method("offline", conv, "offline", False)[0]
    assert requests[0]["max_tokens"] == 321
    assert row["answer_trace"]["messages"] == requests[0]["messages"]
    assert row["answer_trace"]["context"] == "Berlin"
    records.clear()
    row = run_method("offline", conv, "offline", False)[0]
    assert row["predicted"] == "None" and len(requests) == 1
    assert row["answer_trace"]["messages"] is None
    monkeypatch.setenv("EVAL_EMPTY_CONTEXT", "answer")
    row = run_method("offline", conv, "offline", False)[0]
    assert len(requests) == 2 and row["answer_trace"]["answer_called"]


def test_failed_answer_not_correct_abstention_and_judge_failure(monkeypatch):
    import agents_memory.systems._helpers as helpers
    convs, _ = freeze_manifest([conversation()])
    def fail(q):
        raise RuntimeError("unavailable")
    rows = _qa_results(convs[0], fail, False)
    assert all(r["f1"] == 0 and r["status"] == "error" for r in rows)
    def fail_judge(*a, **kw):
        raise RuntimeError("judge unavailable")
    monkeypatch.setattr(helpers, "evaluate_longmemeval", fail_judge)
    rows = _qa_results(convs[0], lambda q: "Berlin", True, judge_fn="longmemeval")
    assert all(r["answer_status"] == "ok" and r["judge_status"] == "error" for r in rows)
    summary = compute_summary(rows, True, "longmemeval")
    assert summary["answer_coverage"] == 1 and summary["coverage"] == 0
    assert summary["longmemeval_accuracy"] is None


def test_manifest_missing_results_all_failed_and_duplicate_ids():
    original = conversation()
    convs, manifest = freeze_manifest([original, original])
    assert original["sample_id"] == "original"
    assert len({q["evaluation_id"] for q in manifest["questions"]}) == 4
    conv = convs[0]
    row = {"question": "Where?", "predicted": "Berlin", "f1": -100}
    rows = normalize_results(conv, [row], {}, False, None)
    assert len(rows) == 2 and rows[0]["f1"] == 1 and rows[1]["f1"] == 0
    summary = compute_summary(rows)
    assert summary["coverage"] == .5 and summary["n_questions"] == 2
    assert summary["overall_f1_mean"] == .5
    failed = failed_results(conv, {}, True, stage="conversation", error_type="RuntimeError")
    summary = compute_summary(failed, True, "longmemeval")
    assert summary["run_status"] == "incomplete" and summary["n_questions"] == 2
    assert summary["longmemeval_accuracy_failures_zero"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        normalize_results(conv, [{**row, "evaluation_id": conv["qa"][0]["evaluation_id"]}] * 2, {}, False, None)


def test_lme_mixed_failure_uses_full_denominator_and_null_primary():
    convs, _ = freeze_manifest([conversation()])
    rows = normalize_results(convs[0], [
        {"question": "Where?", "predicted": "Berlin", "longmemeval_correct": 1},
        {"question": "Unknown?", "predicted": "None", "longmemeval_correct": None,
         "judge_status": "error"}], {}, True, "longmemeval")
    summary = compute_summary(rows, True, "longmemeval")
    assert summary["longmemeval_accuracy"] is None
    assert summary["longmemeval_accuracy_failures_zero"] == .5
    assert summary["judge_coverage"] == .5
    assert summary["answer_coverage"] == 1
    assert rows[1]["longmemeval_correct"] is None


def test_runner_same_manifest_failed_system_preserved(tmp_path, monkeypatch):
    import agents_memory.runner as runner
    data = tmp_path / "input.json"
    data.write_text(json.dumps([conversation()]), encoding="utf-8")
    def good(conv, model, run_judge, **kwargs):
        # A method cannot mutate the source seen by the next system.
        conv["conversation"].clear()
        return _qa_results(conv, lambda q: "Berlin" if q == "Where?" else "None", False)
    def bad(conv, *a, **kwargs):
        assert conv["conversation"]
        raise RuntimeError("offline failure")
    systems = {name: {"fn": fn, "architecture": "test", "infrastructure": "none"}
               for name, fn in (("good", good), ("bad", bad))}
    monkeypatch.setattr(runner, "SYSTEMS", systems)
    monkeypatch.setattr(runner, "start", lambda: None)
    monkeypatch.setattr("sys.argv", ["eval", "--systems", "good,bad", "--skip-judge",
                                    "--data-file", str(data), "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    good_payload = json.loads(next(tmp_path.glob("good_*_results.json")).read_text(encoding="utf-8"))
    bad_payload = json.loads(next(tmp_path.glob("bad_*_results.json")).read_text(encoding="utf-8"))
    assert good_payload["manifest"] == bad_payload["manifest"]
    assert good_payload["summary"]["coverage"] == 1
    assert bad_payload["summary"]["n_questions"] == 2
    assert bad_payload["summary"]["coverage"] == 0
    assert bad_payload["run_status"] == "incomplete"
    assert bad_payload["cost_accounting"]["coverage"] == "partial"
    summary = json.loads(next(tmp_path.glob("benchmark_summary_*.json")).read_text(encoding="utf-8"))
    assert summary["run_status"] == "incomplete"
