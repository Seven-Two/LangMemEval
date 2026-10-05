"""Offline checks for persistent diagnostics and blocking-call visibility."""
import json
import threading
from types import SimpleNamespace as NS

import pytest

from agents_memory import diagnostics as log


def test_waiting_failure_chain_and_handler_cleanup(tmp_path, monkeypatch):
    waiting = threading.Event()
    original = log._emit
    def observed(status, fields, **kwargs):
        original(status, fields, **kwargs)
        if status == "waiting" and fields["stage"] == "embedding.load":
            waiting.set()
    monkeypatch.setattr(log, "_emit", observed)
    previous = log.logger.handlers[:], log.logger.level, log.logger.propagate
    path = tmp_path / "run.log"
    with pytest.raises(RuntimeError, match="driver too old"):
        with log.run_logging(path, heartbeat_seconds=0.01):
            with log.stage("conversation", conversation="c0"):
                with log.stage("embedding.load", device="cuda"):
                    assert waiting.wait(2), "Blocked stage must be visible before it finishes"
                    # FileHandler flushes each event, even while work is blocked.
                    assert 'status="waiting"' in path.read_text(encoding="utf-8")
                    try:
                        raise ValueError("underlying error")
                    except ValueError as exc:
                        raise RuntimeError("driver too old") from exc
    text = path.read_text(encoding="utf-8")
    assert 'conversation="c0"' in text and 'device="cuda"' in text
    assert "ValueError: underlying error" in text
    assert text.count("RuntimeError: driver too old") == 1
    assert 'status="failed"' in text and "elapsed_s=" in text
    assert (log.logger.handlers, log.logger.level, log.logger.propagate) == previous
    assert not any(t.name == "evaluation-progress" for t in threading.enumerate())
    second = tmp_path / "second.log"
    with log.run_logging(second):
        log.event("second.run")
    assert "second.run" not in path.read_text(encoding="utf-8")


def test_runner_logs_failure_and_keeps_manifest(tmp_path, monkeypatch):
    from agents_memory import runner
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"conversation": {"session_1": []}, "qa": [
        {"question": "q", "answer": "a", "category": 1}]}]), encoding="utf-8")
    def fail(*args, **kwargs):
        raise RuntimeError("device unavailable")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setattr(runner, "SYSTEMS", {"offline": {
        "fn": fail, "architecture": "test", "infrastructure": "none"}})
    monkeypatch.setattr(runner, "start", lambda: None)
    monkeypatch.setattr("sys.argv", ["eval", "--systems", "offline", "--skip-judge",
        "--data-file", str(data), "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    payload = json.loads(next(tmp_path.glob("offline_*_results.json")).read_text(encoding="utf-8"))
    text = (tmp_path / payload["config"]["runtime_log"]).read_text(encoding="utf-8")
    assert "RuntimeError: device unavailable" in text
    assert 'stage="dataset.load"' in text and 'stage="results.save"' in text
    assert 'run_status="incomplete"' in text
    assert payload["summary"]["coverage"] == 0
    assert payload["summary"]["n_questions"] == 1


def test_amem_write_retrieve_answer_logged_without_payloads(tmp_path, monkeypatch):
    import openai
    import numpy as np
    from langmem_eval import registry
    from langmem_eval.adapter import run_method
    from langmem_eval.methods.amem.backend import AMemBackend
    from langmem_eval.methods.amem.config import AMemSettings
    replies = iter([
        {"keywords": ["home"], "context": "home", "tags": ["home"]},
        {"should_evolve": False, "actions": [], "suggested_connections": [],
         "tags_to_update": [], "new_context_neighborhood": [], "new_tags_neighborhood": []},
        {"keywords": "home"},
    ])
    controller = NS(complete=lambda *args, **kwargs: next(replies))
    embedder = NS(encode=lambda texts: np.array([[1., 0.] for _ in texts]))
    monkeypatch.setattr(registry, "create_backend", lambda *args: AMemBackend(
        "offline", settings=AMemSettings(), controller=controller, embedder=embedder))
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(message=NS(content="Berlin"), finish_reason="stop")])
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: NS(chat=NS(completions=NS(create=create))))
    conv = {"sample_id": "c0", "conversation": {"session_1": [
        {"speaker": "Alice", "text": "private-history-marker Berlin"}]},
        "qa": [{"question": "private-question-marker", "answer": "Berlin", "category": 1}]}
    path = tmp_path / "amem.log"
    with log.run_logging(path):
        rows = run_method("amem", conv, "offline", False)
    text = path.read_text(encoding="utf-8")
    for name in ("backend.initialize", "write", "write.session", "amem.write.note", "amem.analyze",
                 "amem.evolve", "amem.query.rewrite", "retrieve", "answer.api"):
        assert f'stage="{name}"' in text
    assert 'message_index=1' in text and 'question_index=1' in text
    assert 'status="context.selected"' in text and 'status="question.result"' in text
    assert "private-history-marker" not in text and "private-question-marker" not in text
    assert len(calls) == 1 and rows[0]["f1"] == 1


def test_interrupted_stage_is_not_reported_done(tmp_path):
    path = tmp_path / "interrupt.log"
    with pytest.raises(KeyboardInterrupt):
        with log.run_logging(path):
            with log.stage("write"):
                raise KeyboardInterrupt
    text = path.read_text(encoding="utf-8")
    assert 'status="stopped"' in text
    assert 'status="done"' not in text


def test_concise_console_keeps_full_file_and_prints_failure_only_once(tmp_path, capsys):
    path = tmp_path / "concise.log"
    with pytest.raises(ValueError):
        with log.run_logging(path):
            with log.stage("conversation", conversation="c0"):
                for index in range(30):
                    log.event("llm.response", request_id=str(index), usage={"completion_tokens": 100})
                    log.event("waiting", elapsed_s=30)
                with log.stage("amem.evolve"):
                    raise ValueError("detailed-error-only-in-file")
    console = capsys.readouterr().err
    saved = path.read_text(encoding="utf-8")
    assert "console=concise" in console
    assert console.count("ERROR ") == 1 and "ValueError" in console
    assert str(path) in console
    assert "llm.response" not in console and "waiting" not in console
    assert "Traceback" not in console and "detailed-error-only-in-file" not in console
    assert saved.count('status="llm.response"') == 30
    assert saved.count('status="waiting"') == 30
    assert "Traceback" in saved and "ValueError: detailed-error-only-in-file" in saved


def test_full_console_keeps_request_details_and_traceback(tmp_path, capsys):
    path = tmp_path / "full.log"
    with pytest.raises(RuntimeError):
        with log.run_logging(path, console_mode="full"):
            log.event("llm.response", usage={"completion_tokens": 100})
            with log.stage("answer.api"):
                raise RuntimeError("offline failure")
    console = capsys.readouterr().err
    saved = path.read_text(encoding="utf-8")
    for text in (console, saved):
        assert 'status="llm.response"' in text and '"completion_tokens": 100' in text
        assert "Traceback" in text and "RuntimeError: offline failure" in text


def test_invalid_console_mode_does_not_open_log_or_change_handlers(tmp_path):
    previous = log.logger.handlers[:], log.logger.level, log.logger.propagate
    path = tmp_path / "invalid.log"
    with pytest.raises(ValueError, match="console_mode"):
        with log.run_logging(path, console_mode="invalid"):
            pass
    assert not path.exists()
    assert (log.logger.handlers, log.logger.level, log.logger.propagate) == previous
