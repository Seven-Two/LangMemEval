"""Keep the supported method scope small while preserving the extension contract."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tomllib

import pytest

from agents_memory import usage
from langmem_eval.adapter import run_method
from langmem_eval.registry import discover_methods


def conversation():
    return {"conversation": {"session_1_date_time": "2025-01-01", "session_1": [
        {"speaker": "Alice", "text": "Alice lives in Berlin", "dia_id": "s1:1"}],
        "session_2_date_time": "2025-02-01", "session_2": [
        {"speaker": "Bob", "text": "Bob likes tea", "dia_id": "s2:1"}]},
        "qa": [{"question": "Where does Alice live?", "answer": "SECRET_GOLD", "category": 1}]}


def test_only_requested_methods_are_bundled_and_share_one_runner():
    assert set(discover_methods()) == {"amem", "langmem"}
    root = Path(__file__).resolve().parents[1]
    methods = root / "src/langmem_eval/methods"
    assert {p.name for p in methods.iterdir() if (p / "__init__.py").is_file()} == {"amem", "langmem"}
    assert not (root / "src/agents_memory/training").exists()
    runner = (root / "src/agents_memory/runner.py").read_text(encoding="utf-8")
    assert "SYSTEMS" not in runner and '"native"' not in runner
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert set(project["optional-dependencies"]) == {"dev", "aml", "amem", "charts"}
    locked = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    assert {p["name"] for p in locked["package"]}.isdisjoint({
        "mem0ai", "simplemem", "graphiti-core", "hindsight-client", "memu-py", "trl", "peft", "bitsandbytes"})


def test_batch_finalize_precedes_qa_and_failure_is_not_silently_answered(monkeypatch):
    from langmem_eval import registry
    class Backend:
        def __init__(self): self.calls = []
        def ingest(self, session): self.calls.append(session.id)
        def finalize(self):
            assert self.calls == ["session_1", "session_2"]
            assert usage._phase.get() == "write"
            raise RuntimeError("batch write failed")
        def retrieve(self, *args): raise AssertionError("Must not retrieve")
        def close(self): self.calls.append("close")
    backend = Backend()
    monkeypatch.setattr(registry, "create_backend", lambda *args: backend)
    with pytest.raises(RuntimeError, match="batch write failed"):
        run_method("offline", conversation(), "offline", False)
    assert backend.calls[-1] == "close"


def test_real_runner_continues_next_conversation_after_finalize_failure(tmp_path, monkeypatch):
    import openai
    from agents_memory import runner
    from langmem_eval import registry
    created = []
    class Backend:
        def __init__(self):
            self.records = []
            self.ordinal = len(created)
            self.closed = False
            created.append(self)
        def ingest(self, session):
            self.records.extend(turn.text for turn in session.turns())
        def retrieve(self, query, limit):
            return self.records[:limit]
        def finalize(self):
            if self.ordinal == 0:
                raise RuntimeError("first write failed")
        def close(self): self.closed = True
    monkeypatch.setattr(registry, "create_backend", lambda *a: Backend())
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: NS(chat=NS(completions=NS(create=lambda **kw:
        NS(choices=[NS(message=NS(content="Berlin"), finish_reason="stop")])))))
    monkeypatch.setattr(runner, "start", lambda: None)
    path = tmp_path / "input.json"
    path.write_text(json.dumps([conversation(), conversation()]), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["eval", "--systems", "amem", "--num-samples", "2",
        "--skip-judge", "--data-file", str(path), "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2 and all(b.closed for b in created)
    result = json.loads(next(tmp_path.glob("amem_*_results.json")).read_text(encoding="utf-8"))
    assert result["summary"]["n_questions"] == 2 and result["summary"]["coverage"] == .5
    assert [r["answer_status"] for r in result["results"]] == ["error", "ok"]
    assert result["config"]["adapter_protocol"] == "unified_v1"
    assert result["config"]["context_trace"] == "per_question"
