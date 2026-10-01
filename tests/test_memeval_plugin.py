"""Exercise upstream registry and scoring with a stubbed memory/model boundary."""
from types import SimpleNamespace

import pytest

pytest.importorskip("agents_memory")


def test_registered_plugin_and_real_scoring(monkeypatch):
    import openai
    import langmem_eval.backend
    from agents_memory.systems import SYSTEMS
    from langmem_eval.adapter import run

    assert SYSTEMS["langmem"]["fn"] is run
    assert all("prob" not in name for name in SYSTEMS if name.startswith("langmem"))

    writes = []

    class Backend:
        def __init__(self, model):
            pass
        def ingest(self, session):
            writes.append(str(session))
        def retrieve(self, question, limit):
            return ["Alice lives in Berlin"]

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="Berlin"))])

    monkeypatch.setattr(langmem_eval.backend, "LangMemBackend", Backend)
    monkeypatch.setattr(openai, "OpenAI", Client)
    conv = {"sample_id": "offline", "conversation": {
        "session_1": [{"speaker": "Alice", "text": "I live in Berlin"}]
    }, "qa": [{"question": "Where does Alice live?", "answer": "Berlin", "category": 1}]}
    result = SYSTEMS["langmem"]["fn"](conv, "offline", False)
    assert len(result) == 1 and result[0]["f1"] == 1.0
    assert len(writes) == 1 and "Where does Alice" not in writes[0]
