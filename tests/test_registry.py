import importlib
import sys

import pytest

from langmem_eval.registry import METHODS, create_backend, discover_methods, register_method


def test_invalid_duplicate_and_unknown():
    discover_methods()
    with pytest.raises(ValueError, match="already registered"):
        register_method("langmem", architecture="test")(lambda model: None)
    for name in ["all", "bad-name", "Bad"]:
        with pytest.raises(ValueError, match="Invalid"):
            register_method(name, architecture="test")
    with pytest.raises(ValueError, match="Unknown method"):
        create_backend("missing_method", "offline")


def test_new_file_discovery_and_memeval_registration(tmp_path, monkeypatch):
    import langmem_eval.methods as package
    import agents_memory.systems as systems

    (tmp_path / "demo.py").write_text('''
from langmem_eval.registry import register_method
@register_method("test_demo", architecture="offline test")
class Backend:
    def __init__(self, model):
        self.model = model
        self.sessions = []
    def ingest(self, session):
        self.sessions.append(session)
    def retrieve(self, query, limit):
        return ["Berlin"][:limit]
''', encoding="utf-8")
    monkeypatch.setattr(package, "__path__", [*package.__path__, str(tmp_path)])
    bridge = importlib.import_module("agents_memory.systems.langmem")
    try:
        importlib.invalidate_caches()
        importlib.reload(bridge)
        importlib.reload(systems)
        assert "test_demo" in systems.SYSTEMS
        assert systems.SYSTEMS["test_demo"]["architecture"] == "offline test"
        first = create_backend("test_demo", "offline")
        second = create_backend("test_demo", "offline")
        first.ingest("history")
        assert second.sessions == []
        assert first.retrieve("q", 1) == ["Berlin"]
        assert discover_methods() == discover_methods()
    finally:
        METHODS.pop("test_demo", None)
        sys.modules.pop("langmem_eval.methods.demo", None)
        monkeypatch.undo()
        importlib.reload(bridge)
        importlib.reload(systems)
