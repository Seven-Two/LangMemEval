"""Package discovery, extension template and backwards-compatible imports."""
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from langmem_eval.interfaces import Session
from langmem_eval.registry import METHODS, create_backend, discover_methods


def test_discovery_does_not_import_implementations_or_optional_sdks():
    # A fresh interpreter catches eager imports even when other tests have already
    # loaded the algorithm and model SDKs. The finder fails before any model loads.
    code = '''
import importlib.abc
import sys
class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sentence_transformers', 'openai', 'langmem', 'langgraph', 'numpy'}:
            raise AssertionError('Eager SDK import: ' + fullname)
        if fullname in {'langmem_eval.methods.amem.backend', 'langmem_eval.methods.langmem.backend'}:
            raise AssertionError('Eager algorithm import: ' + fullname)
sys.meta_path.insert(0, RejectHeavyImports())
from langmem_eval.registry import discover_methods
assert {'amem', 'langmem'} <= discover_methods().keys()
assert discover_methods() == discover_methods()
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_copyable_template_registers_and_isolates_state(tmp_path, monkeypatch):
    import langmem_eval.methods as package
    root = Path(__file__).resolve().parents[1]
    assert "my_method" not in discover_methods()  # examples are not auto-discovered
    shutil.copytree(root / "examples/method_template", tmp_path / "my_method")
    monkeypatch.setattr(package, "__path__", [*package.__path__, str(tmp_path)])
    try:
        importlib.invalidate_caches()
        assert "my_method" in discover_methods()
        first = create_backend("my_method", "offline")
        second = create_backend("my_method", "offline")
        first.ingest(Session("s1", "2025-01-01", [
            {"role": "user", "content": '{"speaker":"Alice","text":"Berlin"}'},
            {"role": "user", "content": '{"speaker":"Alice","text":"Shanghai"}'}]))
        assert first.retrieve("Where?", 1) == ['{"speaker":"Alice","text":"Shanghai"}']
        assert second.retrieve("Where?", 1) == []
        assert first.describe()["note_count"] == 2
    finally:
        METHODS.pop("my_method", None)
        for name in list(sys.modules):
            if name == "langmem_eval.methods.my_method" or name.startswith("langmem_eval.methods.my_method."):
                sys.modules.pop(name)


def test_old_imports_resolve_to_single_canonical_implementation():
    from langmem_eval import amem, backend, benchmark, _amem_prompts
    from langmem_eval.interfaces import MemoryBackend
    from langmem_eval.methods.amem.backend import AMemBackend
    from langmem_eval.methods.amem.config import AMemSettings
    from langmem_eval.methods.amem import prompts
    from langmem_eval.methods.langmem.backend import LangMemBackend
    assert amem.AMemBackend is AMemBackend
    assert amem.AMemSettings is AMemSettings
    assert backend.LangMemBackend is LangMemBackend
    assert benchmark.Session is Session
    assert benchmark.MemoryBackend is MemoryBackend
    assert _amem_prompts.ANALYSIS_PROMPT is prompts.ANALYSIS_PROMPT
    assert _amem_prompts.EVOLUTION_PROMPT is prompts.EVOLUTION_PROMPT
    assert _amem_prompts.QUERY_PROMPT is prompts.QUERY_PROMPT
