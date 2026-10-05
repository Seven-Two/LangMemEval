"""Evidence for extension boundaries, isolated imports and resource ownership."""
from argparse import ArgumentParser
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from langmem_eval.configuration import add_model_arguments, configured_environment, environment_flags
from langmem_eval.interfaces import Session
from langmem_eval.lifecycle import managed_backend
from langmem_eval.registry import METHODS, MethodOption, create_backend, register_method


@pytest.fixture
def extension(tmp_path, monkeypatch):
    import langmem_eval.methods as package
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "examples/method_template", tmp_path / "my_method")
    monkeypatch.setattr(package, "__path__", [*package.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    try:
        yield
    finally:
        METHODS.pop("my_method", None)
        for name in list(sys.modules):
            if name == "langmem_eval.methods.my_method" or name.startswith("langmem_eval.methods.my_method."):
                sys.modules.pop(name)


def test_new_method_cli_dotenv_and_settings_need_no_core_edits(extension, tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED")
    monkeypatch.setenv("MY_METHOD_WINDOW", "3")
    env = tmp_path / "settings.env"
    env.write_text("MY_METHOD_WINDOW=2\n", encoding="utf-8")
    parser = ArgumentParser()
    add_model_arguments(parser)
    for cli, expected, source in [([], 2, ".env"), (["--my-method-window", "1"], 1, "cli")]:
        with configured_environment(parser.parse_args(["--env-file", str(env), *cli])) as sources:
            first, second = create_backend("my_method", "offline"), create_backend("my_method", "offline")
            first.ingest(Session("s1", "date", [{"content": "one"}, {"content": "two"}]))
            assert first.settings.window == expected and len(first.records) == expected
            assert second.records == [] and sources["MY_METHOD_WINDOW"] == source
        assert os.environ["MY_METHOD_WINDOW"] == "3"
    with pytest.raises(RuntimeError):
        with configured_environment(parser.parse_args(["--env-file", str(env)])):
            raise RuntimeError("experiment failed")
    assert os.environ["MY_METHOD_WINDOW"] == "3"


def test_new_method_full_runner_records_configuration_and_protocol(extension, tmp_path, monkeypatch):
    import openai
    from agents_memory import runner
    from langmem_eval.methods.my_method.config import Settings

    monkeypatch.delenv("MY_METHOD_WINDOW", raising=False)
    monkeypatch.setattr(runner, "start", lambda: None)
    sent = []

    class Client:
        def __init__(self, **kwargs):
            self.chat = NS(completions=self)

        def create(self, **kwargs):
            sent.append(kwargs)
            return NS(choices=[NS(message=NS(content="Berlin"), finish_reason="stop")])

    monkeypatch.setattr(openai, "OpenAI", Client)
    data = tmp_path / "input.json"
    data.write_text(json.dumps([{"conversation": {"session_1": [
        {"speaker": "Alice", "text": "OLD_MEMORY"}, {"speaker": "Alice", "text": "Berlin"}]},
        "qa": [{"question": "Where?", "answer": "PRIVATE_GOLD", "category": 1}]}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["eval", "--systems", "my_method", "--my-method-window", "1",
        "--llm-model", "offline", "--data-file", str(data), "--skip-judge", "--output-dir", str(tmp_path)])
    runner.main()
    result = json.loads(next(tmp_path.glob("my_method_*_results.json")).read_text(encoding="utf-8"))
    assert result["run_status"] == "complete"
    assert result["config"]["method_settings"] == {"window": 1}
    assert result["config"]["configuration_sources"]["MY_METHOD_WINDOW"] == "cli"
    trace = result["results"][0]["answer_trace"]
    assert trace["method_config"]["settings"] == {"window": 1}
    assert "Berlin" in trace["context"] and "OLD_MEMORY" not in trace["context"]
    assert "PRIVATE_GOLD" not in json.dumps(sent)
    assert Settings.from_env().window == 100  # CLI overrides never leak


def test_missing_selected_dependency_is_explicit_and_lazy(monkeypatch):
    from langmem_eval.registry import Method, check_dependencies
    method = Method(factory=lambda model: None, architecture="test", infrastructure="test",
                    dependencies=("missing_test_sdk",), extra="test-extra")
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "missing_test_sdk" else original(name))
    with pytest.raises(ValueError, match="uv sync --locked --extra test-extra"):
        check_dependencies("test_method", method)


def test_duplicate_configuration_bindings_fail_before_model_calls(monkeypatch):
    @register_method("conflicting_test", architecture="test", options=(
        MethodOption("amem-neighbor-k", "UNRELATED", int),))
    def create(model):
        raise AssertionError("Must fail before construction")
    try:
        with pytest.raises(ValueError, match="collision"):
            environment_flags()
    finally:
        METHODS.pop("conflicting_test")


def test_cli_and_scoring_import_no_optional_baseline_implementations():
    code = '''
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sentence_transformers', 'simplemem', 'mem0', 'graphiti_core'}:
            raise AssertionError('Unexpected optional SDK: ' + fullname)
        if fullname.startswith('agents_memory.systems'):
            raise AssertionError('Removed legacy baseline path: ' + fullname)
        if fullname.endswith('.backend') and fullname.startswith('langmem_eval.methods.'):
            raise AssertionError('Method constructed during discovery: ' + fullname)
sys.meta_path.insert(0, Guard())
from agents_memory import runner
from langmem_eval.registry import discover_methods
from langmem_eval.evaluation import evaluate_questions
assert set(discover_methods()) == {'amem', 'langmem'}
sys.argv = ['eval', '--help']
try:
    runner.main()
except SystemExit as exc:
    assert exc.code == 0
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--amem-neighbor-k" in result.stdout


@pytest.mark.parametrize("failure", [False, True])
def test_backend_cleanup_runs_on_all_exits(failure):
    closed = []
    backend = NS(close=lambda: closed.append(True))
    if failure:
        with pytest.raises(RuntimeError, match="write failed"):
            with managed_backend(backend):
                raise RuntimeError("write failed")
    else:
        with managed_backend(backend):
            pass
    assert closed == [True]


def test_cleanup_failure_does_not_replace_original_error():
    def close():
        raise ValueError("cleanup failed")
    with pytest.raises(RuntimeError, match="original"):
        with managed_backend(NS(close=close)):
            raise RuntimeError("original")
    with pytest.raises(ValueError, match="cleanup failed"):
        with managed_backend(NS(close=close)):
            pass


def test_adapter_closes_backend_after_history_failure(monkeypatch):
    from langmem_eval import registry
    from langmem_eval.adapter import run_method
    closed = []

    class Backend:
        def ingest(self, session):
            raise RuntimeError("write failed")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(registry, "create_backend", lambda *args: Backend())
    conv = {"conversation": {"session_1": [{"speaker": "Alice", "text": "Berlin"}]}, "qa": []}
    with pytest.raises(RuntimeError, match="write failed"):
        run_method("offline", conv, "offline", False)
    assert closed == [True]


def test_amem_construction_failure_closes_owned_controller(monkeypatch):
    from langmem_eval.methods.amem import backend
    closed = []
    monkeypatch.setattr(backend, "OpenAIController", lambda *args: NS(close=lambda: closed.append(True)))

    def fail(*args):
        raise RuntimeError("embedding initialization")

    monkeypatch.setattr(backend, "create_embedder", fail)
    with pytest.raises(RuntimeError, match="embedding initialization"):
        backend.AMemBackend("offline")
    assert closed == [True]


def test_injected_resources_remain_owned_by_caller():
    from langmem_eval.methods.amem.backend import AMemBackend
    def forbidden():
        raise AssertionError("Caller owns injected resources")
    obj = AMemBackend("offline", controller=NS(close=forbidden), embedder=NS(close=forbidden))
    obj.close()
    obj.close()


def test_typed_history_preserves_provenance():
    session = Session("s1", "2025-01-01", [{"content": json.dumps({"speaker": "Alice", "text": "Berlin"})}])
    turn, = session.turns()
    assert (turn.speaker, turn.text, turn.date, turn.source_id) == ("Alice", "Berlin", "2025-01-01", "s1:0")
    with pytest.raises(ValueError, match="speaker"):
        list(Session("s1", "date", [{"content": '{"speaker": 1, "text": "x"}'}]).turns())


def test_evolution_transition_is_pure_and_independent_of_response_objects():
    from langmem_eval.methods.amem.evolution import apply_evolution
    from langmem_eval.methods.amem.memory import Note
    from test_amem_contract import decision
    old = Note("old", "s:0", "s", "date", "Berlin", ["home"], "home", ["personal"])
    new = Note("new", "s:1", "s", "date", "Shanghai", ["move"], "move", ["personal"])
    command = decision(True, actions=["strengthen", "update_neighbor"], suggested_connections=[0, 99],
                       tags_to_update=["move"], new_context_neighborhood=["former home"],
                       new_tags_neighborhood=[["old"]])
    before = deepcopy(([old], new, command))
    result = apply_evolution([old], new, command, [0])
    assert ([old], new, command) == before
    assert result.notes[0].context == "former home" and result.notes[1].links == ["old"]
    command["tags_to_update"].append("mutated later")
    assert result.notes[1].tags == ["move"]
    assert result.adjustments["filtered_links"] == 1
