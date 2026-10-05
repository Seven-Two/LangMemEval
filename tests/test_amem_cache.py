"""Offline cache behavior: exact state, invalidation, failure and audit boundaries."""
from argparse import ArgumentParser
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from langmem_eval.benchmark import build_memory, extract_sessions
from langmem_eval.configuration import add_model_arguments, configured_environment
from langmem_eval.methods.amem.backend import AMemBackend
from langmem_eval.methods.amem.config import AMemSettings
from langmem_eval.methods.amem import cache
from test_amem_contract import Controller, Encoder, analysis, decision


def history():
    return {"sample_id": "original", "conversation": {
        "session_1_date_time": "2025-01-01", "session_1": [
            {"speaker": "Alice", "text": "Berlin", "dia_id": "s1:1"}],
        "session_2_date_time": "2025-02-01", "session_2": [
            {"speaker": "Alice", "text": "Shanghai", "dia_id": "s2:1"}]},
        "qa": [{"question": "Where?", "answer": "PRIVATE_GOLD", "category": 1}]}


def backend(tmp_path, mode="reuse", responses=None, **kwargs):
    settings = AMemSettings(cache_mode=mode, cache_dir=str(tmp_path / "cache"), **kwargs)
    if responses is None:
        responses = [analysis(), decision(), analysis(), decision(True,
            actions=["strengthen", "update_neighbor"], suggested_connections=[0],
            tags_to_update=["move"], new_context_neighborhood=["updated residence"],
            new_tags_neighborhood=[["past"]])]
    return AMemBackend("offline", settings=settings, controller=Controller(responses), embedder=Encoder())


def test_cache_hit_preserves_stale_index_links_counters_and_retrieval(tmp_path):
    original = backend(tmp_path)
    build_memory(history(), original)
    assert original.cache_info["status"] == "saved"
    # Evolution changed the old note; its original vector has not been rebuilt.
    assert original.notes[0].context == "updated residence"
    assert original.evolution_count == 1
    restored = backend(tmp_path, mode="require", responses=[])
    build_memory(history(), restored)
    assert restored.cache_info["status"] == "hit"
    assert restored.cache_info["build_id"] == original.cache_info["build_id"]
    assert restored.snapshot() == original.snapshot()
    np.testing.assert_array_equal(restored.vectors, original.vectors)
    assert restored.evolution_stats == original.evolution_stats
    assert restored.output_adjustments == original.output_adjustments
    assert restored.controller.calls == [] and restored.embedder.calls == []
    for obj in (original, restored):
        obj.controller.responses = [{"keywords": "Shanghai"}]
    assert restored.retrieve("Where?", 1) == original.retrieve("Where?", 1)
    assert restored.last_retrieval == original.last_retrieval


def test_qa_protocol_and_credentials_do_not_enter_cache_key_or_payload(tmp_path, monkeypatch):
    obj = backend(tmp_path, extra_body={"private_service_option": "payload-secret"})
    monkeypatch.setenv("OPENAI_API_KEY", "chat-secret")
    obj.settings = replace(obj.settings, embedding_api_key="embedding-secret")
    build_memory(history(), obj)
    raw = next((tmp_path / "cache").glob("*.json")).read_text(encoding="utf-8")
    assert all(secret not in raw for secret in ("PRIVATE_GOLD", "Where?", "payload-secret", "chat-secret", "embedding-secret"))
    changed = history()
    changed.update(sample_id="new-manifest-index", qa=[{"question": "New question", "answer": "New answer"}])
    monkeypatch.setenv("EVAL_TOP_K", "50")
    monkeypatch.setenv("EVAL_CONTEXT_TOKENS", "1000")
    monkeypatch.setenv("EVAL_MAX_OUTPUT_TOKENS", "400")
    monkeypatch.setenv("OPENAI_API_KEY", "rotated")
    restored = backend(tmp_path, mode="require", responses=[], extra_body=obj.settings.extra_body)
    restored.settings = replace(restored.settings, embedding_api_key="rotated-embedding")
    build_memory(changed, restored)
    assert restored.cache_info["key"] == obj.cache_info["key"]


@pytest.mark.parametrize("change", ["history", "date", "speaker", "model", "endpoint", "neighbor_k",
    "evolution_threshold", "temperature", "max_output_tokens", "extra_body", "embedding_model",
    "embedding_revision", "embedding_base_url", "code"])
def test_write_changes_invalidate_without_paid_rebuild_in_require_mode(tmp_path, monkeypatch, change):
    original = backend(tmp_path)
    build_memory(history(), original)
    obj = backend(tmp_path, mode="require", responses=[])
    conv = history()
    changes = {"neighbor_k": 7, "evolution_threshold": 10, "temperature": .1,
               "max_output_tokens": 500, "extra_body": {"enable_thinking": False},
               "embedding_model": "other-model", "embedding_revision": "fixed-commit",
               "embedding_base_url": "https://embedding.invalid/v1"}
    if change in changes:
        obj.settings = replace(obj.settings, **{change: changes[change]})
    elif change == "history":
        conv["conversation"]["session_1"][0]["text"] = "Paris"
    elif change == "date":
        conv["conversation"]["session_1_date_time"] = "2026-01-01"
    elif change == "speaker":
        conv["conversation"]["session_1"][0]["speaker"] = "Bob"
    elif change == "model":
        obj.model = "changed-model"
    elif change == "endpoint":
        monkeypatch.setenv("OPENAI_BASE_URL", "https://different.invalid/v1")
    else:
        monkeypatch.setattr(cache, "code_identity", lambda: {"code": "new"})
    with pytest.raises(ValueError, match="cache miss"):
        build_memory(conv, obj)
    assert obj.controller.calls == [] and obj.embedder.calls == []


@pytest.mark.parametrize("damage", ["json", "checksum", "vector_rows", "dangling_link", "duplicate_id"])
def test_corrupt_cache_fails_without_mutating_backend_or_rebuilding(tmp_path, damage):
    original = backend(tmp_path)
    build_memory(history(), original)
    path = next((tmp_path / "cache").glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if damage == "json":
        path.write_text("{", encoding="utf-8")
    else:
        if damage == "checksum": payload["state_sha256"] = "bad"
        if damage == "vector_rows": payload["state"]["vectors"] = [[1., 0.]]
        if damage == "dangling_link": payload["state"]["notes"][0]["links"] = ["unknown"]
        if damage == "duplicate_id": payload["state"]["notes"][1]["id"] = payload["state"]["notes"][0]["id"]
        if damage != "checksum": payload["state_sha256"] = cache.digest(payload["state"])
        path.write_text(json.dumps(payload), encoding="utf-8")
    obj = backend(tmp_path, responses=[])
    with pytest.raises(ValueError, match="Invalid A-Mem cache"):
        build_memory(history(), obj)
    assert obj.notes == [] and obj.vectors is None
    assert obj.controller.calls == [] and obj.embedder.calls == []


def test_disabled_cache_and_failed_writes_do_not_publish_state(tmp_path):
    off = backend(tmp_path, mode="off")
    build_memory(history(), off)
    assert not (tmp_path / "cache").exists()
    failed = backend(tmp_path, responses=[RuntimeError("write failed")])
    with pytest.raises(RuntimeError, match="write failed"):
        build_memory(history(), failed)
    assert not list(tmp_path.rglob("*.json"))


def test_invalid_cache_directory_fails_before_history_model_calls(tmp_path):
    (tmp_path / "cache").write_text("not a directory", encoding="utf-8")
    obj = backend(tmp_path, responses=[])
    with pytest.raises(OSError):
        build_memory(history(), obj)
    assert obj.controller.calls == [] and obj.embedder.calls == []


def test_refresh_and_atomic_save_keep_previous_cache_on_failure(tmp_path, monkeypatch):
    original = backend(tmp_path)
    build_memory(history(), original)
    path = next((tmp_path / "cache").glob("*.json"))
    old = path.read_bytes()
    failed = backend(tmp_path, mode="refresh", responses=[RuntimeError("write failed")])
    with pytest.raises(RuntimeError): build_memory(history(), failed)
    assert path.read_bytes() == old
    fresh = backend(tmp_path, mode="refresh")
    real_replace = cache.os.replace
    def fail(*args): raise OSError("disk failed")
    monkeypatch.setattr(cache.os, "replace", fail)
    with pytest.raises(OSError, match="disk failed"):
        build_memory(history(), fresh)
    assert path.read_bytes() == old and not list(path.parent.glob("*.tmp"))
    monkeypatch.setattr(cache.os, "replace", real_replace)
    fresh.save_cached_memory(extract_sessions(history()))
    assert fresh.cache_info["build_id"] != original.cache_info["build_id"]
    assert len(list(path.parent.glob("*.json"))) == 1


def test_cache_settings_follow_cli_over_dotenv_and_validate(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED")
    env = tmp_path / "test.env"
    env.write_text("AMEM_CACHE_MODE=reuse\nAMEM_CACHE_DIR=from-dotenv\n", encoding="utf-8")
    parser = ArgumentParser()
    add_model_arguments(parser)
    args = parser.parse_args(["--env-file", str(env), "--amem-cache-mode", "require"])
    with configured_environment(args) as sources:
        settings = AMemSettings.from_env()
        assert settings.cache_mode == "require" and settings.cache_dir == "from-dotenv"
        assert sources["AMEM_CACHE_MODE"] == "cli" and sources["AMEM_CACHE_DIR"] == ".env"
    with pytest.raises(ValueError): AMemSettings(cache_mode="invalid")


def test_adapter_saves_before_qa_and_reports_hit_even_when_answering_failed(tmp_path, monkeypatch):
    import openai
    from langmem_eval import registry
    from langmem_eval.adapter import run_method
    objects = []
    def factory(*args):
        obj = backend(tmp_path)
        objects.append(obj)
        return obj
    def fail(**kwargs): raise RuntimeError("answer unavailable")
    monkeypatch.setattr(registry, "create_backend", factory)
    monkeypatch.setattr(AMemBackend, "retrieve", lambda *args: ["Berlin"])
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: NS(chat=NS(completions=NS(create=fail))))
    first = run_method("amem", history(), "offline", False)
    assert first[0]["answer_status"] == "error"
    second = run_method("amem", history(), "offline", False)
    assert second[0]["answer_trace"]["method_config"]["memory_cache"]["status"] == "hit"
    assert objects[1].controller.calls == [] and objects[1].embedder.calls == []
