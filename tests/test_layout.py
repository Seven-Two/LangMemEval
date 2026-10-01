"""Relocation must not put datasets or outputs inside installed source packages."""
import json
from pathlib import Path

from agents_memory.paths import data_dir, models_dir, results_dir


def test_runtime_paths_follow_workdir_and_explicit_overrides(tmp_path, monkeypatch):
    for name in ("EVAL_DATA_DIR", "EVAL_MODELS_DIR", "EVAL_RESULTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    assert data_dir() == tmp_path / "data"
    assert models_dir() == tmp_path / "models"
    assert results_dir() == tmp_path / "results"
    monkeypatch.setenv("EVAL_DATA_DIR", "datasets")
    monkeypatch.setenv("EVAL_MODELS_DIR", str(tmp_path / "weights"))
    assert data_dir() == tmp_path / "datasets"
    assert models_dir() == tmp_path / "weights"


def test_loaders_use_relocated_cache_without_network(tmp_path, monkeypatch):
    from agents_memory import locomo
    from agents_memory.benchmarks import longmemeval
    cache = tmp_path / "custom-data"
    (cache / "longmemeval").mkdir(parents=True)
    expected = [{"sample_id": "offline"}]
    (cache / "locomo10.json").write_text(json.dumps(expected), encoding="utf-8")
    (cache / "longmemeval/longmemeval_oracle.json").write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setenv("EVAL_DATA_DIR", str(cache))
    def no_network(*args, **kwargs):
        raise AssertionError("Cache hit must not download")
    monkeypatch.setattr(locomo.httpx, "get", no_network)
    monkeypatch.setattr(longmemeval, "hf_hub_download", no_network)
    assert locomo.download_locomo() == expected
    assert longmemeval._download_split("oracle") == expected


def test_single_source_layout():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/agents_memory/runner.py").is_file()
    assert (root / "src/langmem_eval/cli.py").is_file()
    assert not (root / "MemEval").exists()
    assert not (root / "LangMemEval").exists()
    assert (root / "third_party/memeval/LICENSE").is_file()
    assert (root / "third_party/memeval/NOTICE").is_file()
