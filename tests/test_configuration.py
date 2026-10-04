"""CLI/dotenv precedence and shared provider routing, with no real inference."""
from argparse import Namespace
import json
import os

import pytest

from langmem_eval.configuration import ENV_FLAGS, EmbeddingSettings, configured_environment


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    for key in list(os.environ):
        if key in ENV_FLAGS.values() or key.startswith("AMEM_"):
            monkeypatch.delenv(key)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)


def test_cli_then_dotenv_then_environment_and_restore(tmp_path, monkeypatch):
    env = tmp_path / "experiment.env"
    env.write_text("LLM_MODEL=file-model\nEMBEDDING_PROVIDER=local\nEMBEDDING_MODEL=file-embedding\n"
                   "EVAL_TEMPERATURE=0.8\nEVAL_SKIP_JUDGE=true\nEMBEDDING_LOCAL_FILES_ONLY=true\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "shell-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "shell-embedding")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")
    args = Namespace(env_file=str(env), llm_model="cli-model", answer_temperature=0.0,
                     skip_judge=False, embedding_local_files_only=False)
    with configured_environment(args) as sources:
        assert args.llm_model == "cli-model"
        assert EmbeddingSettings.from_env().embedding_model == "file-embedding"
        assert not EmbeddingSettings.from_env().local_files_only
        assert os.environ["EVAL_TEMPERATURE"] == "0.0"
        assert not args.skip_judge
        assert os.environ["OPENAI_API_KEY"] == "shell-key"
        assert sources["LLM_MODEL"] == "cli" and sources["EMBEDDING_MODEL"] == ".env"
    assert os.environ["LLM_MODEL"] == "shell-model"
    assert os.environ["EMBEDDING_MODEL"] == "shell-embedding"
    assert "EVAL_TEMPERATURE" not in os.environ


def test_missing_file_defaults_and_failure_restore(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with configured_environment(Namespace(env_file=None)):
        assert EmbeddingSettings.from_env().embedding_provider == "local"
    with pytest.raises(ValueError, match="does not exist"):
        with configured_environment(Namespace(env_file="missing.env")):
            pass
    monkeypatch.setenv("OPENAI_BASE_URL", "https://original.invalid/v1")
    with pytest.raises(ValueError, match="base URL"):
        with configured_environment(Namespace(env_file=None, llm_base_url="https://host/v1/chat/completions")):
            pass
    assert os.environ["OPENAI_BASE_URL"] == "https://original.invalid/v1"


def test_reject_old_embedding_names_with_actionable_message(tmp_path):
    env = tmp_path / "old.env"
    env.write_text("AMEM_EMBEDDING_MODEL=old\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Rename old A-Mem"):
        with configured_environment(Namespace(env_file=str(env))):
            pass


def test_show_config_has_no_inference_and_redacts_secrets(tmp_path, monkeypatch, capsys):
    from langmem_eval.cli import main
    import langmem_eval.registry as registry
    env = tmp_path / "experiment.env"
    env.write_text("LLM_MODEL=from-file\nOPENAI_API_KEY=private-chat-key\n"
                   "EMBEDDING_PROVIDER=openai\nEMBEDDING_MODEL=custom-embed\n"
                   "EMBEDDING_API_KEY=private-embedding-key\n"
                   "EMBEDDING_BASE_URL=https://emb.invalid/v1\nEMBEDDING_DIMS=16\n", encoding="utf-8")
    def forbidden(*args):
        raise AssertionError("Config inspection must not construct a backend")
    monkeypatch.setattr(registry, "create_backend", forbidden)
    monkeypatch.setattr("sys.argv", ["eval", "--systems", "amem,langmem", "--env-file", str(env),
                                    "--llm-model", "from-cli", "--show-config"])
    main()
    raw = capsys.readouterr().out
    report = json.loads(raw)
    assert report["models"]["llm_model"] == "from-cli"
    assert report["models"]["embedding"]["embedding_model"] == "custom-embed"
    assert "private-chat-key" not in raw and "private-embedding-key" not in raw
    assert report["sources"]["LLM_MODEL"] == "cli"
    assert not list(tmp_path.glob("results*"))


def test_both_backends_use_shared_embedding_and_chat_options(tmp_path, monkeypatch):
    import numpy as np
    import langchain_openai
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langmem_eval.methods.amem.backend import AMemBackend
    from langmem_eval.methods.langmem.backend import LangMemBackend
    class Encoder:
        def encode(self, texts):
            return np.array([[1., 0.] for text in texts])
    captured = []
    def chat(**kwargs):
        captured.append(kwargs)
        return FakeListChatModel(responses=["unused"])
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", chat)
    monkeypatch.chdir(tmp_path)
    args = Namespace(env_file=None, llm_model="chat-test", embedding_provider="openai",
                     embedding_model="shared-test", embedding_dims="2",
                     embedding_base_url="https://embedding.invalid/v1",
                     embedding_api_key="offline", llm_extra_body='{"enable_thinking":false}')
    with configured_environment(args):
        amem = AMemBackend(args.llm_model, controller=object(), embedder=Encoder())
        langmem = LangMemBackend(args.llm_model, embedder=Encoder())
        assert amem.settings.embedding_model == langmem.embedding_settings.embedding_model == "shared-test"
        assert amem.settings.embedding_base_url == langmem.embedding_settings.embedding_base_url
        langmem.store.put(langmem.namespace, "fact", {"content": "Berlin"})
        assert "Berlin" in langmem.retrieve("Where?", 1)[0]
        assert captured[0]["model"] == "chat-test"
        assert captured[0]["extra_body"] == {"enable_thinking": False}


def test_api_embedding_dims_auto_is_distinct_from_invalid_zero(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIMS", "auto")
    assert EmbeddingSettings.from_env().embedding_dims is None
    monkeypatch.setenv("EMBEDDING_DIMS", "0")
    with pytest.raises(ValueError):
        EmbeddingSettings.from_env()
    monkeypatch.delenv("EMBEDDING_DIMS")
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        EmbeddingSettings.from_env()


def test_native_adapters_cannot_silently_ignore_embedding_cli(tmp_path, monkeypatch):
    from agents_memory import runner
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "SYSTEMS", {"native": {"fn": lambda *args: [],
                        "architecture": "test", "infrastructure": "none"}})
    monkeypatch.setattr("sys.argv", ["eval", "--systems", "native", "--embedding-model", "test", "--show-config"])
    with pytest.raises(SystemExit, match="registered methods only"):
        runner.main()
