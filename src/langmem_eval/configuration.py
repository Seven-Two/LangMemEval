"""Resolve evaluation configuration once: CLI > dotenv > process env > defaults.

The scoped environment bridges existing SDKs and native evaluators. Its previous
values are restored even when evaluation fails; no CLI overrides leak into the
next experiment in the same Python process. Service (AML_*) settings are separate.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path

from .model_api import validate_base_url


def boolean(value):
    if isinstance(value, bool):
        return value
    values = {"true": True, "1": True, "false": False, "0": False}
    if str(value).lower() not in values:
        raise ValueError("Boolean configuration must be true/false or 1/0")
    return values[str(value).lower()]


@dataclass(frozen=True)
class EmbeddingSettings:
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dims: int | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_batch_size: int = 10
    embedding_revision: str | None = None
    device: str = "cpu"
    local_files_only: bool = False

    def __post_init__(self):
        if self.embedding_provider not in {"local", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER must be local or openai")
        if not self.embedding_model.strip():
            raise ValueError("EMBEDDING_MODEL is required for API embeddings")
        for value in (self.embedding_batch_size,):
            if type(value) is not int or value < 1:
                raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        if self.embedding_dims is not None and (type(self.embedding_dims) is not int or self.embedding_dims < 1):
            raise ValueError("EMBEDDING_DIMS must be positive or auto")
        validate_base_url(self.embedding_base_url)

    @classmethod
    def from_env(cls):
        provider = os.getenv("EMBEDDING_PROVIDER", "local")
        model = os.getenv("EMBEDDING_MODEL",
                          "sentence-transformers/all-MiniLM-L6-v2" if provider == "local" else "")
        dims = os.getenv("EMBEDDING_DIMS")
        return cls(
            embedding_provider=provider, embedding_model=model,
            embedding_dims=int(dims) if dims and dims != "auto" else None,
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL") or None,
            embedding_api_key=os.getenv("EMBEDDING_API_KEY") or None,
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
            embedding_revision=os.getenv("EMBEDDING_REVISION") or None,
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            local_files_only=boolean(os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "false")),
        )

    def public_config(self):
        config = asdict(self)
        config["embedding_api_key"] = "<redacted>" if self.embedding_api_key else None
        return config


# CLI attribute -> canonical environment key. argparse defaults must remain None.
ENV_FLAGS = {
    "llm_model": "LLM_MODEL", "llm_api_key": "OPENAI_API_KEY",
    "llm_base_url": "OPENAI_BASE_URL", "llm_extra_body": "LLM_EXTRA_BODY",
    "embedding_provider": "EMBEDDING_PROVIDER", "embedding_model": "EMBEDDING_MODEL",
    "embedding_api_key": "EMBEDDING_API_KEY", "embedding_base_url": "EMBEDDING_BASE_URL",
    "embedding_dims": "EMBEDDING_DIMS", "embedding_batch_size": "EMBEDDING_BATCH_SIZE",
    "embedding_device": "EMBEDDING_DEVICE", "embedding_revision": "EMBEDDING_REVISION",
    "embedding_local_files_only": "EMBEDDING_LOCAL_FILES_ONLY",
    "protocol": "EVAL_PROTOCOL", "context_tokens": "EVAL_CONTEXT_TOKENS",
    "tokenizer": "EVAL_TOKENIZER", "top_k": "EVAL_TOP_K",
    "max_output_tokens": "EVAL_MAX_OUTPUT_TOKENS", "answer_temperature": "EVAL_TEMPERATURE",
    "answer_style": "EVAL_ANSWER_STYLE", "empty_context": "EVAL_EMPTY_CONTEXT",
    "abstention_text": "EVAL_ABSTENTION_TEXT",
    "benchmark": "EVAL_BENCHMARK", "split": "EVAL_SPLIT", "systems": "EVAL_SYSTEMS",
    "num_samples": "EVAL_NUM_SAMPLES", "skip_judge": "EVAL_SKIP_JUDGE",
    "data_file": "EVAL_DATA_FILE", "output_dir": "EVAL_RESULTS_DIR",
    "log_mode": "EVAL_LOG_MODE",
    "judge_model": "JUDGE_MODEL", "longmemeval_judge_model": "LONGMEMEVAL_JUDGE_MODEL",
}


def environment_flags():
    """Resolve registered method bindings without central method-name branches."""
    from .registry import discover_methods
    flags = dict(ENV_FLAGS)
    for name, spec in discover_methods().items():
        for option in spec.options:
            if option.dest in flags or option.env in flags.values():
                raise ValueError(f"Configuration binding collision in method {name}: {option.flag}")
            flags[option.dest] = option.env
    return flags


def add_model_arguments(parser):
    import argparse
    parser.add_argument("--env-file", help="Dotenv file, default ./.env; CLI values override it")
    parser.add_argument("--show-config", action="store_true", help="Print resolved settings without model/data calls")
    parser.add_argument("--log-mode", choices=("concise", "full"),
                        help="Console detail: concise (default) or full; log file always retains full diagnostics")
    group = parser.add_argument_group("Shared chat and embedding configuration")
    for flag in ("llm-api-key", "llm-base-url", "llm-extra-body", "embedding-model",
                 "embedding-api-key", "embedding-base-url", "embedding-revision",
                 "embedding-device", "judge-model", "longmemeval-judge-model"):
        group.add_argument("--" + flag)
    group.add_argument("--embedding-provider", choices=("local", "openai"))
    group.add_argument("--embedding-dims", help="Positive integer; auto omits the API dimensions parameter")
    group.add_argument("--embedding-batch-size", type=int)
    group.add_argument("--embedding-local-files-only", action=argparse.BooleanOptionalAction, default=None)
    from .registry import discover_methods
    environment_flags()  # fail early on ambiguous CLI/environment ownership
    for name, spec in sorted(discover_methods().items()):
        if spec.options:
            group = parser.add_argument_group(f"{name} method parameters")
            for option in spec.options:
                group.add_argument("--" + option.flag, type=option.value_type,
                                   choices=option.choices, help=option.help, default=None)


@contextmanager
def configured_environment(args):
    from dotenv import dotenv_values
    path = Path(args.env_file) if args.env_file else Path.cwd() / ".env"
    if args.env_file and not path.is_file():
        raise ValueError("The requested --env-file does not exist")
    disabled = os.getenv("PYTHON_DOTENV_DISABLED", "").lower() in {"1", "true", "yes", "t", "y"}
    # No interpolation: avoid unexpected cross-source substitution of credentials.
    file_values = dotenv_values(path, interpolate=False) if path.is_file() and not disabled else {}
    flags = environment_flags()
    keys = set(flags.values()) | {"EVAL_DATA_DIR", "EVAL_MODELS_DIR"}
    values = {key: os.environ[key] for key in keys if key in os.environ}
    values.update({key: value for key, value in file_values.items() if key in keys and value is not None})
    sources = {key: ".env" if key in file_values else "environment" for key in values}
    for attr, key in flags.items():
        value = getattr(args, attr, None)
        if value is not None:
            values[key] = str(value).lower() if isinstance(value, bool) else str(value)
            sources[key] = "cli"
    old_keys = set(os.environ) | set(file_values)
    removed = sorted(key for key in old_keys if key.startswith("AMEM_EMBEDDING_")
                     or key in {"AMEM_DEVICE", "AMEM_LOCAL_FILES_ONLY"})
    if removed:
        raise ValueError("Rename old A-Mem embedding settings to EMBEDDING_* (AMEM_DEVICE -> "
                         "EMBEDDING_DEVICE): " + ", ".join(removed))
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        defaults = {"benchmark": "locomo", "systems": "all", "num_samples": 10, "skip_judge": False,
                    "log_mode": "concise"}
        for attr in ("benchmark", "split", "systems", "num_samples", "skip_judge", "data_file", "output_dir",
                     "llm_model", "log_mode"):
            value = values.get(ENV_FLAGS[attr], defaults.get(attr))
            if attr == "num_samples":
                value = int(value)
            elif attr == "skip_judge":
                value = boolean(value)
            elif attr == "llm_model":
                value = value if value is not None else "gpt-4.1"
                if not value.strip():
                    raise ValueError("LLM_MODEL must not be empty")
            elif attr == "log_mode" and value not in {"concise", "full"}:
                raise ValueError("EVAL_LOG_MODE must be concise or full")
            setattr(args, attr, value)
        validate_base_url(os.getenv("OPENAI_BASE_URL"))
        yield sources
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def public_model_config(model):
    from .model_api import llm_extra_body
    extra = llm_extra_body()
    return {
        "llm_model": model,
        "llm_base_url": os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        "llm_api_key": "<redacted>" if os.getenv("OPENAI_API_KEY") else None,
        "llm_extra_body": {k: v for k, v in extra.items()
                           if k in {"enable_thinking", "top_k", "repetition_penalty", "reasoning_effort"}},
        "embedding": EmbeddingSettings.from_env().public_config(),
    }
