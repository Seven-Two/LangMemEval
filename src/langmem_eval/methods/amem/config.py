"""A-Mem experiment settings and sanitized provenance configuration."""
from dataclasses import asdict, dataclass, field
import math
import os

from ...model_api import llm_extra_body, validate_base_url


@dataclass(frozen=True)
class AMemSettings:
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dims: int | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_batch_size: int = 10
    embedding_revision: str | None = None
    device: str = "cpu"
    local_files_only: bool = False
    evolution_threshold: int = 100
    neighbor_k: int = 5
    response_format: str = "json_schema"
    temperature: float = 0.7
    max_output_tokens: int = 1000
    extra_body: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.embedding_provider not in {"local", "openai"}:
            raise ValueError("AMEM_EMBEDDING_PROVIDER must be local or openai")
        if self.response_format not in {"json_schema", "json_object", "prompt"}:
            raise ValueError("AMEM_RESPONSE_FORMAT must be json_schema, json_object or prompt")
        if not self.embedding_model.strip():
            raise ValueError("AMEM_EMBEDDING_MODEL is required")
        for n in (self.embedding_batch_size, self.evolution_threshold, self.neighbor_k, self.max_output_tokens):
            if type(n) is not int or n < 1:
                raise ValueError("A-Mem count/budget settings must be positive integers")
        if self.embedding_dims is not None and (type(self.embedding_dims) is not int or self.embedding_dims < 1):
            raise ValueError("AMEM_EMBEDDING_DIMS must be positive")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("AMEM_TEMPERATURE must be between 0 and 2")
        validate_base_url(self.embedding_base_url)

    @classmethod
    def from_env(cls):
        def flag(name):
            value = os.getenv(name, "false").lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError(f"{name} must be true or false")
            return value in {"true", "1"}
        provider = os.getenv("AMEM_EMBEDDING_PROVIDER", "local")
        model = os.getenv("AMEM_EMBEDDING_MODEL") or (
            "sentence-transformers/all-MiniLM-L6-v2" if provider == "local" else "")
        dims = os.getenv("AMEM_EMBEDDING_DIMS")
        return cls(
            embedding_provider=provider, embedding_model=model,
            embedding_dims=int(dims) if dims else None,
            embedding_base_url=os.getenv("AMEM_EMBEDDING_BASE_URL") or None,
            embedding_api_key=os.getenv("AMEM_EMBEDDING_API_KEY") or None,
            embedding_batch_size=int(os.getenv("AMEM_EMBEDDING_BATCH_SIZE", "10")),
            embedding_revision=os.getenv("AMEM_EMBEDDING_REVISION") or None,
            device=os.getenv("AMEM_DEVICE", "cpu"), local_files_only=flag("AMEM_LOCAL_FILES_ONLY"),
            evolution_threshold=int(os.getenv("AMEM_EVOLUTION_THRESHOLD", "100")),
            neighbor_k=int(os.getenv("AMEM_NEIGHBOR_K", "5")),
            response_format=os.getenv("AMEM_RESPONSE_FORMAT", "json_schema"),
            temperature=float(os.getenv("AMEM_TEMPERATURE", "0.7")),
            max_output_tokens=int(os.getenv("AMEM_MAX_OUTPUT_TOKENS", "1000")),
            extra_body=llm_extra_body(),
        )

    def public_config(self):
        result = asdict(self)
        result.pop("embedding_api_key")
        # Extra payloads can contain private service options: record known sampler options only.
        result["extra_body"] = {k: v for k, v in self.extra_body.items()
                                if k in {"enable_thinking", "top_k", "repetition_penalty", "reasoning_effort"}}
        return result
