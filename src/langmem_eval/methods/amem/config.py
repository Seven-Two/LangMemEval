"""A-Mem experiment settings and sanitized provenance configuration."""
from dataclasses import asdict, dataclass, field
import math
import os

from ...model_api import llm_extra_body
from ...configuration import EmbeddingSettings


@dataclass(frozen=True)
class AMemSettings(EmbeddingSettings):
    evolution_threshold: int = 100
    neighbor_k: int = 5
    response_format: str = "json_schema"
    temperature: float = 0.7
    max_output_tokens: int = 1000
    extra_body: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.response_format not in {"json_schema", "json_object", "prompt"}:
            raise ValueError("AMEM_RESPONSE_FORMAT must be json_schema, json_object or prompt")
        for n in (self.evolution_threshold, self.neighbor_k, self.max_output_tokens):
            if type(n) is not int or n < 1:
                raise ValueError("A-Mem count/budget settings must be positive integers")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("AMEM_TEMPERATURE must be between 0 and 2")

    @classmethod
    def from_env(cls):
        return cls(
            **asdict(EmbeddingSettings.from_env()),
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
