import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_key: str
    database: str = "data/aml.sqlite3"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-v4"
    embedding_dims: int = 1024
    llm_key: str | None = None
    llm_url: str | None = None
    embedding_key: str | None = None
    embedding_url: str | None = None
    version: str = "langmem-aml-v1"
    method: str = "langmem"

    def __post_init__(self):
        if not self.api_key.strip():
            raise ValueError("AML_API_KEY must be set; anonymous API is disabled")
        if self.embedding_dims < 1:
            raise ValueError("Embedding dimensions must be positive")

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv("AML_API_KEY", ""),
            database=os.getenv("AML_DATABASE", "data/aml.sqlite3"),
            llm_model=os.getenv("AML_LLM_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("AML_EMBEDDING_MODEL", "text-embedding-v4"),
            embedding_dims=int(os.getenv("AML_EMBEDDING_DIMS", "1024")),
            llm_key=os.getenv("AML_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            llm_url=os.getenv("AML_LLM_BASE_URL"),
            embedding_key=os.getenv("AML_EMBEDDING_API_KEY"),
            embedding_url=os.getenv("AML_EMBEDDING_BASE_URL"),
            version=os.getenv("AML_SYSTEM_VERSION", "langmem-aml-v1"),
            method=os.getenv("AML_METHOD", "langmem"),
        )
