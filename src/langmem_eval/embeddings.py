"""Shared local/API embedding clients for registered memory methods."""
from functools import lru_cache
import numpy as np
from agents_memory.diagnostics import stage

from .configuration import EmbeddingSettings


@lru_cache(maxsize=2)
def _load_local_model(name, revision, device, local_files_only):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Local embeddings need: uv sync --locked --extra dev --extra amem") from None
    with stage("embedding.load", embedding_model=name, device=device, local_files_only=local_files_only):
        return SentenceTransformer(name, revision=revision, device=device, local_files_only=local_files_only)


class LocalEmbedder:
    def __init__(self, settings):
        self.settings = settings
        self.model = _load_local_model(settings.embedding_model, settings.embedding_revision,
                                       settings.device, settings.local_files_only)

    def encode(self, texts):
        from agents_memory.usage import record_external_usage
        failed = True
        try:
            with stage("embedding.encode", provider="local", items=len(texts), device=self.settings.device):
                result = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            failed = False
            return result
        finally:
            record_external_usage(provider="sentence_transformers_local", model=self.settings.embedding_model,
                                  kind="embedding", failed=failed)


class APIEmbedder:
    def __init__(self, settings: EmbeddingSettings, *, client=None):
        from openai import OpenAI
        self.settings = settings
        if client is None:
            if not settings.embedding_base_url or not settings.embedding_api_key:
                raise ValueError("Set EMBEDDING_BASE_URL and EMBEDDING_API_KEY for API embeddings")
            client = OpenAI(api_key=settings.embedding_api_key, base_url=settings.embedding_base_url)
        self.client = client

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.settings.embedding_dims or 0))
        vectors = []
        for offset in range(0, len(texts), self.settings.embedding_batch_size):
            chunk = texts[offset:offset + self.settings.embedding_batch_size]
            kwargs = {"dimensions": self.settings.embedding_dims} if self.settings.embedding_dims else {}
            with stage("embedding.api", embedding_model=self.settings.embedding_model,
                       batch_start=offset + 1, items=len(chunk), total=len(texts)):
                result = self.client.embeddings.create(model=self.settings.embedding_model, input=chunk,
                                                        encoding_format="float", **kwargs)
            ordered = sorted(result.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(chunk))):
                raise ValueError("Embedding service returned missing or duplicate vector indices")
            vectors.extend(item.embedding for item in ordered)
        return np.asarray(vectors, dtype=float)


def create_embedder(settings=None):
    settings = settings if settings is not None else EmbeddingSettings.from_env()
    return LocalEmbedder(settings) if settings.embedding_provider == "local" else APIEmbedder(settings)
