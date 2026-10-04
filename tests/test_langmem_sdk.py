"""Real SDK/store tests with local fake embeddings; no network or API costs."""
import pytest

pytest.importorskip("langmem")

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langmem_eval.methods.langmem.backend import LangMemBackend
from langmem_eval.configuration import EmbeddingSettings


class LocalEmbedding(Embeddings):
    def encode(self, texts):
        return self.embed_documents(texts)

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [0.0] * 16
        for i, byte in enumerate(text.encode()):
            vector[i % 16] += byte
        return vector


def test_real_manager_store_and_isolation(monkeypatch):
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI",
                        lambda **kwargs: FakeListChatModel(responses=["unused"]))
    settings = EmbeddingSettings(embedding_dims=16)
    first = LangMemBackend("offline", embedding_settings=settings, embedder=LocalEmbedding())
    second = LangMemBackend("offline", embedding_settings=settings, embedder=LocalEmbedding())
    first.store.put(first.namespace, "fact", {"content": "Alice lives in Berlin"})
    assert "Berlin" in first.retrieve("Where does Alice live?", 1)[0]
    assert second.retrieve("Where does Alice live?", 1) == []
    assert first.manager is not None
