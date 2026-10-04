"""Optional real LangMem backend; dependencies are loaded only on construction."""
from __future__ import annotations

import json
from uuid import uuid4

from ...interfaces import Session
from ...configuration import EmbeddingSettings
from ...embeddings import create_embedder
from ...model_api import llm_extra_body


class LangMemBackend:
    def __init__(self, model: str, *, embedding_settings=None, embedder=None,
                 enable_deletes: bool = False, query_limit: int = 5, instructions: str | None = None):
        from langchain_openai import ChatOpenAI
        from langgraph.store.memory import InMemoryStore
        from langmem import create_memory_store_manager

        self.embedding_settings = embedding_settings or EmbeddingSettings.from_env()
        if self.embedding_settings.embedding_provider == "openai" and self.embedding_settings.embedding_dims is None:
            raise ValueError("LangMem's vector store requires --embedding-dims / EMBEDDING_DIMS for API embeddings")
        self.embedder = embedder if embedder is not None else create_embedder(self.embedding_settings)
        dims = self.embedding_settings.embedding_dims
        if dims is None:
            dims = self.embedder.model.get_sentence_embedding_dimension()
        if not dims or dims < 1:
            raise ValueError("Cannot determine LangMem embedding dimensions; set EMBEDDING_DIMS")
        self.embedding_dims = dims
        self.namespace = ("langmem-eval", uuid4().hex)
        self.store = InMemoryStore(index={
            "dims": dims,
            "embed": self._embed,
        })
        options = {} if instructions is None else {"instructions": instructions}
        self.manager = create_memory_store_manager(
            ChatOpenAI(model=model, temperature=0.1, extra_body=llm_extra_body() or None),
            namespace=self.namespace, store=self.store,
            enable_deletes=enable_deletes, query_limit=query_limit, **options,
        )

    def _embed(self, texts):
        vectors = self.embedder.encode(texts)
        if len(vectors) != len(texts) or any(len(vector) != self.embedding_dims for vector in vectors):
            raise ValueError("Embedding dimensions do not match LangMem's vector store")
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors

    def describe(self):
        return {"implementation": "langmem_shared_embedding_v1",
                "embedding": self.embedding_settings.public_config(),
                "embedding_dimensions_actual": self.embedding_dims}

    def ingest(self, session: Session) -> None:
        # Synchronous barrier: all writes complete before evaluation starts.
        self.manager.invoke({"messages": session.messages})

    def retrieve(self, query: str, limit: int) -> list[str]:
        items = self.store.search(self.namespace, query=query, limit=limit)
        return [json.dumps(item.value, ensure_ascii=False) for item in items]
