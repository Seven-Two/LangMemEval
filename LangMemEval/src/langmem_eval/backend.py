"""Optional real LangMem backend; dependencies are loaded only on construction."""
from __future__ import annotations

import json
from uuid import uuid4

from .benchmark import Session


class LangMemBackend:
    def __init__(self, model: str, *, embedding_model: str = "text-embedding-3-small",
                 embedding_dims: int = 1536, enable_deletes: bool = False,
                 query_limit: int = 5, instructions: str | None = None):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langgraph.store.memory import InMemoryStore
        from langmem import create_memory_store_manager

        self.namespace = ("langmem-eval", uuid4().hex)
        self.store = InMemoryStore(index={
            "dims": embedding_dims,
            "embed": OpenAIEmbeddings(model=embedding_model, dimensions=embedding_dims),
        })
        options = {} if instructions is None else {"instructions": instructions}
        self.manager = create_memory_store_manager(
            ChatOpenAI(model=model, temperature=0.1),
            namespace=self.namespace, store=self.store,
            enable_deletes=enable_deletes, query_limit=query_limit, **options,
        )

    def ingest(self, session: Session) -> None:
        # Synchronous barrier: all writes complete before evaluation starts.
        self.manager.invoke({"messages": session.messages})

    def retrieve(self, query: str, limit: int) -> list[str]:
        items = self.store.search(self.namespace, query=query, limit=limit)
        return [json.dumps(item.value, ensure_ascii=False) for item in items]
