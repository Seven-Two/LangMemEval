"""LangMem memory transformations with durable snapshots and cached embeddings.

The service commits a new snapshot only after an entire Add succeeds. No model
answer generation is exposed. Each request gets an isolated working store.
"""
import hashlib
import json
from threading import RLock

from langchain_core.embeddings import Embeddings

from .schemas import Evidence


class CachedEmbeddings(Embeddings):
    def __init__(self, delegate, connection, signature):
        self.delegate, self.connection, self.signature = delegate, connection, signature
        self.lock = RLock()

    def embed_documents(self, texts):
        with self.lock:
            return self._embed_documents(texts)

    def _embed_documents(self, texts):
        output = [None] * len(texts)
        missing = {}
        for i, text in enumerate(texts):
            key = hashlib.sha256((self.signature + "\0" + text).encode()).hexdigest()
            row = self.connection.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
            if row:
                output[i] = json.loads(row[0])
            else:
                missing.setdefault(key, (text, []))[1].append(i)
        if missing:
            vectors = self.delegate.embed_documents([v[0] for v in missing.values()])
            if len(vectors) != len(missing):
                raise ValueError("Embedding provider returned unexpected vector count")
            for (key, (_, indices)), vector in zip(missing.items(), vectors):
                self.connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?)",
                                        (key, json.dumps(vector)))
                for i in indices:
                    output[i] = vector
        return output

    def embed_query(self, text):
        return self.delegate.embed_query(text)


class AMLLangMemBackend:
    def __init__(self, settings, user_id, snapshot, connection):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langgraph.store.memory import InMemoryStore
        from langmem import create_memory_store_manager

        self.namespace = ("aml", user_id)
        embedding = OpenAIEmbeddings(
            model=settings.embedding_model, dimensions=settings.embedding_dims,
            api_key=settings.embedding_key, base_url=settings.embedding_url,
            check_embedding_ctx_length=False,
        )
        # Include user boundary; even embedding caches are isolated across users.
        signature = json.dumps([user_id, settings.version, settings.embedding_model,
                                settings.embedding_dims, settings.embedding_url])
        self.store = InMemoryStore(index={"dims": settings.embedding_dims,
            "embed": CachedEmbeddings(embedding, connection, signature)})
        for item in snapshot:
            self.store.put(tuple(item["namespace"]), item["key"], item["value"])
        self.manager = create_memory_store_manager(
            ChatOpenAI(model=settings.llm_model, api_key=settings.llm_key,
                       base_url=settings.llm_url, temperature=0.1),
            store=self.store, namespace=self.namespace, enable_deletes=False,
        )

    def add(self, request):
        # Preserve role, event time and session provenance as historical data.
        messages = [{"role": m.role, "content": json.dumps({
            "text": m.content, "timestamp_ms": m.timestamp,
            "session_id": request.session_id, "request_id": request.request_id,
            "message_index": i}, ensure_ascii=False)} for i, m in enumerate(request.messages)]
        self.manager.invoke({"messages": messages})

    def search(self, request):
        # Options are accepted but not used by this baseline to rewrite queries.
        items = self.store.search(self.namespace, query=request.query, limit=request.top_k)
        return [Evidence(id=item.key, content=json.dumps(item.value, ensure_ascii=False),
                         score=item.score) for item in items]

    def snapshot(self):
        records, offset = [], 0
        while True:
            batch = self.store.search(self.namespace, limit=100, offset=offset)
            records.extend({"namespace": list(x.namespace), "key": x.key, "value": x.value}
                           for x in batch)
            if len(batch) < 100:
                return records
            offset += len(batch)
