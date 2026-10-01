import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from langmem_eval.aml.app import create_app
from langmem_eval.aml.config import Settings
from langmem_eval.aml.schemas import AddRequest, Evidence, SearchRequest
from langmem_eval.aml.service import MemoryService


class FakeBackend:
    calls = 0
    fail = False

    def __init__(self, settings, user_id, snapshot, connection):
        self.records = snapshot
    def add(self, request):
        type(self).calls += 1
        self.records.append({"id": request.request_id, "content": request.messages[0].content})
        if self.fail:
            raise RuntimeError("private-provider-details")
    def snapshot(self):
        return self.records
    def search(self, request):
        return [Evidence(**x) for x in self.records[:request.top_k]]


@pytest.fixture
def setup(tmp_path):
    FakeBackend.calls, FakeBackend.fail = 0, False
    settings = Settings(api_key="test-key", database=str(tmp_path / "db.sqlite3"))
    service = MemoryService(settings, FakeBackend)
    return settings, service, TestClient(create_app(settings, service))


def payload(user="u1", request="r1"):
    return {"request_id": request, "user_id": user, "session_id": "s1",
            "messages": [{"role": "user", "content": "Alice lives in Berlin", "timestamp": 1}]}


AUTH = {"Authorization": "Bearer test-key"}


def test_contract_auth_idempotence_restart_isolation(setup):
    settings, service, client = setup
    assert client.get("/health").status_code == 200
    assert client.post("/add", json=payload()).status_code == 401
    for _ in range(2):
        response = client.post("/add", json=payload(), headers=AUTH)
        assert response.json() == {"success": True, "request_id": "r1", "user_id": "u1", "session_id": "s1"}
    assert FakeBackend.calls == 1
    changed = payload()
    changed["messages"][0]["content"] = "different"
    assert client.post("/add", json=changed, headers=AUTH).status_code == 409
    restarted = TestClient(create_app(settings, MemoryService(settings, FakeBackend)))
    query = {"query": "where?", "user_id": "u1", "top_k": 100, "options": ["Berlin", "Paris"]}
    found = restarted.post("/search", json=query, headers={"X-Api-Key": "test-key"})
    assert found.json()["data"][0]["id"] == "r1"
    query["user_id"] = "other"
    assert restarted.post("/search", json=query, headers=AUTH).json() == {"data": []}
    query["top_k"] = 0
    assert restarted.post("/search", json=query, headers=AUTH).status_code == 422


def test_failed_write_rolls_back_and_retry(setup):
    _, service, client = setup
    FakeBackend.fail = True
    response = client.post("/add", json=payload(), headers=AUTH)
    assert response.status_code == 503 and "private" not in response.text
    assert service.search(SearchRequest(query="q", user_id="u1", top_k=1)).data == []
    FakeBackend.fail = False
    assert client.post("/add", json=payload(), headers=AUTH).status_code == 200


def test_concurrent_duplicate_add(setup):
    _, service, _ = setup
    request = AddRequest(**payload())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: service.add(request), range(2)))
    assert results[0] == results[1]
    assert FakeBackend.calls == 1


def test_embedding_cache_is_durable(tmp_path):
    from langmem_eval.aml.backend import CachedEmbeddings
    class Embedding:
        calls = 0
        def embed_documents(self, texts):
            self.calls += 1
            return [[1.0, 0.0] for _ in texts]
    provider = Embedding()
    with sqlite3.connect(tmp_path / "cache.db") as db:
        db.execute("CREATE TABLE embeddings (key TEXT PRIMARY KEY, vector TEXT)")
        cache = CachedEmbeddings(provider, db, "user1")
        assert cache.embed_documents(["a", "a"]) == [[1.0, 0.0], [1.0, 0.0]]
        cache.embed_documents(["a"])
        assert provider.calls == 1
        CachedEmbeddings(provider, db, "user2").embed_documents(["a"])
        assert provider.calls == 2


def test_real_store_snapshot_recovery(tmp_path, monkeypatch):
    import langchain_openai
    import langmem
    from langchain_core.embeddings import Embeddings
    from langmem_eval.aml.backend import AMLLangMemBackend

    class LocalEmbedding(Embeddings):
        calls = 0
        def embed_documents(self, texts):
            type(self).calls += 1
            return [[1.0, 0.0] for _ in texts]
        def embed_query(self, text):
            return [1.0, 0.0]

    def manager(model, *, store, namespace, **kwargs):
        class Manager:
            def invoke(self, payload):
                # Exercise the cross-thread embedding path used by model managers.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(store.put, namespace, "stable-id",
                                    {"content": "Berlin"}).result()
        return Manager()

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: object())
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", lambda **kwargs: LocalEmbedding())
    monkeypatch.setattr(langmem, "create_memory_store_manager", manager)
    settings = Settings(api_key="test", database=str(tmp_path / "real.db"), embedding_dims=2)
    first = MemoryService(settings, AMLLangMemBackend)
    first.add(AddRequest(**payload()))
    assert LocalEmbedding.calls == 1
    restarted = MemoryService(settings, AMLLangMemBackend)
    result = restarted.search(SearchRequest(query="where", user_id="u1", top_k=100))
    assert result.data[0].id == "stable-id"
    assert "Berlin" in result.data[0].content
    assert LocalEmbedding.calls == 1
