"""Mock transport checks: no paid models, stage attribution includes failures."""
import asyncio
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from agents_memory import usage


@pytest.fixture(autouse=True)
def reset_usage():
    usage.reset()
    yield
    usage.reset()


def test_sdk_chat_embedding_judge_and_failed_call():
    usage.start()
    usage.start()
    def handle(request):
        if request.url.path.endswith("embeddings"):
            return httpx.Response(200, json={"object": "list", "model": "embedding",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 7, "total_tokens": 7}})
        return httpx.Response(200, json={"id": "chat", "object": "chat.completion", "created": 0,
            "model": "offline", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "yes"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
    with openai.OpenAI(api_key="offline", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        with usage.phase("write"):
            client.chat.completions.create(model="offline", messages=[])
        with usage.phase("retrieve"):
            client.embeddings.create(model="embedding", input="query")
        with usage.phase("judge"):
            client.chat.completions.create(model="offline", messages=[])
    report = usage.get_report()
    assert report["method"]["calls"] == 2
    assert report["method"]["total_tokens"] == 19
    assert report["judge"]["total_tokens"] == 12
    assert report["by_phase"]["retrieve"]["embedding_tokens"] == 7
    assert usage.get_stats_by_model()["offline"]["total_tokens"] == 12
    assert report["method"]["reported_cost_usd"] is None
    def fail(**kw):
        raise RuntimeError("failed")
    with usage.phase("answer"), pytest.raises(RuntimeError):
        usage._wrapper(fail, "llm")(model="offline")
    assert usage.get_report()["by_phase"]["answer"]["failed_calls"] == 1
    assert usage.get_stats()["calls_without_usage"] == 1


def test_stream_keeps_original_phase_and_missing_usage_unknown():
    call = usage._wrapper(lambda **kw: iter([
        NS(usage=None), NS(usage=NS(prompt_tokens=4, completion_tokens=3))]), "llm")
    with usage.phase("answer"):
        stream = call(model="offline", stream=True)
    with usage.phase("judge"):
        list(stream)
    report = usage.get_report()
    assert report["by_phase"]["answer"]["total_tokens"] == 7
    assert report["judge"]["calls"] == 0
    usage.record_external_usage(provider="remote", model="unknown", kind="service")
    assert usage.get_stats()["calls_without_usage"] == 1


def test_async_context_isolation_and_nested_parse_not_double_counted():
    async def raw(**kw):
        await asyncio.sleep(0)
        return NS(usage=NS(input_tokens=2, output_tokens=1))
    inner = usage._wrapper(raw, "llm")
    async def parse(**kw):
        return await inner(**kw)
    outer = usage._wrapper(parse, "llm")
    async def task(stage):
        with usage.phase(stage):
            await outer(model="offline")
    async def run():
        await asyncio.gather(task("answer"), task("judge"))
    asyncio.run(run())
    assert usage.get_report()["method"]["calls"] == 1
    assert usage.get_report()["judge"]["calls"] == 1


def test_external_report_and_judge_error_are_explicit(monkeypatch):
    from agents_memory import evaluation
    usage.record_external_usage(provider="local", model="small", stage="write",
                                prompt_tokens=10, completion_tokens=5, cost_usd=.01)
    assert usage.get_report()["by_phase"]["write"]["reported_cost_usd"] == .01
    class Client:
        def __init__(self, **kw):
            self.chat = NS(completions=self)
        def create(self, **kw):
            usage.record_external_usage(provider="fake", model="judge", prompt_tokens=1)
            return NS(choices=[NS(message=NS(content="maybe"))])
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setattr(evaluation, "OpenAI", Client)
    result = evaluation.evaluate_longmemeval("q", "a", "b")
    assert result["longmemeval_correct"] is None
    assert result["judge_status"] == "error"
    assert usage.get_report()["judge"]["calls"] == 1
    assert usage.get_report()["method"]["calls"] == 1
