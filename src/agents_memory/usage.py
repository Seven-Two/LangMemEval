"""Best-effort usage accounting. Unobserved costs are unknown, not zero."""
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import threading
from time import perf_counter

from openai.resources.chat.completions import Completions, AsyncCompletions
from openai.resources.responses.responses import Responses, AsyncResponses
from openai.resources.embeddings import Embeddings, AsyncEmbeddings

PHASES = ("write", "retrieve", "answer", "judge", "unclassified")
_phase = ContextVar("memory_cost_phase", default="unclassified")
_inside = ContextVar("memory_cost_nested_sdk", default=False)
_lock = threading.Lock()
_events = []
_phase_seconds = defaultdict(float)
_originals = {}


@contextmanager
def phase(name):
    if name not in PHASES:
        raise ValueError(f"Unknown cost phase: {name}")
    parent = _phase.get()
    token = _phase.set(name)
    started = perf_counter()
    try:
        yield
    finally:
        if parent != name:
            with _lock:
                _phase_seconds[name] += perf_counter() - started
        _phase.reset(token)


def _begin(model, kind, provider="openai_sdk", stage=None):
    event = {"phase": stage or _phase.get(), "model": model or "unknown",
             "kind": kind, "provider": provider, "calls": 1, "failed_calls": 0,
             "prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0,
             "usage_reported": False, "reported_cost_usd": None}
    with _lock:
        _events.append(event)
    return event


def _usage(event, response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    with _lock:
        event["usage_reported"] = True
        if event["kind"] == "embedding":
            event["embedding_tokens"] = getattr(usage, "total_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
        else:
            event["prompt_tokens"] = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
            event["completion_tokens"] = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0


def _stream_usage(event, chunk):
    _usage(event, chunk)
    if getattr(chunk, "response", None) is not None:
        _usage(event, chunk.response)


class _Stream:
    def __init__(self, stream, event):
        self.stream, self.event = stream, event

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __iter__(self):
        try:
            for chunk in self.stream:
                _stream_usage(self.event, chunk)
                yield chunk
        except Exception:
            self.event["failed_calls"] = 1
            raise

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)


class _AsyncStream:
    def __init__(self, stream, event):
        self.stream, self.event = stream, event

    def __getattr__(self, name):
        return getattr(self.stream, name)

    async def __aiter__(self):
        try:
            async for chunk in self.stream:
                _stream_usage(self.event, chunk)
                yield chunk
        except Exception:
            self.event["failed_calls"] = 1
            raise

    async def __aenter__(self):
        await self.stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self.stream.__aexit__(*args)


def _wrapper(original, kind):
    if inspect.iscoroutinefunction(original):
        @wraps(original)
        async def wrapped(*args, **kwargs):
            if _inside.get():
                return await original(*args, **kwargs)
            event = _begin(kwargs.get("model"), kind)
            token = _inside.set(True)
            try:
                result = await original(*args, **kwargs)
                if kwargs.get("stream"):
                    return _AsyncStream(result, event)
                _usage(event, result)
                return result
            except Exception:
                event["failed_calls"] = 1
                raise
            finally:
                _inside.reset(token)
    else:
        @wraps(original)
        def wrapped(*args, **kwargs):
            if _inside.get():
                return original(*args, **kwargs)
            event = _begin(kwargs.get("model"), kind)
            token = _inside.set(True)
            try:
                result = original(*args, **kwargs)
                if kwargs.get("stream"):
                    return _Stream(result, event)
                _usage(event, result)
                return result
            except Exception:
                event["failed_calls"] = 1
                raise
            finally:
                _inside.reset(token)
    return wrapped


def start():
    """Observe common SDK calls without altering model requests."""
    for cls, methods, kind in (
        (Completions, ("create", "parse"), "llm"),
        (AsyncCompletions, ("create", "parse"), "llm"),
        (Responses, ("create", "parse"), "llm"),
        (AsyncResponses, ("create", "parse"), "llm"),
        (Embeddings, ("create",), "embedding"),
        (AsyncEmbeddings, ("create",), "embedding"),
    ):
        for name in methods:
            key = (cls, name)
            if key not in _originals and hasattr(cls, name):
                original = getattr(cls, name)
                _originals[key] = original
                setattr(cls, name, _wrapper(original, kind))


def record_external_usage(*, provider, model, kind="llm", stage=None,
                          prompt_tokens=None, completion_tokens=None,
                          embedding_tokens=None, cost_usd=None, failed=False):
    """Report calls not already observed by the SDK wrappers."""
    if stage is not None and stage not in PHASES:
        raise ValueError("Invalid stage")
    if kind not in {"llm", "embedding", "service"}:
        raise ValueError("Invalid usage kind")
    values = (prompt_tokens, completion_tokens, embedding_tokens, cost_usd)
    if any(v is not None and v < 0 for v in values):
        raise ValueError("Usage cannot be negative")
    event = _begin(model, kind, provider, stage)
    with _lock:
        event.update(prompt_tokens=prompt_tokens or 0, completion_tokens=completion_tokens or 0,
                     embedding_tokens=embedding_tokens or 0, reported_cost_usd=cost_usd,
                     usage_reported=any(v is not None for v in values[:3]), failed_calls=int(failed))


def _sum(events):
    keys = ("calls", "failed_calls", "prompt_tokens", "completion_tokens", "embedding_tokens")
    result = {key: sum(e[key] for e in events) for key in keys}
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"] + result["embedding_tokens"]
    result["calls_without_usage"] = sum(not e["usage_reported"] for e in events)
    costs = [e["reported_cost_usd"] for e in events if e["reported_cost_usd"] is not None]
    result["reported_cost_usd"] = sum(costs) if costs else None
    result["calls_without_price"] = sum(e["reported_cost_usd"] is None for e in events)
    return result


def get_stats():
    """Observed method usage, excluding judge calls."""
    with _lock:
        return _sum([e for e in _events if e["phase"] != "judge"])


def get_stats_by_model():
    with _lock:
        events = [e for e in _events if e["phase"] != "judge"]
        return {model: _sum([e for e in events if e["model"] == model])
                for model in sorted({e["model"] for e in events})}


def get_report():
    with _lock:
        return {
            "coverage": "partial",
            "scope": "Observed SDK calls plus adapter reports; not total system cost",
            "limitations": ["Other providers/local compute/remote services require explicit instrumentation",
                            "SDK-internal retries are not separate calls; missing usage is unknown, not zero",
                            "Plain worker threads need context propagation; otherwise phase is unclassified",
                            "USD totals contain reported prices only; no automatic price estimates",
                            "Phase times are scoped wall seconds; concurrent/nested scopes may overlap"],
            "method": _sum([e for e in _events if e["phase"] != "judge"]),
            "judge": _sum([e for e in _events if e["phase"] == "judge"]),
            "by_phase": {p: {**_sum([e for e in _events if e["phase"] == p]),
                             "wall_seconds": _phase_seconds[p]} for p in PHASES},
            "by_provider": {p: _sum([e for e in _events if e["provider"] == p])
                            for p in sorted({e["provider"] for e in _events})},
        }


def reset():
    with _lock:
        _events.clear()
        _phase_seconds.clear()
