"""Benchmark boundary: only historical sessions reach the memory backend."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Session:
    id: str
    date: str
    messages: list[dict[str, str]]


def extract_sessions(conv: dict) -> list[Session]:
    history = conv["conversation"]
    keys = sorted(
        (k for k in history if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    sessions = []
    for key in keys:
        date = str(history.get(f"{key}_date_time", "unknown"))
        messages = []
        for i, turn in enumerate(history[key]):
            # Both participants are quoted historical speakers, not the model.
            payload = {"speaker": turn["speaker"], "text": turn["text"],
                       "date": date, "source_id": str(turn.get("dia_id", f"{key}:{i}"))}
            messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
        sessions.append(Session(key, date, messages))
    if not sessions:
        raise ValueError("No session_N history found; provide MemEval-normalized data")
    return sessions


class MemoryBackend(Protocol):
    def ingest(self, session: Session) -> None: ...
    def retrieve(self, query: str, limit: int) -> list[str]: ...


def build_memory(conv: dict, backend: MemoryBackend) -> None:
    for session in extract_sessions(conv):
        backend.ingest(session)


def memory_context(backend: MemoryBackend, query: str, *, top_k: int,
                   max_chars: int) -> str:
    if top_k < 1 or max_chars < 1:
        raise ValueError("top_k and max_chars must be positive")
    # Explicit character budget, not a claimed token budget.
    return "\n".join(backend.retrieve(query, top_k))[:max_chars]
