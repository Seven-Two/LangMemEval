"""Benchmark boundary: only historical sessions reach the memory backend."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from .interfaces import MemoryBackend, Session  # Re-export for existing callers.


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


def build_memory(conv: dict, backend: MemoryBackend) -> None:
    for session in extract_sessions(conv):
        backend.ingest(session)


@lru_cache(maxsize=8)
def token_encoding(name: str):
    import tiktoken
    return tiktoken.get_encoding(name)


@dataclass(frozen=True)
class ContextSelection:
    context: str
    token_count: int
    tokenizer: str
    budget: int
    selected_indices: list[int]
    dropped_indices: list[int]
    candidate_count: int
    selection_policy: str = "ranked_whole_records_skip_nonfitting_v1"

    def to_dict(self):
        return asdict(self)


def select_context(records: list[str], *, max_tokens: int,
                   tokenizer: str = "cl100k_base") -> ContextSelection:
    """Greedy rank-order packing; never truncate a record, count full serialization."""
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative")
    encoding = token_encoding(tokenizer)
    selected, dropped, blocks = [], [], []
    context, count = "", 0
    for index, record in enumerate(records):
        if not isinstance(record, str):
            raise TypeError("Memory records must be strings")
        if not record.strip():
            dropped.append(index)
            continue
        proposed = "\n\n".join([*blocks, record])
        size = len(encoding.encode(proposed, disallowed_special=()))
        if size <= max_tokens:
            blocks.append(record)
            selected.append(index)
            context, count = proposed, size
        else:
            dropped.append(index)
    return ContextSelection(context, count, tokenizer, max_tokens, selected, dropped, len(records))


def memory_context(backend: MemoryBackend, query: str, *, top_k: int,
                   max_tokens: int, tokenizer: str = "cl100k_base") -> str:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    return select_context(backend.retrieve(query, top_k)[:top_k],
                          max_tokens=max_tokens, tokenizer=tokenizer).context
