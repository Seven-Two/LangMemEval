"""Minimal contracts shared by every method and the benchmark adapter.

Implement these by structure: no inheritance or framework-specific base class is
required. Each factory creates independent state for one evaluation conversation.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Session:
    """Historical session only; messages contain normalized JSON speaker/text data."""

    id: str
    date: str
    messages: list[dict[str, str]]


class MemoryBackend(Protocol):
    """Synchronous writes and ranked, complete evidence records for retrieval.

    Methods receive history and questions, never evaluation answers. Retrieval
    returns at most limit records; the adapter owns token budgets and answering.
    Optional describe() and last_retrieval expose JSON-serializable audit data.
    """

    def ingest(self, session: Session) -> None: ...

    def retrieve(self, query: str, limit: int) -> list[str]: ...
