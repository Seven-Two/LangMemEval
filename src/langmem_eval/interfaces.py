"""Minimal contracts shared by every method and the benchmark adapter.

Implement these by structure: no inheritance or framework-specific base class is
required. Each factory creates independent state for one evaluation conversation.
"""
from dataclasses import dataclass
import json
from typing import Protocol


@dataclass(frozen=True)
class HistoricalTurn:
    """A quoted historical utterance, not an instruction from the current user."""

    speaker: str
    text: str
    date: str
    source_id: str


@dataclass(frozen=True)
class Session:
    """Historical session only; messages contain normalized JSON speaker/text data."""

    id: str
    date: str
    messages: list[dict[str, str]]

    def turns(self):
        """Read typed turns without duplicating JSON decoding in each new method."""
        for index, message in enumerate(self.messages):
            try:
                value = json.loads(message["content"])
            except (KeyError, TypeError, json.JSONDecodeError):
                raise ValueError("History must contain normalized JSON messages") from None
            if not isinstance(value, dict) or not all(isinstance(value.get(k), str) for k in ("speaker", "text")):
                raise ValueError("History must contain string speaker and text")
            yield HistoricalTurn(value["speaker"], value["text"], str(value.get("date", self.date)),
                                 str(value.get("source_id", f"{self.id}:{index}")))


class MemoryBackend(Protocol):
    """Synchronous writes and ranked, complete evidence records for retrieval.

    Methods receive history and questions, never evaluation answers. Retrieval
    returns at most limit records; the adapter owns token budgets and answering.
    Optional describe() and last_retrieval expose JSON-serializable audit data.
    Optional close() releases owned resources after success or failure. Injected
    resources belong to the caller. Instances are conversation-local, not shared.
    """

    def ingest(self, session: Session) -> None: ...

    def retrieve(self, query: str, limit: int) -> list[str]: ...
