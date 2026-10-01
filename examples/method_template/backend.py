"""Runnable interface example, not a reproduction of a research baseline.

Replace the recent-turn selection with your algorithm. Keep the method's model
clients, prompts and configuration in sibling files as the method grows.
"""
from langmem_eval.interfaces import Session


class MyMethodBackend:
    def __init__(self, model: str):
        self.model = model
        self.records: list[str] = []

    def ingest(self, session: Session) -> None:
        # Normalized JSON already preserves speaker, date, text and source ID.
        self.records.extend(message["content"] for message in session.messages)

    def retrieve(self, query: str, limit: int) -> list[str]:
        if limit < 1:
            raise ValueError("Retrieval limit must be positive")
        # Deliberately simple example: most recent first, complete evidence only.
        return list(reversed(self.records[-limit:]))

    def describe(self) -> dict:
        return {"implementation": "recent_turns_example_v1", "note_count": len(self.records)}
