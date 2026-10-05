"""Runnable interface example, not a reproduction of a research baseline.

Replace the recent-turn selection with your algorithm. Keep the method's model
clients, prompts and configuration in sibling files as the method grows.
"""
from langmem_eval.interfaces import Session
from .config import Settings


class MyMethodBackend:
    def __init__(self, model: str, *, settings: Settings | None = None):
        self.model = model
        self.settings = settings if settings is not None else Settings.from_env()
        self.records: list[str] = []

    def ingest(self, session: Session) -> None:
        # Normalized JSON already preserves speaker, date, text and source ID.
        self.records.extend(message["content"] for message in session.messages)
        self.records = self.records[-self.settings.window:]

    def retrieve(self, query: str, limit: int) -> list[str]:
        if limit < 1:
            raise ValueError("Retrieval limit must be positive")
        # Deliberately simple example: most recent first, complete evidence only.
        return list(reversed(self.records[-limit:]))

    def describe(self) -> dict:
        return {"implementation": "recent_turns_example_v2", "note_count": len(self.records),
                "settings": self.settings.public_config()}
