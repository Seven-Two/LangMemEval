"""A-Mem note schema and the original index/context serialization."""
from dataclasses import dataclass, field


@dataclass
class Note:
    id: str
    source_id: str
    session_id: str
    timestamp: str
    content: str
    keywords: list[str]
    context: str
    tags: list[str]
    links: list[str] = field(default_factory=list)

    def indexed_text(self, consolidated=False):
        if consolidated:
            return f"{self.content} , {self.context} {' '.join(self.keywords)} {' '.join(self.tags)}"
        return ("content:" + self.content + " context:" + self.context + " keywords: "
                + ", ".join(self.keywords) + " tags: " + ", ".join(self.tags))

    def record(self):
        return (f"talk start time:{self.timestamp}memory content: {self.content}"
                f"memory context: {self.context}memory keywords: {self.keywords}"
                f"memory tags: {self.tags}")
