"""Method-owned settings; the registry binds CLI/.env sources before construction."""
from dataclasses import asdict, dataclass
import os


@dataclass(frozen=True)
class Settings:
    window: int = 100

    def __post_init__(self):
        if type(self.window) is not int or self.window < 1:
            raise ValueError("MY_METHOD_WINDOW must be a positive integer")

    @classmethod
    def from_env(cls):
        return cls(window=int(os.getenv("MY_METHOD_WINDOW", "100")))

    def public_config(self):
        return asdict(self)  # Never include API keys in this dictionary.
