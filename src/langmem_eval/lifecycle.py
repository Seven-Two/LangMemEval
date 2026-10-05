"""Conversation-scoped ownership for optional backend resources."""
from contextlib import contextmanager
import sys

from agents_memory.diagnostics import failure, stage


@contextmanager
def managed_backend(backend):
    """Close on every exit; cleanup failures must not replace the primary error."""
    try:
        yield backend
    finally:
        primary_error = sys.exception()
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                with stage("backend.close"):
                    close()
            except Exception:
                if primary_error is None:
                    raise
                failure("backend.close")
