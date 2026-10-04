"""Scoped console/file diagnostics, without logging prompts or request bodies."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import sys
import threading
from time import perf_counter

from tqdm import tqdm

logger = logging.getLogger("langmem_eval.progress")
_context = ContextVar("progress_context", default={})
_state = ContextVar("progress_state", default=None)


class _Console(logging.StreamHandler):
    def emit(self, record):
        tqdm.write(self.format(record), file=self.stream)


def _emit(status, fields, *, exc_info=False):
    message = " ".join(f"{key}={json.dumps(value, ensure_ascii=False)}"
                       for key, value in {"status": status, **fields}.items())
    logger.log(logging.ERROR if status == "failed" else logging.INFO, message, exc_info=exc_info)


def _new_exception():
    state = _state.get()
    if state is None:
        return True
    exception = sys.exception()
    fresh = state.get("last_exception") is not exception
    state["last_exception"] = exception
    return fresh


def event(name, **fields):
    _emit(name, {**_context.get(), **fields})


def failure(name, **fields):
    """Call inside an exception handler to retain the original cause chain."""
    _emit("failed", {**_context.get(), "stage": name, **fields}, exc_info=_new_exception())


@contextmanager
def stage(name, **fields):
    """Time a stage and expose its identifiers to nested operations."""
    state = _state.get()
    context = {**_context.get(), **fields, "parent_stage": _context.get().get("stage"), "stage": name}
    token = _context.set(context)
    started = perf_counter()
    key = object()
    details = {**context, "stage": name}
    if state is not None:
        with state["lock"]:
            state["active"][key] = (started, details)
        _emit("start", details)
    try:
        yield
    except Exception:
        if state is not None:
            _emit("failed", {**details, "elapsed_s": round(perf_counter() - started, 3)},
                  exc_info=_new_exception())
        raise
    except BaseException:
        if state is not None:
            _emit("stopped", {**details, "elapsed_s": round(perf_counter() - started, 3)})
        raise
    else:
        if state is not None:
            _emit("done", {**details, "elapsed_s": round(perf_counter() - started, 3)})
    finally:
        if state is not None:
            with state["lock"]:
                state["active"].pop(key, None)
        _context.reset(token)


@contextmanager
def run_logging(path, *, heartbeat_seconds=30):
    """One run owns its handlers and heartbeat; close and restore on every exit."""
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    state = {"active": {}, "lock": threading.Lock()}
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(heartbeat_seconds):
            with state["lock"]:
                # The most recently entered active stage locates a blocked call.
                current = next(reversed(state["active"].values()), None)
                if current is not None:
                    started, details = current
                    elapsed = perf_counter() - started
                    if elapsed >= heartbeat_seconds:
                        _emit("waiting", {**details, "elapsed_s": round(elapsed, 1)})

    file_handler = logging.FileHandler(path, encoding="utf-8")
    console = _Console(sys.stderr)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (file_handler, console):
        handler.setFormatter(formatter)
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = [console, file_handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    token = _state.set(state)
    worker = threading.Thread(target=heartbeat, name="evaluation-progress", daemon=True)
    worker.start()
    try:
        event("log.open", path=str(path))
        yield
    finally:
        stop.set()
        worker.join()
        _state.reset(token)
        logger.handlers, logger.level, logger.propagate = previous
        for handler in (console, file_handler):
            handler.close()
