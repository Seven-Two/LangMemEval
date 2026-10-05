"""Completed A-Mem state cache. JSON only; never pickle or re-embed stored notes.

The actual matrix must survive unchanged: neighbor evolution intentionally leaves
some embeddings stale until the next scheduled rebuild. Re-encoding notes on load
would change the method. Query/answer outputs are never written to this cache.
"""
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np

from agents_memory.diagnostics import event
from ...embeddings import validate_vectors
from .memory import Note

SCHEMA = 1


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def code_identity():
    """Invalidate conservatively when write implementation or dependencies change."""
    root = Path(__file__).resolve().parents[2]
    files = [*Path(__file__).parent.glob("*.py"), root / "embeddings.py",
             root / "model_api.py", root / "configuration.py", root / "interfaces.py",
             root / "benchmark.py", root / "llm_diagnostics.py"]
    sources = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
               for p in files}
    dependencies = {}
    for name in ("numpy", "openai", "torch", "sentence-transformers", "transformers"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {"sources": sources, "dependencies": dependencies}


def cache_key(backend, sessions):
    settings = asdict(backend.settings)
    for field in ("embedding_api_key", "cache_mode", "cache_dir"):
        settings.pop(field)
    # Hash full extra_body and endpoints for identity, but never serialize them to
    # disk or logs. Rotating credentials alone does not invalidate a memory.
    return digest({"schema": SCHEMA, "history": [asdict(s) for s in sessions],
                   "model": backend.model, "llm_base_url": os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
                   "settings": settings, "code": code_identity()})


def validate_state(state, backend):
    if not isinstance(state, dict):
        raise ValueError("Invalid state object")
    raw_notes = state["notes"]
    if not isinstance(raw_notes, list):
        raise ValueError("Invalid notes list")
    notes = []
    for item in raw_notes:
        if not isinstance(item, dict) or set(item) != {f.name for f in fields(Note)}:
            raise ValueError("Invalid note fields")
        for key in ("id", "source_id", "session_id", "timestamp", "content", "context"):
            if not isinstance(item[key], str):
                raise ValueError("Invalid note text")
        for key in ("keywords", "tags", "links"):
            if not isinstance(item[key], list) or not all(isinstance(v, str) for v in item[key]):
                raise ValueError("Invalid note list")
        notes.append(Note(**item))
    ids = {n.id for n in notes}
    if "" in ids or len(ids) != len(notes) or any(link not in ids for n in notes for link in n.links):
        raise ValueError("Invalid note identities or links")
    if notes:
        vectors = validate_vectors(state["vectors"], rows=len(notes), dims=backend.settings.embedding_dims)
    else:
        if state["vectors"] is not None:
            raise ValueError("Empty memory must not contain vectors")
        vectors = None
    count = state["evolution_count"]
    if type(count) is not int or not 0 <= count <= len(notes):
        raise ValueError("Invalid evolution counter")
    for key, expected in (("evolution_stats", backend.evolution_stats),
                          ("output_adjustments", backend.output_adjustments)):
        values = state[key]
        if not isinstance(values, dict) or set(values) != set(expected):
            raise ValueError("Invalid diagnostic counters")
        if any(type(v) is not int or v < 0 for v in values.values()):
            raise ValueError("Invalid diagnostic counter value")
    return notes, vectors, count


def restore(backend, sessions):
    mode = backend.settings.cache_mode
    if mode == "off":
        return False
    if backend.notes or backend.vectors is not None:
        raise ValueError("Cache restore requires a fresh A-Mem backend")
    key = cache_key(backend, sessions)
    path = Path(backend.settings.cache_dir).expanduser() / f"{key}.json"
    backend.cache_info = {"mode": mode, "status": "refresh" if mode == "refresh" else "miss",
                          "key": key, "path": str(path)}
    if mode == "refresh":
        prepare_write(path)
        event("amem.cache.refresh", key=key, path=str(path))
        return False
    if not path.is_file():
        event("amem.cache.miss", key=key, path=str(path))
        if mode == "require":
            raise ValueError("A-Mem cache miss in require mode; build once with --amem-cache-mode reuse")
        prepare_write(path)
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["schema"] != SCHEMA or payload["key"] != key:
            raise ValueError("Cache schema or identity mismatch")
        if payload["state_sha256"] != digest(payload["state"]):
            raise ValueError("Cache checksum mismatch")
        if not all(isinstance(payload[k], str) and payload[k] for k in ("build_id", "created_at")):
            raise ValueError("Invalid cache provenance")
        notes, vectors, count = validate_state(payload["state"], backend)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        # Never silently trigger hours of rebuilding after a corrupted read.
        raise ValueError("Invalid A-Mem cache; use --amem-cache-mode refresh to rebuild") from exc
    backend.notes, backend.vectors, backend.evolution_count = notes, vectors, count
    backend.evolution_stats = payload["state"]["evolution_stats"]
    backend.output_adjustments = payload["state"]["output_adjustments"]
    backend.cache_info.update(status="hit", build_id=payload["build_id"],
                              created_at=payload["created_at"], state_sha256=payload["state_sha256"])
    event("amem.cache.hit", key=key, build_id=payload["build_id"], notes=len(notes), path=str(path))
    return True


def prepare_write(path):
    """Catch a misconfigured/unwritable cache directory before expensive writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".amem-check-", suffix=".tmp"):
        pass


def save(backend, sessions):
    mode = backend.settings.cache_mode
    if mode == "off":
        return
    if mode == "require":
        raise ValueError("require mode cannot build a new cache")
    key = cache_key(backend, sessions)
    state = {**backend.snapshot(), "vectors": backend.vectors.tolist() if backend.vectors is not None else None,
             "evolution_stats": dict(backend.evolution_stats), "output_adjustments": dict(backend.output_adjustments)}
    validate_state(state, backend)
    payload = {"schema": SCHEMA, "key": key, "build_id": uuid4().hex,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "state_sha256": digest(state), "state": state}
    path = Path(backend.settings.cache_dir).expanduser() / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".amem-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(canonical(payload))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    backend.cache_info = {"mode": mode, "status": "saved", "key": key, "path": str(path),
                          "build_id": payload["build_id"], "created_at": payload["created_at"],
                          "state_sha256": payload["state_sha256"]}
    event("amem.cache.saved", key=key, build_id=payload["build_id"], notes=len(backend.notes), path=str(path))
