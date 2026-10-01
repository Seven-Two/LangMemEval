"""Discover method modules without constructing models or calling APIs."""
import importlib
import pkgutil
import re
from dataclasses import dataclass
from functools import partial
from typing import Callable

from .interfaces import MemoryBackend


@dataclass(frozen=True)
class Method:
    factory: Callable[[str], MemoryBackend]
    architecture: str
    infrastructure: str
    aml_factory: Callable | None = None


METHODS: dict[str, Method] = {}


def register_method(name: str, *, architecture: str, infrastructure: str = "custom", aml_factory=None):
    """Decorate a factory/class accepting the benchmark model as one argument."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name == "all":
        raise ValueError(f"Invalid method name: {name!r}")

    def register(factory):
        if name in METHODS:
            raise ValueError(f"Method already registered: {name}")
        if not callable(factory):
            raise TypeError("Method factory must be callable")
        METHODS[name] = Method(factory, architecture, infrastructure, aml_factory)
        return factory
    return register


def discover_methods():
    """Import public modules or package __init__ files, never recurse into internals."""
    from . import methods
    for module in sorted(pkgutil.iter_modules(methods.__path__), key=lambda m: m.name):
        if not module.name.startswith("_"):
            importlib.import_module(f"{methods.__name__}.{module.name}")
    return dict(METHODS)


def create_backend(name: str, model: str) -> MemoryBackend:
    methods = discover_methods()
    if name not in methods:
        raise ValueError(f"Unknown method {name!r}; available: {', '.join(sorted(methods))}")
    backend = methods[name].factory(model)
    if not all(callable(getattr(backend, attr, None)) for attr in ("ingest", "retrieve")):
        raise TypeError(f"Method {name!r} must implement ingest and retrieve")
    return backend


def system_entries():
    from .adapter import run, run_method
    return {
        name: {"architecture": spec.architecture, "infrastructure": spec.infrastructure,
               "fn": run if name == "langmem" else partial(run_method, name)}
        for name, spec in discover_methods().items()
    }
