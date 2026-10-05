"""Discover method modules without constructing models or calling APIs."""
import importlib
import pkgutil
import re
from dataclasses import dataclass
from typing import Callable

from .interfaces import MemoryBackend


@dataclass(frozen=True)
class MethodOption:
    """Method-owned CLI/.env binding; defaults live in the method settings."""

    flag: str
    env: str
    value_type: type = str
    choices: tuple | None = None
    help: str = ""

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9-]*", self.flag):
            raise ValueError(f"Invalid method option flag: {self.flag!r}")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.env):
            raise ValueError(f"Invalid method environment key: {self.env!r}")
        if self.value_type not in (str, int, float):
            raise ValueError("Method options support str, int or float converters")

    @property
    def dest(self):
        return self.flag.replace("-", "_")


@dataclass(frozen=True)
class Method:
    factory: Callable[[str], MemoryBackend]
    architecture: str
    infrastructure: str
    aml_factory: Callable | None = None
    options: tuple[MethodOption, ...] = ()
    public_config: Callable[[], dict] | None = None
    dependencies: tuple[str, ...] = ()
    extra: str | None = None
    embedding_scope: str = "shared"


METHODS: dict[str, Method] = {}


def register_method(name: str, *, architecture: str, infrastructure: str = "custom", aml_factory=None,
                    options: tuple[MethodOption, ...] = (), public_config=None,
                    dependencies: tuple[str, ...] = (), extra=None, embedding_scope="shared"):
    """Decorate a factory/class accepting the benchmark model as one argument."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name == "all":
        raise ValueError(f"Invalid method name: {name!r}")

    def register(factory):
        if name in METHODS:
            raise ValueError(f"Method already registered: {name}")
        if not callable(factory):
            raise TypeError("Method factory must be callable")
        if not all(isinstance(option, MethodOption) for option in options):
            raise TypeError("Method options must be MethodOption instances")
        if public_config is not None and not callable(public_config):
            raise TypeError("public_config must be a callable without model initialization")
        METHODS[name] = Method(factory, architecture, infrastructure, aml_factory, tuple(options), public_config,
                               tuple(dependencies), extra, embedding_scope)
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
    check_dependencies(name, methods[name])
    backend = methods[name].factory(model)
    if not all(callable(getattr(backend, attr, None)) for attr in ("ingest", "retrieve")):
        raise TypeError(f"Method {name!r} must implement ingest and retrieve")
    return backend


def check_dependencies(name: str, method: Method):
    """Check selected SDK availability without importing it or constructing models."""
    import importlib.util
    missing = [module for module in method.dependencies if importlib.util.find_spec(module) is None]
    if missing:
        install = f"uv sync --locked --extra {method.extra}" if method.extra else "install its dependencies"
        raise ValueError(f"Method {name!r} needs {', '.join(missing)}; {install}")
