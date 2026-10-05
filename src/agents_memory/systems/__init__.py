"""Lazy native adapters plus the independent unified-method registry.

Listing names or importing scoring never loads optional native baselines.
Only reading a selected native entry imports its implementation.
"""
from collections.abc import Mapping
import importlib
import pkgutil

from langmem_eval.registry import system_entries


UNAVAILABLE_SYSTEMS: dict[str, str] = {}


class _NativeSystem(Mapping):
    def __init__(self, name):
        self.name = name
        self._entry = None

    def _load(self):
        if self._entry is None:
            try:
                module = importlib.import_module(f"{__name__}.{self.name}")
            except ModuleNotFoundError as exc:
                UNAVAILABLE_SYSTEMS[self.name] = str(exc)
                raise ValueError(f"Native system {self.name!r} needs an optional dependency: {exc}. "
                                 "Install its extra or select another --systems value.") from exc
            self._entry = {**module.SYSTEM_INFO, "fn": module.run}
        return self._entry

    def __getitem__(self, key):
        return self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


def discover_systems():
    entries = system_entries()
    for module in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):
        if module.name.startswith("_"):
            continue
        if module.name in entries:
            raise ValueError(f"Duplicate system name: {module.name}")
        entries[module.name] = _NativeSystem(module.name)
    return entries


SYSTEMS = discover_systems()
