"""Memory system registry. Drop a .py file here to add a system."""

import importlib
import pkgutil
import warnings

SYSTEMS: dict[str, dict] = {}
UNAVAILABLE_SYSTEMS: dict[str, str] = {}

for _, name, _ in pkgutil.iter_modules(__path__):
    if name.startswith("_"):
        continue
    try:
        mod = importlib.import_module(f".{name}", __package__)
    except ModuleNotFoundError as exc:
        UNAVAILABLE_SYSTEMS[name] = str(exc)
        warnings.warn(f"System {name} unavailable: {exc}", stacklevel=2)
        continue
    if hasattr(mod, "SYSTEM_INFO") and hasattr(mod, "run"):
        info = mod.SYSTEM_INFO.copy()
        info["fn"] = mod.run
        if name in SYSTEMS:
            raise ValueError(f"Duplicate system name: {name}")
        SYSTEMS[name] = info
    for system_name, info in getattr(mod, "SYSTEM_ENTRIES", {}).items():
        if system_name in SYSTEMS:
            raise ValueError(f"Duplicate system name: {system_name}")
        SYSTEMS[system_name] = info
