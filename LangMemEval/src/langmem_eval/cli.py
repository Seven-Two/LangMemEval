"""One entry point for the editable LangMem + MemEval research workspace."""
import runpy
import sys
from pathlib import Path


def main():
    if sys.argv[1:] == ["--list-methods"]:
        from .registry import discover_methods
        for name, spec in sorted(discover_methods().items()):
            print(f"{name}: {spec.architecture}")
        return
    import agents_memory

    root = Path(agents_memory.__file__).resolve().parents[2]
    runner = root / "scripts" / "run_full_benchmark.py"
    if not runner.is_file():
        raise SystemExit("MemEval must be installed editable from its source checkout. "
                         "Run: uv sync --extra dev")
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
