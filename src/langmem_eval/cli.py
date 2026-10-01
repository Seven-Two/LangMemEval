"""One entry point for the editable LangMem + MemEval research workspace."""
import sys
from pathlib import Path


def main():
    if sys.argv[1:] == ["--list-methods"]:
        from .registry import discover_methods
        for name, spec in sorted(discover_methods().items()):
            print(f"{name}: {spec.architecture}")
        return
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    from agents_memory.runner import main as run_benchmark
    run_benchmark()


if __name__ == "__main__":
    main()
