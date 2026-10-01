"""Runtime data belongs to the working directory, never the installed package."""
import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.getenv("EVAL_DATA_DIR", "data")).expanduser().resolve()


def models_dir() -> Path:
    return Path(os.getenv("EVAL_MODELS_DIR", "models")).expanduser().resolve()


def results_dir() -> Path:
    return Path(os.getenv("EVAL_RESULTS_DIR", "results")).expanduser().resolve()
