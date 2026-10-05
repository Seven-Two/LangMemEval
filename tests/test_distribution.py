"""Run after uv build --wheel to verify the standalone distribution."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


def test_wheel_runner_outside_checkout(tmp_path):
    wheels = list((Path(__file__).resolve().parents[1] / "dist").glob("*.whl"))
    if not wheels:
        pytest.skip("Run uv build --wheel first")
    with zipfile.ZipFile(max(wheels, key=lambda p: p.stat().st_mtime)) as archive:
        removed = {"langmem_eval/amem.py", "langmem_eval/backend.py",
                   "langmem_eval/_amem_prompts.py", "agents_memory/token_tracker.py",
                   "agents_memory/systems/langmem.py"}
        assert removed.isdisjoint(archive.namelist())
        assert not any(name.startswith("agents_memory/systems/") for name in archive.namelist())
        assert not any(name.startswith("agents_memory/training/") for name in archive.namelist())
        for name in ("agents_memory/runner.py", "agents_memory/paths.py", "agents_memory/LICENSE",
                     "agents_memory/NOTICE", "langmem_eval/cli.py", "langmem_eval/lifecycle.py",
                     "langmem_eval/methods/amem/evolution.py", "langmem_eval/methods/amem/responses.py"):
            assert name in archive.namelist()
        for method in ("amem", "langmem"):
            assert f"langmem_eval/methods/{method}/backend.py" in archive.namelist()
        packaged_methods = {name.split('/')[2] for name in archive.namelist()
                            if name.startswith('langmem_eval/methods/') and name.endswith('/backend.py')}
        assert packaged_methods == {"amem", "langmem"}
        archive.extractall(tmp_path)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for key in ("EVAL_DATA_DIR", "EVAL_MODELS_DIR", "EVAL_RESULTS_DIR"):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); "
         "from pathlib import Path; import agents_memory.runner as runner; "
         "assert Path(runner.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()); "
         "from agents_memory.paths import data_dir, results_dir; "
         "assert data_dir() == Path.cwd() / 'data'; "
         "assert results_dir() == Path.cwd() / 'results'; "
         "from langmem_eval.cli import main; "
         "sys.argv=['langmem-eval', '--help']; main()", str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--systems" in result.stdout
