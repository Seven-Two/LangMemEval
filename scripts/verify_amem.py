"""Fixed autoresearch verification; no model requests, no benchmark score claims."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHON_DOTENV_DISABLED="1")
# Make local secrets and paid providers unavailable to all child tests.
for key in list(env):
    if key.endswith("API_KEY") or key.startswith(("AMEM_", "EVAL_", "LLM_")):
        env.pop(key, None)
env["OPENAI_API_KEY"] = "offline-test-placeholder"
env["OPENAI_BASE_URL"] = "http://127.0.0.1:1/v1"
guard = "--guard" in sys.argv
temporary_root = ROOT / "results" / "amem-verification"
temporary_root.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
    report = Path(directory) / "report.xml"
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
               "--basetemp", str(Path(directory) / "tmp"), "--junitxml", str(report)]
    command += ["tests", "--ignore=tests/test_amem_contract.py"] if guard else ["tests/test_amem_contract.py"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8")
    # Keep diagnostics in the control log while the last stdout line is numeric.
    print(result.stdout, file=sys.stderr)
    print(result.stderr, file=sys.stderr)
    if guard:
        sys.exit(result.returncode)
    if not report.exists() or result.returncode not in (0, 1):
        raise SystemExit("Verification infrastructure failed, not a measured experiment")
    cases = ET.parse(report).findall(".//testcase")
    if len(cases) != 10:
        raise SystemExit(f"Contract changed: expected 10 cases, found {len(cases)}")
    passed = sum(not any(c.tag in {"failure", "error", "skipped"} for c in case) for case in cases)
    print(json.dumps({"amem_contract_pass_rate": passed / 10}))
