# New method template

Copy this directory to `src/langmem_eval/methods/my_method/`. Change the folder,
registration name and class name for your method. This example is not registered
while it remains under `examples/`, so it cannot accidentally join `--systems all`.

The example returns recent historical turns without memory LLM/embedding calls.
Running it through the evaluator still calls the configured answer model and,
unless `--skip-judge` is used, the judge. It is an interface example, not a paper baseline.

Keep registration in `__init__.py` lightweight; implement `ingest`/`retrieve` in
`backend.py`. Add `config.py`, `prompts.py`, `memory.py` and `clients.py` only as
needed. Never share mutable memory across backend instances or access gold answers.

See [the development guide](../../docs/development.md) for commands and contracts.
