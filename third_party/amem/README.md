# A-Mem provenance

Upstream: https://github.com/WujiangXu/A-mem

Pinned commit: `0c8039f28fdcc08189a23c07a3437d9d2482f9c2`.

This integration adapts the original JSON implementation in `memory_layer.py`
and query-keyword generation in `test_advanced.py`. It does not use the separately
packaged A-mem-sys or the robust variant. Original analysis/evolution/query prompt
strings and JSON schemas were extracted from the pinned Python AST into
`src/langmem_eval/_amem_prompts.py`; query f-string literals were escaped for
equivalent `.format(question=...)` rendering. No upstream execution was needed.

SHA-256 of the downloaded upstream `memory_layer.py`:
`b9a5b5797881b25c5524b98a543b22a1bbbcc7f285437ca667a5673e7f293eba`.

The core is reimplemented around the repository's `MemoryBackend` interface,
with isolated conversations, stable validated links, atomic writes, explicit
failures and independent model providers. See [the integration guide](../../docs/amem.md)
for all evaluation and algorithmic differences. This is not a claim of exact
paper-score reproduction. Upstream MIT license is retained verbatim in
[LICENSE](LICENSE) and packaged as `langmem_eval/AMEM_LICENSE` in wheels.
