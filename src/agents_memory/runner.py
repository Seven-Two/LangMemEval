#!/usr/bin/env python3
"""Run methods against a frozen question manifest with auditable failures and costs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from dotenv import load_dotenv
from tqdm import tqdm

from agents_memory.benchmarks import BENCHMARKS
from agents_memory.locomo import CATEGORY_NAMES
from agents_memory.systems import SYSTEMS
from agents_memory.experiment import compute_summary, failed_results, freeze_manifest, normalize_results
from agents_memory.usage import get_report, get_stats, get_stats_by_model, reset, start
from agents_memory.paths import results_dir
from langmem_eval.protocol import AnswerProtocol
from langmem_eval.registry import discover_methods

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), default="locomo")
    parser.add_argument("--split", help="Benchmark split, e.g. oracle/s/m")
    parser.add_argument("--systems", default="all")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--llm-model")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--output-dir", help="Default: EVAL_RESULTS_DIR or ./results")
    parser.add_argument("--data-file", help="Custom MemEval-normalized JSON")
    parser.add_argument("--protocol", choices=("auto", "locomo", "longmemeval"),
                        help="Unified adapter answer defaults (not an official reproduction protocol)")
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--tokenizer", help="Explicit tiktoken encoding, default cl100k_base")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--answer-temperature", type=float)
    parser.add_argument("--answer-style", choices=("concise", "complete"))
    parser.add_argument("--empty-context", choices=("abstain", "answer"))
    parser.add_argument("--abstention-text")
    return parser.parse_args()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    load_dotenv()
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be positive")
    overrides = {
        "EVAL_PROTOCOL": args.protocol, "EVAL_CONTEXT_TOKENS": args.context_tokens,
        "EVAL_TOKENIZER": args.tokenizer, "EVAL_TOP_K": args.top_k,
        "EVAL_MAX_OUTPUT_TOKENS": args.max_output_tokens,
        "EVAL_TEMPERATURE": args.answer_temperature, "EVAL_ANSWER_STYLE": args.answer_style,
        "EVAL_EMPTY_CONTEXT": args.empty_context, "EVAL_ABSTENTION_TEXT": args.abstention_text,
    }
    for key, value in overrides.items():
        if value is not None:
            os.environ[key] = str(value)
    llm_model = args.llm_model or os.getenv("LLM_MODEL", "gpt-4.1")
    run_judge = not args.skip_judge
    system_names = list(SYSTEMS) if args.systems.lower() == "all" else [s.strip() for s in args.systems.split(",")]
    if not system_names or len(set(system_names)) != len(system_names):
        raise SystemExit("Select distinct system names")
    unknown = [s for s in system_names if s not in SYSTEMS]
    if unknown:
        raise SystemExit(f"Unknown/unavailable systems: {unknown}. Available: {list(SYSTEMS)}")
    bench = BENCHMARKS[args.benchmark]
    categories = bench.get("category_names", CATEGORY_NAMES)
    judge_fn = bench.get("judge_fn")
    judge_model = (os.getenv("LONGMEMEVAL_JUDGE_MODEL", "gpt-4o") if judge_fn == "longmemeval"
                   else os.getenv("JUDGE_MODEL", "gpt-5.2")) if run_judge else None
    protocol = AnswerProtocol.from_env(judge_fn)
    # Fail before paid calls if the selected encoding cannot be loaded.
    from langmem_eval.benchmark import token_encoding
    registered = discover_methods()
    if any(s in registered for s in system_names):
        token_encoding(protocol.tokenizer)
    if args.data_file:
        data = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
        conversations = (data if isinstance(data, list) else [data])[:args.num_samples]
    else:
        conversations = bench["download"](split=args.split, num_samples=args.num_samples)
    conversations, manifest = freeze_manifest(conversations)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = re.sub(r"[^A-Za-z0-9_-]", "_", llm_model)
    run_tag = f"{args.benchmark}_{model_tag}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    manifest_path = output_dir / f"manifest_{run_tag}.json"
    _write_json(manifest_path, manifest)
    print(f"Benchmark: {bench['name']}; questions: {manifest['n_questions']}; judge: {judge_model}")
    print(f"Manifest: {manifest_path}")
    start()
    all_summaries = {}
    for name in system_names:
        info = SYSTEMS[name]
        unified = name in registered
        print(f"\nSYSTEM: {name}; protocol: {'unified' if unified else 'native adapter (own settings)'}")
        reset()
        results = []
        checkpoints = output_dir / f"{name}_{run_tag}_progress.jsonl"
        with checkpoints.open("w", encoding="utf-8") as progress:
            for conv in tqdm(conversations, desc=name):
                try:
                    returned = info["fn"](
                        deepcopy(conv), llm_model, run_judge,
                        category_names=categories, judge_fn=judge_fn)
                    rows = normalize_results(conv, returned, categories, run_judge, judge_fn)
                except Exception as exc:
                    print(f"  {conv['sample_id']}: {type(exc).__name__}")
                    rows = failed_results(conv, categories, run_judge,
                                          stage="conversation", error_type=type(exc).__name__)
                results.extend(rows)
                for row in rows:
                    progress.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                progress.flush()
        summary = compute_summary(results, run_judge, judge_fn)
        usage = get_report()
        config = {
            "architecture": info["architecture"], "infrastructure": info["infrastructure"],
            "llm_model": llm_model, "judge_model": judge_model,
            "adapter_protocol": "unified_v1" if unified else "native",
            "answer_protocol": protocol.to_dict() if unified else None,
            "context_trace": "per_question" if unified else "not_instrumented",
            "phase_instrumentation": "write/retrieve/answer/judge" if unified else "partial; inspect unclassified",
        }
        payload = {
            "schema_version": 2, "system": name, "benchmark": bench["name"], "split": args.split,
            "llm_model": llm_model, "timestamp": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest_path.name, "dataset_sha256": manifest["dataset_sha256"],
            "run_status": summary["run_status"], "config": config, "summary": summary,
            "token_usage": get_stats(), "token_breakdown": get_stats_by_model(),
            "cost_accounting": usage, "results": results,
        }
        _write_json(output_dir / f"{name}_{run_tag}_results.json", payload)
        all_summaries[name] = {**summary, "config": config, "token_usage": payload["token_usage"],
                               "cost_accounting": usage}
        print(f"  {summary['run_status'].upper()}: coverage={summary['coverage']:.1%}, "
              f"F1={summary['overall_f1_mean']:.3f} (fixed denominator), "
              f"observed method tokens={payload['token_usage']['total_tokens']}, "
              f"judge tokens={usage['judge']['total_tokens']}")
        # Save after each system so a later interrupted method cannot erase earlier results.
        _write_json(output_dir / f"benchmark_summary_{run_tag}.json", {
            "schema_version": 2, "benchmark": bench["name"], "split": args.split,
            "llm_model": llm_model, "judge_model": judge_model,
            "manifest": manifest_path.name, "dataset_sha256": manifest["dataset_sha256"],
            "requested_systems": system_names,
            "run_status": "complete" if len(all_summaries) == len(system_names) and all(
                s["run_status"] == "complete" for s in all_summaries.values()) else "incomplete",
            "systems": all_summaries,
        })
    if any(s["run_status"] != "complete" for s in all_summaries.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
