"""Shared utilities for memory system adapters."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from agents_memory.diagnostics import event, failure, stage

from agents_memory.usage import phase

from agents_memory.evaluation import (
    compute_f1,
    evaluate_longmemeval,
    evaluate_with_judge,
)
from agents_memory.locomo import CATEGORY_NAMES as _DEFAULT_CATEGORIES


async def _qa_results_async(
    conv: dict,
    answer_fn,
    run_judge: bool,
    category_names: dict | None = None,
    judge_fn: str | None = None,
) -> list[dict]:
    """Evaluate all QA pairs for a conversation.

    Accepts both sync and async answer_fn -- dispatches accordingly.

    Parameters
    ----------
    judge_fn : str | None
        Which judge to use.  ``"longmemeval"`` uses the native LongMemEval
        binary-accuracy judge (matching the ICLR 2025 paper prompts).
        ``None`` (default) uses the generic 3-dimension judge.
    """
    cats = category_names or _DEFAULT_CATEGORIES
    is_async = asyncio.iscoroutinefunction(answer_fn)
    qa_pairs = conv.get("qa", [])
    sample_id = conv.get("sample_id", "unknown")
    results = []

    for i, qa in enumerate(qa_pairs, 1):
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")
        category = qa.get("category", 0)
        question_id = qa.get("question_id", "")

        error = None
        try:
            with stage("question.answer", question_index=i, question_total=len(qa_pairs)):
                predicted = (await answer_fn(question)) if is_async else answer_fn(question)
                if not isinstance(predicted, str) or not predicted.strip():
                    raise ValueError("Answer function returned empty or non-string output")
        except Exception as err:
            error = {"stage": "answer", "type": type(err).__name__}
            failure("question.answer", conversation=sample_id, question_index=i)
            predicted = ""

        # An execution error is never a correct abstention, even with empty truth.
        f1 = 0.0 if error else compute_f1(predicted, ground_truth)

        row = {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "category": category,
            "category_name": cats.get(category, str(category)),
            "f1": f1,
            "question_id": question_id,
            "evaluation_id": qa.get("evaluation_id", f"{sample_id}:{i - 1}"),
            "status": "error" if error else "ok",
            "answer_status": "error" if error else "ok",
            "judge_status": "skipped" if not run_judge else "not_run",
            "error": error,
        }
        trace = getattr(answer_fn, "trace", None)
        if trace is not None:
            row["answer_trace"] = deepcopy(trace)
            if error and trace.get("failure_stage"):
                error["stage"] = trace["failure_stage"]

        if run_judge and not error:
            try:
                with phase("judge"), stage("judge", question_index=i, question_total=len(qa_pairs)):
                    if judge_fn == "longmemeval":
                        scores = evaluate_longmemeval(
                            question, ground_truth, predicted,
                            category=str(category), question_id=question_id)
                    else:
                        scores = evaluate_with_judge(question, ground_truth, predicted)
                row.update(scores)
                row["judge_status"] = scores.get("judge_status", "ok")
            except Exception as err:
                failure("judge", conversation=sample_id, question_index=i)
                row.update(judge_status="error", judge_error={"type": type(err).__name__})
            if row["judge_status"] == "error":
                row["status"] = "error"

        results.append(row)
        event("question.result", question_index=i, question_total=len(qa_pairs),
              answer_status=row["answer_status"], judge_status=row["judge_status"], f1=f1)

        if i % 20 == 0:
            print(f"    QA {i}/{len(qa_pairs)} - F1={f1:.3f}")

    return results


def _qa_results(
    conv: dict,
    answer_fn,
    run_judge: bool,
    category_names: dict | None = None,
    judge_fn: str | None = None,
) -> list[dict]:
    """Sync wrapper around _qa_results_async."""
    return asyncio.run(
        _qa_results_async(
            conv, answer_fn, run_judge,
            category_names=category_names,
            judge_fn=judge_fn,
        )
    )


def run_async(async_fn):
    """Wrap an async adapter into the sync signature the runner expects."""
    def wrapper(
        conv, llm_model, run_judge,
        category_names=None, judge_fn=None,
    ):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                async_fn(
                    conv, llm_model, run_judge,
                    category_names=category_names,
                    judge_fn=judge_fn,
                )
            )
        finally:
            loop.close()
    wrapper.__doc__ = async_fn.__doc__
    return wrapper
