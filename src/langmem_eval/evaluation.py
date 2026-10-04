# SPDX-License-Identifier: Apache-2.0
# Adapted from ProsusAI/MemEval systems/_helpers.py; see third_party/memeval/LICENSE
# and NOTICE. Local modification: synchronous execution and existing trace/error fields.
"""Synchronous evaluation for the unified (synchronous) memory interface.

Native adapters retain their async-capable helper. This path needs no event loop
or loopback socket, works inside notebook event loops, and shares the same scoring
functions. Row semantics match agents_memory.systems._helpers._qa_results_async.
"""
from copy import deepcopy
import sys
import traceback


def evaluate_questions(conv, answer_fn, run_judge, category_names=None, judge_fn=None):
    from agents_memory.systems import _helpers as scoring
    from agents_memory.usage import phase

    cats = category_names or scoring._DEFAULT_CATEGORIES
    sample_id = conv.get("sample_id", "unknown")
    qa_pairs = conv.get("qa", [])
    rows = []
    for index, qa in enumerate(qa_pairs):
        question, truth = qa.get("question", ""), qa.get("answer", "")
        category, question_id = qa.get("category", 0), qa.get("question_id", "")
        error = None
        try:
            predicted = answer_fn(question)
            if not isinstance(predicted, str) or not predicted.strip():
                raise ValueError("Answer function returned empty or non-string output")
        except Exception as exc:
            error = {"stage": "answer", "type": type(exc).__name__}
            print(f"    {sample_id} / Q{index + 1}: answer failed", file=sys.stderr)
            traceback.print_exc()
            predicted = ""
        f1 = 0.0 if error else scoring.compute_f1(predicted, truth)
        row = dict(sample_id=sample_id, question=question, ground_truth=truth,
                   predicted=predicted, category=category, category_name=cats.get(category, str(category)),
                   f1=f1, question_id=question_id,
                   evaluation_id=qa.get("evaluation_id", f"{sample_id}:{index}"),
                   status="error" if error else "ok", answer_status="error" if error else "ok",
                   judge_status="skipped" if not run_judge else "not_run", error=error)
        trace = getattr(answer_fn, "trace", None)
        if trace is not None:
            row["answer_trace"] = deepcopy(trace)
            if error and trace.get("failure_stage"):
                error["stage"] = trace["failure_stage"]
        if run_judge and not error:
            try:
                with phase("judge"):
                    scores = (scoring.evaluate_longmemeval(question, truth, predicted,
                                category=str(category), question_id=question_id)
                              if judge_fn == "longmemeval"
                              else scoring.evaluate_with_judge(question, truth, predicted))
                row.update(scores)
                row["judge_status"] = scores.get("judge_status", "ok")
            except Exception as exc:
                print(f"    {sample_id} / Q{index + 1}: judge failed", file=sys.stderr)
                traceback.print_exc()
                row.update(judge_status="error", judge_error={"type": type(exc).__name__})
            if row["judge_status"] == "error":
                row["status"] = "error"
        rows.append(row)
        if (index + 1) % 20 == 0:
            print(f"    QA {index + 1}/{len(qa_pairs)} - F1={f1:.3f}")
    return rows
