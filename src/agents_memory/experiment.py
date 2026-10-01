"""Fixed evaluation denominators and explicit operational failure accounting."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from statistics import mean, pstdev


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def freeze_manifest(conversations):
    conversations = deepcopy(conversations)
    manifest = []
    for ci, conv in enumerate(conversations):
        conv["source_sample_id"] = conv.get("sample_id")
        conv["sample_id"] = f"c{ci:06d}"
        if not conv.get("qa"):
            raise ValueError("Every selected conversation must contain QA pairs")
        for qi, qa in enumerate(conv["qa"]):
            qa["evaluation_id"] = f"{conv['sample_id']}:q{qi:06d}"
            manifest.append({"evaluation_id": qa["evaluation_id"],
                             "sample_id": conv["sample_id"],
                             "source_sample_id": conv["source_sample_id"],
                             "question_id": qa.get("question_id", ""),
                             "question": qa.get("question", ""),
                             "category": qa.get("category", 0),
                             "reference_sha256": fingerprint(qa.get("answer", ""))})
    if not manifest:
        raise ValueError("No evaluation questions selected")
    return conversations, {"schema_version": 1, "dataset_sha256": fingerprint(conversations),
                            "n_questions": len(manifest), "questions": manifest}


def _base(conv, qa, categories):
    category = qa.get("category", 0)
    return {"evaluation_id": qa["evaluation_id"], "sample_id": conv["sample_id"],
            "source_sample_id": conv.get("source_sample_id"),
            "question_id": qa.get("question_id", ""), "question": qa.get("question", ""),
            "ground_truth": qa.get("answer", ""), "category": category,
            "category_name": categories.get(category, str(category))}


def failed_results(conv, categories, run_judge, *, stage, error_type):
    return [{**_base(conv, qa, categories), "predicted": "", "f1": 0.0,
             "status": "error", "answer_status": "error",
             "judge_status": "not_run" if run_judge else "skipped",
             "error": {"stage": stage, "type": error_type}}
            for qa in conv["qa"]]


def normalize_results(conv, results, categories, run_judge, judge_fn):
    """Map results to the frozen manifest; missing rows remain explicit errors.

    Legacy adapters without IDs may return a complete ordered list, or an
    unambiguous subset matched by question. Ambiguous/duplicate IDs are errors.
    """
    if not isinstance(results, list):
        raise ValueError("Adapter must return a list of result rows")
    qas = conv["qa"]
    by_id = {q["evaluation_id"]: q for q in qas}
    mapped = {}
    for i, row in enumerate(results):
        key = row.get("evaluation_id")
        if not key:
            if len(results) == len(qas) and row.get("question") == qas[i].get("question"):
                key = qas[i]["evaluation_id"]
            else:
                matches = [q["evaluation_id"] for q in qas
                           if q.get("question") == row.get("question")]
                if len(matches) != 1:
                    raise ValueError("Ambiguous legacy result; provide evaluation_id")
                key = matches[0]
        if key not in by_id or key in mapped:
            raise ValueError("Unexpected or duplicate evaluation_id")
        mapped[key] = row
    output = failed_results(conv, categories, run_judge, stage="adapter", error_type="MissingResult")
    for i, qa in enumerate(qas):
        raw = mapped.get(qa["evaluation_id"])
        if raw is None:
            continue
        row = {**raw, **_base(conv, qa, categories)}
        row.setdefault("answer_status", "error" if row.get("status") == "error" else "ok")
        if row["answer_status"] == "ok" and (not isinstance(row.get("predicted"), str) or not row["predicted"].strip()):
            row.update(answer_status="error", error={"stage": "answer", "type": "EmptyAnswer"})
        row["status"] = "ok" if row["answer_status"] == "ok" else "error"
        # Recompute with canonical references; do not accept adapter-supplied metrics.
        from .evaluation import compute_f1
        row["f1"] = compute_f1(row["predicted"], qa.get("answer", "")) if row["answer_status"] == "ok" else 0.0
        if not run_judge:
            row["judge_status"] = "skipped"
        elif row["answer_status"] != "ok":
            row["judge_status"] = "not_run"
        else:
            fields = ("longmemeval_correct",) if judge_fn == "longmemeval" else (
                "judge_relevant", "judge_complete", "judge_accurate")
            valid = all(row.get(f) in (0, 1) for f in fields)
            if row.get("judge_status") == "error" or not valid:
                row.update(judge_status="error", status="error")
                row.setdefault("judge_error", {"type": "MissingOrInvalidJudgeResult"})
            else:
                row["judge_status"] = "ok"
        output[i] = row
    return output


def compute_summary(rows, run_judge=False, judge_fn=None):
    if not rows:
        return {}
    by_conv, by_category = defaultdict(list), defaultdict(list)
    for row in rows:
        by_conv[row["sample_id"]].append(row)
        by_category[row.get("category_name", "Unknown")].append(row)
    conv_f1s = [mean(r["f1"] for r in group) for group in by_conv.values()]
    answered = sum(r.get("answer_status") == "ok" for r in rows)
    judged = sum(r.get("judge_status") == "ok" for r in rows)
    completed = sum(r.get("status") == "ok" for r in rows)
    summary = {
        "run_status": "complete" if completed == len(rows) else "incomplete",
        "n_expected_questions": len(rows), "n_questions": len(rows),
        "n_answered": answered, "n_completed": completed, "n_failed": len(rows) - completed,
        "answer_coverage": answered / len(rows), "coverage": completed / len(rows),
        "judge_coverage": judged / len(rows) if run_judge else None,
        "failure_counts": dict(Counter(
            (r.get("error") or {}).get("stage", "judge" if r.get("judge_status") == "error" else "unknown")
            for r in rows if r.get("status") != "ok")),
        "metric_policy": "Fixed manifest denominator; answer failures contribute 0 to F1. "
                         "Incomplete runs are not valid completed benchmark scores. Judge failures remain null.",
        "overall_f1_mean": mean(conv_f1s), "overall_f1_std": pstdev(conv_f1s),
        "question_micro_f1": mean(r["f1"] for r in rows),
        "n_conversations": len(by_conv),
        "per_conversation": {sid: {"f1": mean(r["f1"] for r in group), "n_questions": len(group)}
                             for sid, group in by_conv.items()},
        "by_category": {},
    }
    if run_judge and judge_fn == "longmemeval":
        def score(group):
            return mean((r.get("longmemeval_correct") or 0) if r.get("judge_status") == "ok" else 0 for r in group)
        # Operational, failures-as-zero numbers have explicit names; primary accuracy
        # is unavailable until every requested answer and judge completed.
        summary["longmemeval_accuracy_failures_zero"] = score(rows)
        summary["longmemeval_accuracy"] = score(rows) if completed == len(rows) else None
        summary["longmemeval_task_avg_accuracy"] = mean(score(g) for g in by_category.values()) if completed == len(rows) else None
    for cat, group in sorted(by_category.items()):
        values = [r["f1"] for r in group]
        entry = {"f1_mean": mean(values), "f1_std": pstdev(values), "n": len(group),
                 "coverage": sum(r.get("status") == "ok" for r in group) / len(group)}
        if run_judge and judge_fn == "longmemeval":
            entry["accuracy"] = score(group) if all(r.get("status") == "ok" for r in group) else None
        summary["by_category"][cat] = entry
    return summary
