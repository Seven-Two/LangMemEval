"""Terminal-only progress derived from existing evaluation lifecycle events."""
from tqdm import tqdm


class ConsoleProgress:
    """One detail bar below the runner's conversation bar; no model side effects."""

    def __init__(self, stream):
        self.stream = stream
        self.bar = None
        self.phase = None
        self.errors = 0

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        self.phase = None

    def _start(self, phase, total, fields):
        self.close()
        self.phase, self.errors = phase, 0
        label = "Write memory" if phase == "write" else "Answer questions"
        owner = " ".join(str(fields[key]) for key in ("method", "conversation") if key in fields)
        self.bar = tqdm(total=total, desc=f"{owner} | {label}".strip(" |"),
                        unit="msg" if phase == "write" else "q", position=1,
                        leave=False, dynamic_ncols=True, mininterval=0.5,
                        file=self.stream, disable=None)

    def _advance_to(self, completed):
        # Session totals and per-note notifications describe the same work.
        # Keep an absolute monotonic count so they cannot double-count it.
        target = min(self.bar.total, max(self.bar.n, completed))
        self.bar.update(target - self.bar.n)

    def handle(self, status, fields):
        if status == "write.plan":
            self._start("write", fields["messages"], fields)
        elif status == "question.plan":
            self._start("answer", fields["questions"], fields)
        if self.bar is None:
            return

        if self.phase == "write" and status == "amem.note.saved":
            self._advance_to(self.bar.n + 1)
        elif self.phase == "write" and status == "write.progress":
            self._advance_to(fields["messages_done"])
        elif self.phase == "answer" and status == "question.result":
            self.errors += int(fields.get("answer_status") == "error" or fields.get("judge_status") == "error")
            self.bar.set_postfix_str(f"errors={self.errors}", refresh=False)
            self._advance_to(fields["question_index"])
        elif status == "start":
            label = {
                "write.session": fields.get("session", "session"),
                "amem.analyze": "analyze", "amem.evolve": "evolve",
                "retrieve": "retrieve", "answer.api": "answer", "judge": "judge",
            }.get(fields.get("stage"))
            if label:
                self.bar.set_postfix_str(label, refresh=True)
        elif status == "waiting":
            self.bar.set_postfix_str(f"waiting {fields.get('elapsed_s', 0):.0f}s", refresh=True)
        elif status == "llm.retry":
            self.bar.set_postfix_str(f"HTTP attempt {fields['attempt']}", refresh=True)

        if status == "question.done" or (
            status in {"done", "failed", "stopped"} and
            (fields.get("stage") in {"conversation", "run"} or
             self.phase == "write" and fields.get("stage") == "write")
        ):
            # Never force 100% on errors or interruption.
            self.close()
