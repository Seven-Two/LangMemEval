"""Progress must report completed work, without counting a session twice."""
import io

import pytest

from agents_memory.console_progress import ConsoleProgress
from agents_memory import console_progress, diagnostics as log


@pytest.fixture
def bars(monkeypatch):
    created = []

    class Bar:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.total, self.n = kwargs["total"], 0
            self.closed = False
            self.postfix = ""
            created.append(self)

        def update(self, amount):
            assert not self.closed
            self.n += amount

        def set_postfix_str(self, text, **kwargs):
            self.postfix = text

        def close(self):
            self.closed = True

    monkeypatch.setattr(console_progress, "tqdm", Bar)
    return created


def test_message_and_session_events_do_not_double_count(bars):
    progress = ConsoleProgress(io.StringIO())
    progress.handle("write.plan", {"messages": 5, "method": "amem", "conversation": "c0"})
    progress.handle("amem.note.saved", {})
    progress.handle("amem.note.saved", {})
    progress.handle("write.progress", {"messages_done": 2})
    assert bars[0].n == 2
    progress.handle("write.progress", {"messages_done": 5})
    progress.handle("done", {"stage": "write"})
    assert bars[0].n == 5 and bars[0].closed
    progress.handle("write.plan", {"messages": 3, "conversation": "c1"})
    assert bars[1].n == 0 and bars[1].total == 3
    progress.close()


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_failure_or_interrupt_never_marks_unfinished_work_complete(bars, status):
    progress = ConsoleProgress(io.StringIO())
    progress.handle("write.plan", {"messages": 419})
    for _ in range(8):
        progress.handle("amem.note.saved", {})
    progress.handle(status, {"stage": "write"})
    assert bars[0].n == 8 and bars[0].closed
    assert progress.bar is None


def test_answer_progress_counts_failed_questions_as_processed(bars):
    progress = ConsoleProgress(io.StringIO())
    progress.handle("question.plan", {"questions": 2})
    progress.handle("question.result", {"question_index": 1, "answer_status": "error"})
    assert bars[0].n == 1 and bars[0].postfix == "errors=1"
    progress.handle("question.result", {"question_index": 2, "answer_status": "ok", "judge_status": "skipped"})
    progress.handle("question.done", {})
    assert bars[0].n == 2 and bars[0].postfix == "errors=1" and bars[0].closed


def test_waiting_updates_label_without_advancing_completed_work(bars):
    progress = ConsoleProgress(io.StringIO())
    progress.handle("write.plan", {"messages": 5})
    progress.handle("start", {"stage": "amem.evolve"})
    assert bars[0].postfix == "evolve"
    progress.handle("waiting", {"elapsed_s": 60.1})
    assert bars[0].postfix == "waiting 60s" and bars[0].n == 0
    progress.handle("llm.retry", {"attempt": 2})
    assert bars[0].postfix == "HTTP attempt 2" and bars[0].n == 0
    progress.close()


@pytest.mark.parametrize("mode", ["concise", "full"])
def test_logger_closes_progress_on_unexpected_exit_and_file_has_no_terminal_codes(bars, mode, tmp_path):
    path = tmp_path / "progress.log"
    with pytest.raises(RuntimeError):
        with log.run_logging(path, console_mode=mode):
            log.event("write.plan", messages=3)
            log.event("amem.note.saved", notes=1)
            raise RuntimeError("offline error outside a stage")
    assert bars[0].closed and bars[0].n == 1
    text = path.read_text(encoding="utf-8")
    assert 'status="amem.note.saved"' in text
    assert "\x1b" not in text and "\r" not in text
