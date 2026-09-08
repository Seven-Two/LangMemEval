import pytest

from langmem_eval.benchmark import build_memory, extract_sessions, memory_context


def fixture():
    return {"conversation": {
        "session_10": [{"speaker": "Bob", "text": "Later"}],
        "session_2": [{"speaker": "Alice", "text": "Earlier", "dia_id": "d2"}],
        "session_2_date_time": "2024-01-01",
    }, "qa": [{"question": "SECRET_QUESTION", "answer": "SECRET_ANSWER"}]}


def test_order_provenance_and_no_qa_leakage():
    sessions = extract_sessions(fixture())
    assert [s.id for s in sessions] == ["session_2", "session_10"]
    text = str(sessions)
    assert "Alice" in text and "2024-01-01" in text and "d2" in text
    assert "SECRET" not in text


def test_build_and_read_only_budget():
    class Backend:
        def __init__(self):
            self.sessions = []
        def ingest(self, session):
            self.sessions.append(session)
        def retrieve(self, query, limit):
            return ["abcdef", "ghijkl"][:limit]
    backend = Backend()
    build_memory(fixture(), backend)
    assert memory_context(backend, "q", top_k=2, max_chars=8) == "abcdef\ng"
    assert len(backend.sessions) == 2
    with pytest.raises(ValueError):
        memory_context(backend, "q", top_k=0, max_chars=8)


def test_empty_history_fails_explicitly():
    with pytest.raises(ValueError):
        extract_sessions({"conversation": {}})
