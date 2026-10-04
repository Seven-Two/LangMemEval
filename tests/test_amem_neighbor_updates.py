"""Pinned upstream behavior for variable-length positional neighbor updates."""
import copy

import numpy as np
import pytest

from agents_memory import diagnostics as log
from langmem_eval.methods.amem import prompts
from test_amem_contract import analysis, backend, decision, session


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Neighbor update tests must never use network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize("contexts,tags,expected", [
    (["c0", "c1", "extra"], [["t0"], ["t1"], ["extra"]],
     [("c0", ["t0"]), ("c1", ["t1"])]),
    (["c0"], [["t0"], ["t1"], ["extra"]],
     [("c0", ["t0"]), ("old-0", ["t1"])]),
    (["c0", "c1", "extra"], [["t0"]],
     [("c0", ["t0"]), ("old-0", ["personal"])]),
    (["unused"], [], [("old-2", ["personal"]), ("old-0", ["personal"])]),
    ([], [["t0"], ["t1"]], [("old-2", ["t0"]), ("old-0", ["t1"])]),
])
def test_updates_follow_candidate_order_and_ignore_excess_without_extra_calls(contexts, tags, expected, tmp_path):
    replies = [reply for i in range(3) for reply in (analysis(f"old-{i}"), decision())]
    replies += [analysis("new"), decision(True, actions=["update_neighbor"],
        new_context_neighborhood=contexts, new_tags_neighborhood=tags)]
    obj, llm, _ = backend(replies)
    obj.ingest(session("Berlin", "Shanghai", "Paris"))
    before = obj.snapshot()
    obj._search = lambda query, k: [2, 0]  # candidate order is NOT global index order
    path = tmp_path / "neighbors.log"
    with log.run_logging(path):
        obj.ingest(session("new-message", name="session_2"))
    assert len(obj.notes) == 4 and len(llm.calls) == 8
    assert obj.evolution_count == 1
    for index, (context, note_tags) in zip([2, 0], expected):
        assert obj.notes[index].context == context and obj.notes[index].tags == note_tags
        assert obj.notes[index].content == before["notes"][index]["content"]
    assert obj.snapshot()["notes"][1] == before["notes"][1]
    assert obj.describe()["neighbor_update_policy"] == "upstream_positional_prefix"
    text = path.read_text(encoding="utf-8")
    assert 'status="amem.neighbor_updates.length_mismatch"' in text
    assert f'context_items={len(contexts)}' in text and f'tag_items={len(tags)}' in text
    assert 'status="failed"' not in text
    assert "new-message" not in text  # no memory body in diagnostics


def test_excess_updates_with_no_neighbors_save_only_the_new_note():
    schema = copy.deepcopy(prompts.EVOLUTION_SCHEMA)
    obj, llm, _ = backend([analysis(), decision(True, actions=["update_neighbor"],
        new_context_neighborhood=["invented"], new_tags_neighborhood=[["invented"]])])
    obj.ingest(session("Berlin"))
    assert len(obj.notes) == 1 and obj.notes[0].context == "residence"
    assert obj.notes[0].tags == ["personal"] and len(llm.calls) == 2
    assert llm.calls[-1][1] == schema == prompts.EVOLUTION_SCHEMA


def test_failed_insertion_after_oversized_updates_still_rolls_back():
    obj, _, encoder = backend([analysis(), decision(), analysis(), decision(True,
        actions=["update_neighbor"], new_context_neighborhood=["changed", "extra"],
        new_tags_neighborhood=[["changed"], ["extra"]])])
    obj.ingest(session("Berlin"))
    before, vectors = obj.snapshot(), obj.vectors.copy()
    obj._search = lambda query, k: [0]
    encoder.encode = lambda texts: np.array([[np.nan, 0.]])
    with pytest.raises(ValueError, match="finite"):
        obj.ingest(session("Shanghai"))
    assert obj.snapshot() == before and np.array_equal(obj.vectors, vectors)
