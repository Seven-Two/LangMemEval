"""Safe output tolerance without guessing model intent or adding model calls."""
import copy
import json

import numpy as np
import pytest

from agents_memory import diagnostics as log
from langmem_eval.methods.amem import prompts
from langmem_eval.methods.amem.responses import prepare_response
from test_amem_contract import analysis, backend, decision, session
from test_amem_evolution_fallback import scripted_backend


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Output validation tests must never use network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")


def test_non_candidate_links_are_filtered_without_remapping_and_retrieval_continues(tmp_path):
    replies = [reply for _ in range(3) for reply in (analysis(), decision())]
    replies.extend([analysis(), decision(True, actions=["strengthen", "update_neighbor"],
        suggested_connections=[2, 1, -1, 999, 0, 2], tags_to_update=["new"],
        new_context_neighborhood=["changed-2", "changed-0"],
        new_tags_neighborhood=[["t2"], ["t0"]]), {"keywords": "new"}])
    obj, calls = scripted_backend(replies)
    obj.ingest(session("Berlin", "Shanghai", "Paris"))
    obj._search = lambda *args: [2, 0]
    path = tmp_path / "links.log"
    with log.run_logging(path):
        obj.ingest(session("new", name="session_2"))
    assert obj.notes[-1].links == [obj.notes[2].id, obj.notes[0].id, obj.notes[2].id]
    assert obj.notes[-1].tags == ["new"]
    assert obj.notes[2].context == "changed-2" and obj.notes[0].context == "changed-0"
    assert obj.notes[1].context == "residence"
    assert obj.evolution_count == 1 and len(calls) == 8
    text = path.read_text(encoding="utf-8")
    for field in ('candidate_indices=[2, 0]', 'returned_indices=[2, 1, -1, 999, 0, 2]',
                  'filtered_indices=[1, -1, 999]', 'kept_indices=[2, 0, 2]',
                  'response_id="response-8"'):
        assert field in text
    assert 'status="failed"' not in text
    request_id = obj.controller.last_response_metadata["request_id"]
    assert text.count(f'request_id="{request_id}"') >= 3
    description = obj.describe()
    assert description["link_policy"] == "filter_non_candidates"
    assert description["output_adjustments"]["filtered_links"] == 3
    assert description["output_adjustments"]["filtered_link_responses"] == 1
    description["output_adjustments"]["filtered_links"] = 999
    assert obj.describe()["output_adjustments"]["filtered_links"] == 3
    obj._search = lambda *args: [3]
    assert "Paris" in obj.retrieve("Where?", 3)[0]
    assert len(calls) == 9


@pytest.mark.parametrize("links", [[-1, 999], [0], []])
def test_no_neighbors_filters_all_links_and_next_message_still_saves(links):
    obj, llm, _ = backend([analysis(), decision(True, actions=["strengthen"],
        suggested_connections=links, tags_to_update=["updated"]), analysis(), decision()])
    obj.ingest(session("Berlin", "Shanghai"))
    assert len(obj.notes) == 2 and obj.notes[0].links == []
    assert obj.notes[0].tags == ["updated"] and len(llm.calls) == 4
    assert obj.output_adjustments["filtered_links"] == len(links)


@pytest.mark.parametrize("actions", [["unrecognized"], ["unrecognized", "strengthen"]])
def test_unknown_actions_do_not_block_known_actions_or_threshold(actions, tmp_path):
    obj, calls = scripted_backend([analysis(), decision(), analysis(), decision(True,
        actions=actions, suggested_connections=[0], tags_to_update=["new"])], threshold=1)
    with log.run_logging(tmp_path / "actions.log"):
        obj.ingest(session("Berlin", "Shanghai"))
    assert len(calls) == 4 and obj.evolution_count == 1
    assert obj.notes[-1].links == ([obj.notes[0].id] if "strengthen" in actions else [])
    assert obj.output_adjustments["ignored_actions"] == 1
    assert obj.output_adjustments["ignored_action_responses"] == 1
    assert len(obj.embedder.calls[-1]) == 2  # upstream counts should_evolve=true


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "prompt"])
def test_extra_fields_and_unused_action_fields_do_not_block_writing(mode, tmp_path):
    schemas_before = copy.deepcopy((prompts.ANALYSIS_SCHEMA, prompts.EVOLUTION_SCHEMA))
    replies = [analysis() | {"private-extra-key": "private-extra-value"},
               {"should_evolve": False, "actions": None, "suggested_connections": "unused"},
               analysis(), {"should_evolve": True, "actions": ["strengthen"],
                            "suggested_connections": [0], "tags_to_update": ["move"],
                            "new_tags_neighborhood": None}]
    original = copy.deepcopy(replies)
    obj, calls = scripted_backend(replies, mode=mode)
    path = tmp_path / "extras.log"
    with log.run_logging(path):
        obj.ingest(session("Berlin", "Shanghai"))
    assert obj.notes[-1].links == [obj.notes[0].id]
    assert len(calls) == 4 and replies == original
    assert schemas_before == (prompts.ANALYSIS_SCHEMA, prompts.EVOLUTION_SCHEMA)
    if mode == "json_schema":
        assert calls[1]["response_format"]["json_schema"]["schema"] == prompts.EVOLUTION_SCHEMA
    text = path.read_text(encoding="utf-8")
    assert text.count('status="amem.response.extra_fields_ignored"') == 1
    assert "private-extra" not in text


@pytest.mark.parametrize("value,field", [
    ({"should_evolve": "false"}, "should_evolve"),
    ({"should_evolve": True}, "actions"),
    ({"should_evolve": True, "actions": ["strengthen"]}, "suggested_connections"),
    (decision(True, actions=["strengthen"], suggested_connections=["0"]), "suggested_connections"),
    (decision(True, actions=["strengthen"], suggested_connections=[True]), "suggested_connections"),
    (decision(True, actions=["update_neighbor"], new_tags_neighborhood=["bad"]), "new_tags_neighborhood"),
])
def test_missing_or_malformed_active_fields_still_fail(value, field):
    with pytest.raises(ValueError, match=field):
        prepare_response(value, prompts.EVOLUTION_SCHEMA, purpose="evolution")


@pytest.mark.parametrize("schema,value", [(prompts.ANALYSIS_SCHEMA, analysis()),
                                         (prompts.QUERY_SCHEMA, {"keywords": "Berlin"})])
def test_complete_length_responses_are_accepted_for_analysis_and_query(schema, value):
    obj, calls = scripted_backend([(json.dumps(value), "length")])
    assert obj.controller.complete("test", schema) == value
    assert len(calls) == 1 and calls[0]["max_tokens"] == 1000


def test_filtering_does_not_hide_embedding_failure_or_commit_partial_updates():
    obj, _, encoder = backend([analysis(), decision(), analysis(), decision(True,
        actions=["strengthen", "update_neighbor"], suggested_connections=[0, -1],
        new_context_neighborhood=["changed"], new_tags_neighborhood=[["changed"]])])
    obj.ingest(session("Berlin"))
    before, vectors = obj.snapshot(), obj.vectors.copy()
    obj._search = lambda *args: [0]
    encoder.encode = lambda texts: np.array([[np.nan, 0.]])
    with pytest.raises(ValueError, match="finite"):
        obj.ingest(session("Shanghai"))
    assert obj.snapshot() == before and np.array_equal(obj.vectors, vectors)
    assert obj.output_adjustments["filtered_links"] == 1  # observed even on rollback
