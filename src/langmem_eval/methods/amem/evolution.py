"""Pure note/graph transition: no models, vectors, logging or global settings.

Inputs are validated by responses.py. The backend commits this proposed state
only after embedding succeeds, so failed writes cannot partially mutate memory.
"""
from copy import deepcopy
from dataclasses import dataclass, field

from .memory import Note


@dataclass
class EvolutionUpdate:
    notes: list[Note]
    should_evolve: bool
    diagnostics: list[tuple[str, dict]] = field(default_factory=list)
    adjustments: dict[str, int] = field(default_factory=dict)


def apply_evolution(notes: list[Note], note: Note, decision: dict | None,
                    candidates: list[int]) -> EvolutionUpdate:
    updated = deepcopy(notes)
    note = deepcopy(note)
    should_evolve = decision is not None and decision["should_evolve"]
    result = EvolutionUpdate(updated, should_evolve)
    if should_evolve:
        actions = decision["actions"]
        unknown = [action for action in actions if action not in {"strengthen", "update_neighbor"}]
        if unknown:
            result.adjustments.update(ignored_action_responses=1, ignored_actions=len(unknown))
            result.diagnostics.append(("amem.actions.ignored", {
                "count": len(unknown), "policy": "upstream_ignore"}))
        if "strengthen" in actions:
            links = decision["suggested_connections"]
            valid = [index for index in links if index in candidates]
            invalid = [index for index in links if index not in candidates]
            if invalid:
                result.adjustments.update(filtered_link_responses=1, filtered_links=len(invalid))
                result.diagnostics.append(("amem.links.filtered", {
                    "policy": "filter_non_candidates", "candidate_indices": list(candidates),
                    "returned_indices": list(links), "kept_indices": valid, "filtered_indices": invalid}))
            # Preserve returned order/duplicates; never reinterpret local positions.
            note.links = [updated[index].id for index in valid]
            note.tags = list(decision["tags_to_update"])
        if "update_neighbor" in actions:
            contexts, tags = decision["new_context_neighborhood"], decision["new_tags_neighborhood"]
            applied = min(len(candidates), len(tags))
            if len(contexts) != len(candidates) or len(tags) != len(candidates):
                result.diagnostics.append(("amem.neighbor_updates.length_mismatch", {
                    "policy": "upstream_positional_prefix", "neighbors": len(candidates),
                    "context_items": len(contexts), "tag_items": len(tags), "applied_neighbors": applied,
                    "ignored_context_items": max(0, len(contexts) - applied),
                    "ignored_tag_items": max(0, len(tags) - applied),
                    "preserved_contexts": max(0, applied - len(contexts))}))
            for offset in range(applied):
                updated[candidates[offset]].tags = list(tags[offset])
                if offset < len(contexts):
                    updated[candidates[offset]].context = contexts[offset]
    updated.append(note)
    return result
