"""A-Mem note construction, evolution and retrieval under the unified protocol.

Adapted from WujiangXu/A-mem (MIT) at UPSTREAM_COMMIT. Intentional differences
are documented in docs/amem.md; model transport and prompts are separate modules.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import os
from uuid import uuid4

import numpy as np
from agents_memory.diagnostics import event, stage

from ...interfaces import Session
from ...model_api import validate_base_url
from . import prompts
from ...embeddings import create_embedder, validate_vectors
from .clients import OpenAIController
from .responses import InvalidJSONResponse, prepare_response
from .evolution import apply_evolution
from .config import AMemSettings
from .memory import Note

UPSTREAM_COMMIT = "0c8039f28fdcc08189a23c07a3437d9d2482f9c2"
IMPLEMENTATION = "amem_original_json_unified_v4"
PROMPT_SHA256 = hashlib.sha256((prompts.ANALYSIS_PROMPT + prompts.EVOLUTION_PROMPT
                               + prompts.QUERY_PROMPT).encode()).hexdigest()

class AMemBackend:
    def __init__(self, model: str, *, settings=None, controller=None, embedder=None):
        self.settings = settings if settings is not None else AMemSettings.from_env()
        # Validate configuration before loading weights or making requests.
        validate_base_url(os.getenv("OPENAI_BASE_URL"))
        with ExitStack() as resources:
            self.controller = controller if controller is not None else OpenAIController(model, self.settings)
            if controller is None:
                resources.callback(self.controller.close)
            self.embedder = embedder if embedder is not None else create_embedder(self.settings)
            close = getattr(self.embedder, "close", None)
            if embedder is None and callable(close):
                resources.callback(close)
            self._resources = resources.pop_all()
        self.model = model
        self.notes: list[Note] = []
        self.vectors = None
        self.evolution_count = 0
        self.evolution_stats = {"attempts": 0, "truncated": 0, "skipped_invalid_json": 0}
        self.output_adjustments = {"filtered_link_responses": 0, "filtered_links": 0,
                                   "ignored_action_responses": 0, "ignored_actions": 0}
        self.last_retrieval = None

    def describe(self):
        return {"implementation": IMPLEMENTATION, "upstream_commit": UPSTREAM_COMMIT,
                "prompt_sha256": PROMPT_SHA256, "llm_model": self.model,
                "settings": self.settings.public_config(), "note_count": len(self.notes),
                "neighbor_update_policy": "upstream_positional_prefix",
                "evolution_failure_policy": "upstream_skip_invalid_json",
                "evolution_stats": dict(self.evolution_stats),
                "link_policy": "filter_non_candidates",
                "unknown_action_policy": "upstream_ignore",
                "response_validation_policy": "active_fields_ignore_extras_parse_length",
                "output_adjustments": dict(self.output_adjustments),
                "embedding_dimensions_actual": self.vectors.shape[1] if self.vectors is not None else None,
                "retrieval_unit": "seed_plus_linked_notes", "answer_protocol": "framework_unified"}

    def snapshot(self):
        return {"notes": [asdict(note) for note in self.notes], "evolution_count": self.evolution_count}

    def _complete(self, prompt, schema, *, purpose):
        result = self.controller.complete(prompt, schema, purpose=purpose)
        prepared = prepare_response(result, schema, purpose=purpose)
        if prepared.ignored_extra_fields:
            event("amem.response.extra_fields_ignored", count=prepared.ignored_extra_fields,
                  **getattr(self.controller, "last_response_metadata", {}))
        return prepared.data

    def close(self):
        self._resources.close()

    def _encode(self, texts):
        dims = self.vectors.shape[1] if self.vectors is not None else self.settings.embedding_dims
        return validate_vectors(self.embedder.encode(texts), rows=len(texts), dims=dims)

    def _evolve(self, note, neighbors, neighbor_number):
        self.evolution_stats["attempts"] += 1
        with stage("amem.evolve", model=self.model, neighbors=neighbor_number):
            try:
                return self._complete(prompts.EVOLUTION_PROMPT.format(context=note.context,
                    content=note.content, keywords=note.keywords,
                    nearest_neighbors_memories=neighbors, neighbor_number=neighbor_number),
                    prompts.EVOLUTION_SCHEMA, purpose="evolution")
            except InvalidJSONResponse:
                self.evolution_stats["skipped_invalid_json"] += 1
                event("amem.evolution.skipped", reason="invalid_json",
                      policy="upstream_skip_invalid_json",
                      **getattr(self.controller, "last_response_metadata", {}))
                return None
            finally:
                metadata = getattr(self.controller, "last_response_metadata", {})
                if metadata.get("finish_reason") == "length":
                    self.evolution_stats["truncated"] += 1
                event("amem.evolution.stats", **self.evolution_stats)

    def _search(self, query, k):
        if not self.notes:
            return []
        vector = self._encode([query])[0]
        scores = (self.vectors @ vector) / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(vector))
        # Same reverse argsort cosine ranking used by upstream SimpleEmbeddingRetriever.
        return np.argsort(scores)[-min(k, len(self.notes)):][::-1].tolist()

    def ingest(self, session: Session):
        for index, turn in enumerate(session.turns()):
            # Preserve the original evaluation harness's turn rendering.
            content = "Speaker " + turn.speaker + "says : " + turn.text
            with stage("amem.write.note", message_index=index + 1, message_total=len(session.messages),
                       session=session.id, notes_before=len(self.notes)):
                self._add_note(content, session.date, session.id, turn.source_id)
                event("amem.note.saved", notes=len(self.notes), evolutions=self.evolution_count)

    def _add_note(self, content, timestamp, session_id, source_id):
        with stage("amem.analyze", model=self.model):
            metadata = self._complete(prompts.ANALYSIS_PROMPT + content, prompts.ANALYSIS_SCHEMA, purpose="analysis")
        note = Note(uuid4().hex, source_id, session_id, timestamp, content, metadata["keywords"],
                    metadata["context"] or "General", metadata["tags"])
        with stage("amem.neighbors", limit=self.settings.neighbor_k):
            indices = self._search(content, self.settings.neighbor_k)
        neighbors = "".join(f"memory index:{i}\t talk start time:{self.notes[i].timestamp}"
                            f"\t memory content: {self.notes[i].content}\t memory context: {self.notes[i].context}"
                            f"\t memory keywords: {self.notes[i].keywords}\t memory tags: {self.notes[i].tags}\n"
                            for i in indices)
        evolution = self._evolve(note, neighbors, len(indices))
        should_evolve = evolution is not None and evolution["should_evolve"]
        event("amem.evolution.decision", should_evolve=should_evolve, skipped=evolution is None)
        proposed = apply_evolution(self.notes, note, evolution, indices)
        for key, amount in proposed.adjustments.items():
            self.output_adjustments[key] += amount
        for name, fields in proposed.diagnostics:
            event(name, **fields, **getattr(self.controller, "last_response_metadata", {}))
        updated = proposed.notes
        note = updated[-1]
        count = self.evolution_count + int(proposed.should_evolve)
        # Commit memory/index state only after all embedding work succeeds.
        if should_evolve and count % self.settings.evolution_threshold == 0:
            with stage("amem.index.rebuild", notes=len(updated)):
                vectors = self._encode([n.indexed_text(consolidated=True) for n in updated])
        else:
            new_vector = self._encode([note.indexed_text()])
            vectors = new_vector if self.vectors is None else np.vstack([self.vectors, new_vector])
        self.notes, self.vectors, self.evolution_count = updated, vectors, count

    def retrieve(self, query: str, limit: int) -> list[str]:
        if limit < 1:
            raise ValueError("A-Mem retrieval limit must be positive")
        self.last_retrieval = {"question": query, "query": None, "seed_ids": [], "groups": []}
        if not self.notes:
            return []
        with stage("amem.query.rewrite", model=self.model):
            generated = self._complete(prompts.QUERY_PROMPT.format(question=query), prompts.QUERY_SCHEMA, purpose="query")["keywords"]
        if not generated.strip():
            raise ValueError("A-Mem query generator returned empty keywords")
        indices = self._search(generated, limit)
        by_id = {n.id: n for n in self.notes}
        records, groups = [], []
        for i in indices:
            seed = self.notes[i]
            # One record is an upstream seed plus its linked neighbors. The evaluator
            # applies its token budget to whole groups, retaining A-Mem's graph expansion.
            members = [seed] + [by_id[key] for key in seed.links[:limit]]
            records.append("\n".join(n.record() for n in members))
            groups.append([n.id for n in members])
        self.last_retrieval = {"question": query, "query": generated,
                               "seed_ids": [self.notes[i].id for i in indices], "groups": groups}
        event("amem.search.done", seeds=len(indices), groups=len(groups),
              linked_notes=sum(len(group) - 1 for group in groups))
        return records
