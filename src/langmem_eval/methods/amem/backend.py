"""A-Mem note construction, evolution and retrieval under the unified protocol.

Adapted from WujiangXu/A-mem (MIT) at UPSTREAM_COMMIT. Intentional differences
are documented in docs/amem.md; model transport and prompts are separate modules.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from uuid import uuid4

import numpy as np

from ...interfaces import Session
from ...model_api import validate_base_url
from . import prompts
from ...embeddings import APIEmbedder, LocalEmbedder
from .clients import OpenAIController, _validate
from .config import AMemSettings
from .memory import Note

UPSTREAM_COMMIT = "0c8039f28fdcc08189a23c07a3437d9d2482f9c2"
IMPLEMENTATION = "amem_original_json_unified_v1"
PROMPT_SHA256 = hashlib.sha256((prompts.ANALYSIS_PROMPT + prompts.EVOLUTION_PROMPT
                               + prompts.QUERY_PROMPT).encode()).hexdigest()

class AMemBackend:
    def __init__(self, model: str, *, settings=None, controller=None, embedder=None):
        self.settings = settings if settings is not None else AMemSettings.from_env()
        # Validate configuration before loading weights or making requests.
        validate_base_url(os.getenv("OPENAI_BASE_URL"))
        self.controller = controller if controller is not None else OpenAIController(model, self.settings)
        self.embedder = embedder if embedder is not None else (
            LocalEmbedder(self.settings) if self.settings.embedding_provider == "local" else APIEmbedder(self.settings))
        self.model = model
        self.notes: list[Note] = []
        self.vectors = None
        self.evolution_count = 0
        self.last_retrieval = None

    def describe(self):
        return {"implementation": IMPLEMENTATION, "upstream_commit": UPSTREAM_COMMIT,
                "prompt_sha256": PROMPT_SHA256, "llm_model": self.model,
                "settings": self.settings.public_config(), "note_count": len(self.notes),
                "embedding_dimensions_actual": self.vectors.shape[1] if self.vectors is not None else None,
                "retrieval_unit": "seed_plus_linked_notes", "answer_protocol": "framework_unified"}

    def snapshot(self):
        return {"notes": [asdict(note) for note in self.notes], "evolution_count": self.evolution_count}

    def _complete(self, prompt, schema):
        result = self.controller.complete(prompt, schema)
        _validate(result, schema)
        return result

    def _encode(self, texts):
        vectors = np.asarray(self.embedder.encode(texts), dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
            raise ValueError("Invalid A-Mem embedding shape")
        if not np.isfinite(vectors).all():
            raise ValueError("A-Mem embeddings must be finite")
        dims = self.vectors.shape[1] if self.vectors is not None else self.settings.embedding_dims
        if dims is not None and vectors.shape[1] != dims:
            raise ValueError("A-Mem embedding dimension changed or mismatches configuration")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("A-Mem embeddings must be nonzero")
        return vectors

    def _search(self, query, k):
        if not self.notes:
            return []
        vector = self._encode([query])[0]
        scores = (self.vectors @ vector) / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(vector))
        # Same reverse argsort cosine ranking used by upstream SimpleEmbeddingRetriever.
        return np.argsort(scores)[-min(k, len(self.notes)):][::-1].tolist()

    def ingest(self, session: Session):
        for index, message in enumerate(session.messages):
            try:
                payload = json.loads(message["content"])
            except (KeyError, TypeError, json.JSONDecodeError):
                raise ValueError("A-Mem expects normalized historical messages") from None
            if not isinstance(payload, dict) or not all(isinstance(payload.get(k), str) for k in ("speaker", "text")):
                raise ValueError("A-Mem history must contain speaker and text")
            # Preserve the original evaluation harness's turn rendering (including spacing).
            content = "Speaker " + payload["speaker"] + "says : " + payload["text"]
            self._add_note(content, session.date, session.id, str(payload.get("source_id", f"{session.id}:{index}")))

    def _add_note(self, content, timestamp, session_id, source_id):
        metadata = self._complete(prompts.ANALYSIS_PROMPT + content, prompts.ANALYSIS_SCHEMA)
        note = Note(uuid4().hex, source_id, session_id, timestamp, content, metadata["keywords"],
                    metadata["context"] or "General", metadata["tags"])
        indices = self._search(content, self.settings.neighbor_k)
        neighbors = "".join(f"memory index:{i}\t talk start time:{self.notes[i].timestamp}"
                            f"\t memory content: {self.notes[i].content}\t memory context: {self.notes[i].context}"
                            f"\t memory keywords: {self.notes[i].keywords}\t memory tags: {self.notes[i].tags}\n"
                            for i in indices)
        evolution = self._complete(prompts.EVOLUTION_PROMPT.format(context=note.context, content=note.content,
            keywords=note.keywords, nearest_neighbors_memories=neighbors, neighbor_number=len(indices)),
            prompts.EVOLUTION_SCHEMA)
        # Atomic note write: no memory/index changes become visible before all steps succeed.
        updated = deepcopy(self.notes)
        count = self.evolution_count
        if evolution["should_evolve"]:
            if any(a not in {"strengthen", "update_neighbor"} for a in evolution["actions"]):
                raise ValueError("A-Mem returned an unknown evolution action")
            if "strengthen" in evolution["actions"]:
                links = evolution["suggested_connections"]
                if any(i not in indices for i in links):
                    raise ValueError("A-Mem returned a link outside the supplied neighbors")
                note.links = [updated[i].id for i in links]
                note.tags = evolution["tags_to_update"]
            if "update_neighbor" in evolution["actions"]:
                contexts, tags = evolution["new_context_neighborhood"], evolution["new_tags_neighborhood"]
                if len(contexts) > len(indices) or len(tags) > len(indices):
                    raise ValueError("A-Mem returned more neighbor updates than supplied neighbors")
                # Upstream accepts a shorter update list from small models.
                for offset in range(min(len(indices), len(tags))):
                    updated[indices[offset]].tags = tags[offset]
                    if offset < len(contexts):
                        updated[indices[offset]].context = contexts[offset]
            count += 1
        updated.append(note)
        if evolution["should_evolve"] and count % self.settings.evolution_threshold == 0:
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
        generated = self._complete(prompts.QUERY_PROMPT.format(question=query), prompts.QUERY_SCHEMA)["keywords"]
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
        return records
