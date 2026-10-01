"""A-Mem's original JSON memory algorithm adapted to the unified evaluator.

Prompts/schemas: WujiangXu/A-mem at UPSTREAM_COMMIT (MIT).
The core follows memory_layer.py and query generation in test_advanced.py.
Intentional engineering differences are recorded in docs/amem.md.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import json
import math
import os
from uuid import uuid4

import numpy as np

from . import _amem_prompts as prompts
from .benchmark import Session
from .model_api import llm_extra_body, validate_base_url

UPSTREAM_COMMIT = "0c8039f28fdcc08189a23c07a3437d9d2482f9c2"
IMPLEMENTATION = "amem_original_json_unified_v1"
PROMPT_SHA256 = hashlib.sha256((prompts.ANALYSIS_PROMPT + prompts.EVOLUTION_PROMPT
                               + prompts.QUERY_PROMPT).encode()).hexdigest()


@dataclass(frozen=True)
class AMemSettings:
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dims: int | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_batch_size: int = 10
    embedding_revision: str | None = None
    device: str = "cpu"
    local_files_only: bool = False
    evolution_threshold: int = 100
    neighbor_k: int = 5
    response_format: str = "json_schema"
    temperature: float = 0.7
    max_output_tokens: int = 1000
    extra_body: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.embedding_provider not in {"local", "openai"}:
            raise ValueError("AMEM_EMBEDDING_PROVIDER must be local or openai")
        if self.response_format not in {"json_schema", "json_object", "prompt"}:
            raise ValueError("AMEM_RESPONSE_FORMAT must be json_schema, json_object or prompt")
        if not self.embedding_model.strip():
            raise ValueError("AMEM_EMBEDDING_MODEL is required")
        for n in (self.embedding_batch_size, self.evolution_threshold, self.neighbor_k, self.max_output_tokens):
            if type(n) is not int or n < 1:
                raise ValueError("A-Mem count/budget settings must be positive integers")
        if self.embedding_dims is not None and (type(self.embedding_dims) is not int or self.embedding_dims < 1):
            raise ValueError("AMEM_EMBEDDING_DIMS must be positive")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("AMEM_TEMPERATURE must be between 0 and 2")
        validate_base_url(self.embedding_base_url)

    @classmethod
    def from_env(cls):
        def flag(name):
            value = os.getenv(name, "false").lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError(f"{name} must be true or false")
            return value in {"true", "1"}
        provider = os.getenv("AMEM_EMBEDDING_PROVIDER", "local")
        model = os.getenv("AMEM_EMBEDDING_MODEL") or (
            "sentence-transformers/all-MiniLM-L6-v2" if provider == "local" else "")
        dims = os.getenv("AMEM_EMBEDDING_DIMS")
        return cls(
            embedding_provider=provider, embedding_model=model,
            embedding_dims=int(dims) if dims else None,
            embedding_base_url=os.getenv("AMEM_EMBEDDING_BASE_URL") or None,
            embedding_api_key=os.getenv("AMEM_EMBEDDING_API_KEY") or None,
            embedding_batch_size=int(os.getenv("AMEM_EMBEDDING_BATCH_SIZE", "10")),
            embedding_revision=os.getenv("AMEM_EMBEDDING_REVISION") or None,
            device=os.getenv("AMEM_DEVICE", "cpu"), local_files_only=flag("AMEM_LOCAL_FILES_ONLY"),
            evolution_threshold=int(os.getenv("AMEM_EVOLUTION_THRESHOLD", "100")),
            neighbor_k=int(os.getenv("AMEM_NEIGHBOR_K", "5")),
            response_format=os.getenv("AMEM_RESPONSE_FORMAT", "json_schema"),
            temperature=float(os.getenv("AMEM_TEMPERATURE", "0.7")),
            max_output_tokens=int(os.getenv("AMEM_MAX_OUTPUT_TOKENS", "1000")),
            extra_body=llm_extra_body(),
        )

    def public_config(self):
        result = asdict(self)
        result.pop("embedding_api_key")
        # Extra payloads can contain private service options: record known sampler options only.
        result["extra_body"] = {k: v for k, v in self.extra_body.items()
                                if k in {"enable_thinking", "top_k", "repetition_penalty", "reasoning_effort"}}
        return result


def _validate(value, schema):
    """Validate the small, fixed upstream JSON schemas even in prompt-only mode."""
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": type(value) is bool,
             "integer": type(value) is int}
    if kind in valid and not valid[kind]:
        raise ValueError(f"A-Mem response must have type {kind}")
    if kind == "object":
        if any(k not in value for k in schema.get("required", [])):
            raise ValueError("A-Mem response is missing required fields")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError("A-Mem response contains unknown fields")
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key])
    elif kind == "array":
        for item in value:
            _validate(item, schema["items"])


class OpenAIController:
    def __init__(self, model: str, settings: AMemSettings, *, client=None):
        from openai import OpenAI
        validate_base_url(os.getenv("OPENAI_BASE_URL"))
        self.client = client if client is not None else OpenAI()
        self.model, self.settings = model, settings

    def complete(self, prompt: str, schema: dict) -> dict:
        mode = self.settings.response_format
        kwargs = {}
        if mode == "json_schema":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "response", "schema": schema, "strict": True}}
        elif mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.extra_body:
            kwargs["extra_body"] = self.settings.extra_body
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": "You must respond with a JSON object."},
                                        {"role": "user", "content": prompt}],
            temperature=self.settings.temperature, max_tokens=self.settings.max_output_tokens, **kwargs)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("A-Mem structured response was truncated; increase AMEM_MAX_OUTPUT_TOKENS")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("A-Mem received an empty structured response")
        content = content.strip()
        if content.startswith("```") and content.endswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("A-Mem received invalid JSON") from None
        _validate(result, schema)
        return result


@lru_cache(maxsize=2)
def _load_local_model(name, revision, device, local_files_only):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Local A-Mem embeddings need: uv sync --locked --extra dev --extra amem") from None
    return SentenceTransformer(name, revision=revision, device=device, local_files_only=local_files_only)


class LocalEmbedder:
    def __init__(self, settings):
        self.settings = settings
        self.model = _load_local_model(settings.embedding_model, settings.embedding_revision,
                                       settings.device, settings.local_files_only)

    def encode(self, texts):
        from agents_memory.usage import record_external_usage
        failed = True
        try:
            result = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            failed = False
            return result
        finally:
            record_external_usage(provider="sentence_transformers_local", model=self.settings.embedding_model,
                                  kind="embedding", failed=failed)


class APIEmbedder:
    def __init__(self, settings: AMemSettings, *, client=None):
        from openai import OpenAI
        self.settings = settings
        if client is None:
            if not settings.embedding_base_url or not settings.embedding_api_key:
                raise ValueError("Set AMEM_EMBEDDING_BASE_URL and AMEM_EMBEDDING_API_KEY for API embeddings")
            client = OpenAI(api_key=settings.embedding_api_key, base_url=settings.embedding_base_url)
        self.client = client

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.settings.embedding_dims or 0))
        vectors = []
        for offset in range(0, len(texts), self.settings.embedding_batch_size):
            chunk = texts[offset:offset + self.settings.embedding_batch_size]
            kwargs = {"dimensions": self.settings.embedding_dims} if self.settings.embedding_dims else {}
            result = self.client.embeddings.create(model=self.settings.embedding_model, input=chunk,
                                                    encoding_format="float", **kwargs)
            ordered = sorted(result.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(chunk))):
                raise ValueError("Embedding service returned missing or duplicate vector indices")
            vectors.extend(item.embedding for item in ordered)
        return np.asarray(vectors, dtype=float)


@dataclass
class Note:
    id: str
    source_id: str
    session_id: str
    timestamp: str
    content: str
    keywords: list[str]
    context: str
    tags: list[str]
    links: list[str] = field(default_factory=list)

    def indexed_text(self, consolidated=False):
        if consolidated:
            return f"{self.content} , {self.context} {' '.join(self.keywords)} {' '.join(self.tags)}"
        return ("content:" + self.content + " context:" + self.context + " keywords: "
                + ", ".join(self.keywords) + " tags: " + ", ".join(self.tags))

    def record(self):
        return (f"talk start time:{self.timestamp}memory content: {self.content}"
                f"memory context: {self.context}memory keywords: {self.keywords}"
                f"memory tags: {self.tags}")


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
