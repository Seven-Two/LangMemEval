import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .schemas import AddResponse, SearchResponse


class Conflict(Exception):
    pass


class MemoryService:
    def __init__(self, settings, factory=None):
        if factory is None:
            from ..registry import discover_methods
            spec = discover_methods().get(settings.method)
            if spec is None or spec.aml_factory is None:
                raise ValueError(f"Method {settings.method!r} has no AML snapshot backend")
            factory = spec.aml_factory
        self.settings, self.factory = settings, factory
        Path(settings.database).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(settings.database) as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS states (user_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY,
                    digest TEXT NOT NULL, response TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            ''')
            signature = json.dumps([settings.version, settings.method, settings.llm_model, settings.llm_url,
                settings.embedding_model, settings.embedding_dims, settings.embedding_url])
            db.execute("INSERT OR IGNORE INTO config VALUES (1, ?)", (signature,))
            if db.execute("SELECT value FROM config WHERE id=1").fetchone()[0] != signature:
                raise ValueError("Database configuration differs; use a new database for a new version")

    @contextmanager
    def transaction(self):
        # LangMem may invoke embedding work on worker threads.
        db = sqlite3.connect(self.settings.database, timeout=1, check_same_thread=False)
        try:
            # Serializes writes/searches across processes, including model work.
            # Busy clients receive 503 and may retry; intended for low concurrency.
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def backend(self, db, user_id):
        row = db.execute("SELECT snapshot FROM states WHERE user_id=?", (user_id,)).fetchone()
        return self.factory(self.settings, user_id, json.loads(row[0]) if row else [], db)

    def add(self, request):
        payload = json.dumps(request.model_dump(), sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.transaction() as db:
            previous = db.execute("SELECT digest,response FROM requests WHERE request_id=?",
                                  (request.request_id,)).fetchone()
            if previous:
                if previous[0] != digest:
                    raise Conflict("request_id reused with a different payload")
                return AddResponse.model_validate_json(previous[1])
            backend = self.backend(db, request.user_id)
            backend.add(request)
            response = AddResponse(request_id=request.request_id, user_id=request.user_id,
                                   session_id=request.session_id)
            db.execute("INSERT OR REPLACE INTO states VALUES (?,?)",
                       (request.user_id, json.dumps(backend.snapshot(), ensure_ascii=False)))
            db.execute("INSERT INTO requests VALUES (?,?,?,?)",
                       (request.request_id, digest, response.model_dump_json(), payload))
        return response

    def search(self, request):
        with self.transaction() as db:
            if not db.execute("SELECT 1 FROM states WHERE user_id=?", (request.user_id,)).fetchone():
                return SearchResponse(data=[])
            records = self.backend(db, request.user_id).search(request)
            return SearchResponse(data=records[:request.top_k])
