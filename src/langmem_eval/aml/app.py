import logging
import os
import secrets
import sqlite3

from fastapi import Depends, FastAPI, HTTPException, Request

from .config import Settings
from .schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from .service import Conflict, MemoryService

logger = logging.getLogger(__name__)


def create_app(settings=None, service=None):
    from dotenv import load_dotenv
    load_dotenv()
    settings = settings or Settings.from_env()
    if service is None and (not settings.llm_key or not settings.embedding_key or not settings.embedding_url):
        raise ValueError("Set AML_LLM_API_KEY (or OPENAI_API_KEY), AML_EMBEDDING_API_KEY and AML_EMBEDDING_BASE_URL")
    service = service or MemoryService(settings)
    app = FastAPI(title="AML Textual Memory API", version=settings.version)

    def authenticate(request: Request):
        authorization = request.headers.get("Authorization", "")
        prefix, _, value = authorization.partition(" ")
        key = value if prefix.lower() in ("bearer", "token") else request.headers.get("X-Api-Key", "")
        if not secrets.compare_digest(key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Invalid memory API credential")

    def invoke(operation, payload):
        try:
            return operation(payload)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from None
        except sqlite3.OperationalError:
            raise HTTPException(503, "Storage temporarily unavailable; retry the request") from None
        except Exception:
            # Never serialize provider exceptions: they may contain credentials or history.
            logger.error("Memory operation failed; inspect provider/storage health")
            raise HTTPException(503, "Memory operation failed; retry the request") from None

    @app.get("/health")
    def health():
        return {"status": "ok", "version": settings.version}

    @app.post("/add", response_model=AddResponse, dependencies=[Depends(authenticate)])
    def add(payload: AddRequest):
        return invoke(service.add, payload)

    @app.post("/search", response_model=SearchResponse, response_model_exclude_none=True,
              dependencies=[Depends(authenticate)])
    def search(payload: SearchRequest):
        return invoke(service.search, payload)

    return app


def main():
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    uvicorn.run(create_app(), host=os.getenv("AML_HOST", "127.0.0.1"),
                port=int(os.getenv("AML_PORT", "8000")))
