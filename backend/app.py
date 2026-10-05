"""
Veritas AI - FastAPI service.

Run:  uvicorn app:app --reload --port 8000   (from the backend/ directory)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_engine import RAGEngine, NoExtractableTextError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("veritas.api")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class Source(BaseModel):
    filename: str
    page: int
    snippet: str
    distance: float


class DistanceMetric(BaseModel):
    name: str
    note: str
    top_k: int
    threshold: float


class ChatResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[Source]
    distance_metric: DistanceMetric


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int


# --------------------------------------------------------------------------- #
# App lifecycle
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    app.state.engine = RAGEngine()  # loads the embedding model once, at startup
    yield


app = FastAPI(title="Veritas AI", version="1.0.0", lifespan=lifespan)

# Needed only when the frontend is served from a different origin (e.g. python -m http.server).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


def engine_of(request: Request) -> RAGEngine:
    return request.app.state.engine


# --------------------------------------------------------------------------- #
# Routes (sync handlers run in FastAPI's threadpool, so blocking ML calls
# don't stall the event loop)
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
def upload(request: Request, file: UploadFile = File(...)):
    name = Path(file.filename or "").name  # strips any client-supplied path
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            size = 0
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File exceeds the 25 MB limit.")
                tmp.write(block)

        with open(tmp_path, "rb") as fh:
            if fh.read(5) != b"%PDF-":
                raise HTTPException(400, "File is not a valid PDF.")

        return engine_of(request).ingest_pdf(tmp_path, display_name=name)

    except HTTPException:
        raise
    except NoExtractableTextError as exc:
        raise HTTPException(422, str(exc))
    except Exception:
        log.exception("Ingestion failed for %s", name)
        raise HTTPException(500, "Could not process this PDF. It may be corrupted or encrypted.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest):
    engine = engine_of(request)
    if not engine.list_documents():
        raise HTTPException(409, "No documents indexed yet. Upload a PDF first.")
    try:
        return engine.ask(body.question.strip())
    except Exception:
        log.exception("Chat failed")
        raise HTTPException(502, "The language model request failed. Check your API key and try again.")


@app.get("/documents")
def documents(request: Request):
    return {"documents": engine_of(request).list_documents()}


@app.delete("/documents/{filename}")
def delete_document(request: Request, filename: str):
    removed = engine_of(request).delete_document(filename)
    if removed == 0:
        raise HTTPException(404, "Document not found.")
    return {"deleted": filename, "chunks_removed": removed}


# Serve the frontend from the same origin (no CORS needed). Must be mounted LAST.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
