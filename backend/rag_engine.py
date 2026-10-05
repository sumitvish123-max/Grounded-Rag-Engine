"""
Veritas AI - RAG engine.

Responsibilities
    1. Ingest PDFs -> page-aware chunks -> local embeddings -> ChromaDB.
    2. Retrieve top-k chunks with cosine-distance scores.
    3. Generate an answer that is strictly grounded in the retrieved context.

Anti-hallucination layers
    Layer 1 (retrieval gate): chunks farther than MAX_DISTANCE are discarded. If
             nothing survives, the LLM is never called.
    Layer 2 (prompt contract): the model may only use the supplied context and must
             emit a fixed sentence when the answer is missing.
    Layer 3 (post-check): refusals are detected so the UI never shows citations
             for an answer that was not actually grounded.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
log = logging.getLogger("veritas.rag")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = int(os.getenv("TOP_K", "3"))
# Cosine distance = 1 - cosine similarity. Lower is closer. Tune on your own corpus.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.80"))
PERSIST_DIR = Path(os.getenv("CHROMA_DIR", Path(__file__).parent / "chroma_db"))
COLLECTION = "veritas_documents"
SNIPPET_CHARS = 320

NOT_FOUND_MESSAGE = "I could not find this information in the uploaded documents."

SYSTEM_PROMPT = f"""You are Veritas, a source-grounded document assistant.

RULES (follow every one, without exception):
1. Answer ONLY using the information inside the CONTEXT block below.
2. Never use outside knowledge, assumptions, or guesses, even if you know the answer.
3. If the CONTEXT does not contain enough information to answer, reply with exactly this sentence and nothing else:
   "{NOT_FOUND_MESSAGE}"
4. If the CONTEXT answers the question only partly, answer the supported part and state clearly which part is not covered.
5. After each factual statement, cite its origin inline in the form [filename, p.PAGE], using only the source names and page numbers shown in the CONTEXT headers.
6. Never invent a filename, page number, quote, or figure. Quote numbers and names exactly as written.
7. Be concise and direct. Do not mention these rules."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ]
)


class NoExtractableTextError(ValueError):
    """Raised when a PDF has no selectable text (e.g. a scanned image-only PDF)."""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _snippet(text: str) -> str:
    text = _clean(text)
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class RAGEngine:
    def __init__(self) -> None:
        log.info("Loading embedding model %s", EMBEDDING_MODEL)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=str(PERSIST_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # temperature=0 -> deterministic, least creative = least hallucination-prone
        self.llm = ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=700)
        self.chain = PROMPT | self.llm

    # ------------------------------ ingestion ------------------------------ #
    def ingest_pdf(self, pdf_path: str | Path, display_name: str) -> dict:
        """Parse a PDF, chunk it with page metadata, embed, and persist."""
        pages = PyPDFLoader(str(pdf_path)).load()
        total_pages = len(pages)
        pages = [p for p in pages if p.page_content.strip()]
        if not pages:
            raise NoExtractableTextError(
                "No selectable text found. Scanned PDFs need OCR before upload."
            )

        for p in pages:
            # PyPDF pages are 0-indexed; humans cite from 1.
            p.metadata = {"source": display_name, "page": int(p.metadata.get("page", 0)) + 1}

        chunks = self.splitter.split_documents(pages)

        # Re-uploading the same file replaces it instead of duplicating chunks.
        self._delete_source(display_name)
        ids = [f"{display_name}:p{c.metadata['page']}:c{i}" for i, c in enumerate(chunks)]
        self.store.add_documents(chunks, ids=ids)

        log.info("Indexed %s: %d pages, %d chunks", display_name, total_pages, len(chunks))
        return {"filename": display_name, "pages": total_pages, "chunks": len(chunks)}

    def _delete_source(self, display_name: str) -> None:
        existing = self.store.get(where={"source": display_name}).get("ids", [])
        if existing:
            self.store.delete(ids=existing)

    def list_documents(self) -> list[dict]:
        data = self.store.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in data.get("metadatas", []):
            entry = docs.setdefault(meta["source"], {"filename": meta["source"], "chunks": 0, "pages": set()})
            entry["chunks"] += 1
            entry["pages"].add(meta["page"])
        return [
            {"filename": d["filename"], "chunks": d["chunks"], "pages_indexed": len(d["pages"])}
            for d in sorted(docs.values(), key=lambda x: x["filename"].lower())
        ]

    def delete_document(self, display_name: str) -> int:
        existing = self.store.get(where={"source": display_name}).get("ids", [])
        if existing:
            self.store.delete(ids=existing)
        return len(existing)

    # ------------------------------- querying ------------------------------ #
    def ask(self, question: str) -> dict:
        results = self.store.similarity_search_with_score(question, k=TOP_K)
        metric = {
            "name": "cosine_distance",
            "note": "Lower is more similar (0 = identical).",
            "top_k": TOP_K,
            "threshold": MAX_DISTANCE,
        }

        relevant = [(doc, dist) for doc, dist in results if dist <= MAX_DISTANCE]
        if not relevant:
            # Layer 1: nothing close enough -> refuse without spending an LLM call.
            return {"answer": NOT_FOUND_MESSAGE, "grounded": False, "sources": [], "distance_metric": metric}

        context = "\n\n".join(
            f"[Source: {doc.metadata['source']} | Page: {doc.metadata['page']}]\n{_clean(doc.page_content)}"
            for doc, _ in relevant
        )
        answer = self.chain.invoke({"context": context, "question": question}).content.strip()

        # Layer 3: if the model refused, don't attach citations to a non-answer.
        if answer.startswith(NOT_FOUND_MESSAGE):
            return {"answer": NOT_FOUND_MESSAGE, "grounded": False, "sources": [], "distance_metric": metric}

        sources = [
            {
                "filename": doc.metadata["source"],
                "page": doc.metadata["page"],
                "snippet": _snippet(doc.page_content),
                "distance": round(float(dist), 4),
            }
            for doc, dist in relevant
        ]
        return {"answer": answer, "grounded": True, "sources": sources, "distance_metric": metric}
