![Veritas AI Demo](assets/demo.png)
# Veritas AI — Grounded RAG Document Intelligence System

A production-grade Retrieval-Augmented Generation (RAG) system engineered to eliminate LLM hallucinations by enforcing strict evidence grounding, page-level metadata tracking, and verbatim source citations.

## 🚀 Key Features

- **Strict Source Grounding:** Rejects queries outside document scope instead of guessing.
- **Page-Level Citations:** Pinpoints the exact file and page number for every claim made.
- **Evidence Verification:** Displays top retrieved chunks alongside cosine similarity match percentages.
- **Local Embedding Pipeline:** Computes high-performance dense representations on-device via `sentence-transformers/all-MiniLM-L6-v2`.
- **Low-Latency Inference:** Uses Groq Cloud LPU inference for high-speed generation.
- **Lightweight Architecture:** Vanilla JavaScript frontend with zero build toolchain overhead + FastAPI asynchronous backend.

## 🛠️ Architecture & Tech Stack

- **Backend:** FastAPI (Python 3.11)
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **LLM Engine:** Groq API (`gemma2-9b-it` / `llama-3.3-70b-versatile`)
- **Orchestration:** LangChain
- **Frontend:** HTML5, CSS3, Vanilla JavaScript (XSS-safe DOM construction)

## 📦 Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/veritas-ai.git
cd veritas-ai
