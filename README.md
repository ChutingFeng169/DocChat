# DocChat

A Streamlit-based **Retrieval-Augmented Generation (RAG)** app that lets you upload a document (PDF, Word, PowerPoint, or Excel), chunk & embed it into a Chroma vector store, and ask questions answered by an OpenAI LLM with conversational memory.

---

## Features

- Multi-format document loading: **PDF, DOCX, PPTX, XLSX**
- Chunking with `RecursiveCharacterTextSplitter` (configurable size/overlap in sidebar)
- OpenAI embeddings (`text-embedding-3-small`)
- **Persistent Chroma vector store** on disk — re-uploading the same file skips re-embedding (cached by SHA-256 hash)
- `gpt-4o-mini` chat model with `ConversationalRetrievalChain` + memory
- Friendly Streamlit chat UI with sidebar conversation preview, "clear" button, and **source chunk citations** below each answer
- Configurable retrieval count (`k`) in sidebar
- Robust error handling (missing API key, file load failures, API errors)

---

## Reliability & Evaluation Engineering

Beyond the core RAG app, this project includes a set of engineering improvements built and measured to make the system faster, more reliable, and easier to evaluate objectively:

- RAG Evaluation (RAGAS) — Built an evaluation pipeline comparing chunk-size configurations (256/512/1024, k=5) against faithfulness, answer relevancy, and context precision/recall scores. Found that chunks below 512 fragment relevant content (hurting precision/recall), while chunks above 512 pull in irrelevant context (hurting faithfulness). Settled on chunk size 512, overlap 100 — a 7% higher overall RAGAS score than the chunk=256 baseline.
- Observability — Instrumented call-level tracing (LangSmith) across the retrieval → prompt → output pipeline, which surfaced that LLM inference accounts for 75–85% of total response time, directly motivating the caching work below.
- Semantic Caching — Added a similarity-based cache for repeated/near-duplicate queries. Tuned the similarity threshold to 0.90 after testing 0.85/0.92/0.95, balancing false-positive cache hits against miss rate — reaches 94% hit accuracy, cutting cached response time from ~8–11s to ~2–4s.
- Rate Limiting & Fallback — Added exponential backoff on 429 errors (up to 10 retries) with automatic fallback to a secondary model if the primary model keeps failing, so users get an answer instead of an error.
- Timeout Handling — Added a 15-second request timeout, calibrated against the normal 8–11s response range, so a hung request fails gracefully instead of blocking the UI indefinitely.
- Async Pipelines — Parallelized multi-file processing; processing 3 files concurrently takes ~1.5s versus ~4.5s sequentially. Tested concurrency limits up to 10 files and capped it at 3 to stay under embedding-API rate limits.

Planned next: streaming responses with mistake-retry handling, and agent workflows (ReAct-style reasoning + tool calling with safety guardrails).

---

## Quickstart

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file in the project root:

```bash
cp .env.example .env   # then edit .env and paste your key
```

`.env` should contain:

```
OPENAI_API_KEY=sk-your-real-key-here
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the printed local URL (usually <http://localhost:8501>).

---

## Usage

1. Enter your name in the sidebar.
2. Upload a PDF / Word / PowerPoint / Excel file.
3. Wait for the "✅ Ready!" message (document is chunked & embedded).
4. Ask any question in the chat box; the app keeps conversational context.
5. Click **LangSmith Trace** beneath each answer to see the retrieval-prompt-output, token cost, and time cost.
6. Adjust **Instant**, **Thinking** in the "Response Mode" expander.
7. Click **🗑️ Clear Conversation** in the sidebar to reset.

---

## Project Structure

```
rag-project/
├── .chroma/           # persisted vector store caches (gitignored)
├── .env               # your local API key (gitignored, NEVER commit)
├── .env.example       # template (safe to commit)
├── .gitignore
├── app.py             # the entire Streamlit application
├── requirements.txt   # pinned dependencies
└── README.md
```

---

## What's New

### Phase A — Security, Reproducibility, Robustness
- `.gitignore` (prevents leaking secrets, temp files, caches)
- `.env.example` (safe template)
- `README.md` with security warning
- `requirements.txt` with pinned versions
- `app.py` error handling: missing API key guard, `try/except` on file load / embedding / chain invoke, temp file cleanup

### Phase B — Vector Store Persistence
- Chroma is persisted to `.chroma/<file-hash>/` on disk
- Re-uploading the same file (detected via SHA-256 hash) loads the cached vector store instantly — no re-embedding

### Phase C — UX Enhancements
- Retrieval-prompt-output, token cost, and time cost shown in a collapsible `LangSmith Trace` expander under each answer
- Configurable mode in an "Response Mode" sidebar expander

### Phase D — Reliability & Evaluation Engineering
- RAGAS-based evaluation pipeline for comparing chunk-size configurations
- LangSmith tracing for latency observability
- Semantic caching layer with tuned similarity threshold
- Exponential-backoff rate limiting with model fallback
- Request timeout handling
- Async/concurrent multi-file processing
