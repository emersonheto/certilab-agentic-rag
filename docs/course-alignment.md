# Course stack alignment

This project implements a full Agentic RAG system using the professor's course stack: LangGraph, OpenAI, Tavily, Chainlit, and Phoenix/OpenInference. It supports two modes — **mock** (offline, deterministic, no external services required) and **real** (MySQL + Qdrant + embeddings).

## uv workflow

Use `uv` for every local command:

```bash
# Base dev + course integrations + observability
uv sync --extra dev --extra future-integrations --extra observability

# Add local embeddings (sentence-transformers, no API key required)
uv sync --extra dev --extra future-integrations --extra observability --extra local-embeddings

uv run pytest
uv run ruff check .
uv run mypy app
```

Base installs remain minimal. Course integrations live behind extras so the API runs without optional services or credentials.

## Graph pipeline

`app.graph.build_rag_pipeline()` selects the data stack from `APP_MODE` and the graph engine from `GRAPH_ENGINE`:

- `GRAPH_ENGINE=langgraph` (default): uses a real `StateGraph` with `START`, `END`, `add_node`, `add_conditional_edges`, and `compile().invoke(...)`.
- `GRAPH_ENGINE=deterministic`: uses the original local Python pipeline.

Both paths preserve the same public `ask(question, principal) -> AskResponse` contract.

The LangGraph nodes mirror the course examples:

1. `route_question`
2. `retrieve_structured`
3. `retrieve_semantic`
4. `web_search`
5. `generate_answer`

If LangGraph is not installed, the app falls back to the deterministic pipeline.

## Mock mode and real mode

`APP_MODE=mock` is the default. In mock mode:

- Data loads from JSON fixtures in `data/mock/`.
- Semantic index uses `InMemoryVectorIndex` (bag-of-words cosine similarity, no embeddings).
- No OpenAI, Qdrant, MySQL, or Tavily calls are made.
- Responses are deterministic for tests and demos.
- No Docker or external services required.

`APP_MODE=real` activates the full production-like stack:

- **Data**: `MySQLLoader` reads from the test MySQL database via `DB_*` settings.
- **Vector store**: `QdrantVectorIndex` persists and searches embeddings in Qdrant (Docker Compose).
- **Embeddings**: `EmbeddingsProvider` uses OpenAI `text-embedding-3-small` by default, falls back to `sentence-transformers` locally, and degrades to a zero-vector if both are unavailable.
- **Answer generation**: OpenAI `gpt-4o-mini` when `OPENAI_API_KEY` is set; deterministic fallback otherwise.

Start Qdrant before running in real mode:

```bash
docker compose up qdrant -d
APP_MODE=real uv run uvicorn app.main:app --reload
```

## Qdrant vector database

Qdrant runs via Docker Compose (included in the repo):

```bash
docker compose up qdrant -d   # starts Qdrant on localhost:6333 (REST) and 6334 (gRPC)
docker compose down           # stop when done
```

The Qdrant collection (`certilab-rag` by default) is created idempotently on first `APP_MODE=real` startup. Each stored point includes `customer_id` and `certificate_code` in its payload for mandatory tenant-scoped filtering.

## Embeddings provider

Controlled by `EMBEDDING_PROVIDER`:

| Value | Behavior |
|---|---|
| `auto` (default) | OpenAI if `OPENAI_API_KEY` is set; otherwise sentence-transformers |
| `openai` | OpenAI `text-embedding-3-small` (falls back to local if key is missing) |
| `local` | sentence-transformers offline (no network or credentials) |

Install local embeddings:

```bash
uv sync --extra local-embeddings
```

## Tavily web search

Questions with web/external terms route to `web_search`. Tavily is optional:

- Missing `TAVILY_API_KEY` returns a safe fallback source.
- Missing `tavily-python` returns a safe fallback source.
- Tavily is never required for the base test suite.

## Chainlit UI

Chainlit is optional and runs against the same pipeline and authorization model:

```bash
uv sync --extra future-integrations
export CHAINLIT_DEMO_TOKEN=<demo-client-101-token>
uv run chainlit run ui/chainlit_app.py --port 8001
```

Set `CHAINLIT_DEMO_TOKEN` to choose the demo principal. Customer isolation is preserved because every Chainlit question becomes a `Principal` through the demo auth adapter.

## Phoenix observability

Phoenix spans are optional:

```bash
uv sync --extra observability
uv run phoenix serve                          # starts Phoenix UI at http://localhost:6006
PHOENIX_ENABLED=true uv run uvicorn app.main:app --reload
```

Tracing records safe metadata only: role, route, customer-scope presence, question length, source counts, certificate counts, and timing. Full questions, secrets, DSNs, storage paths, and certificate codes are never recorded.

## FastAPI endpoints

```bash
uv run uvicorn app.main:app --reload
```

- `GET /health` — service status
- `POST /ask` — RAG query (requires `X-Demo-Token` header)
- `GET /docs` — Swagger interactive UI
- `GET /redoc` — ReDoc documentation
