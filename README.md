# Agent Orchestration System

A multi-agent system built with LangGraph, orchestrating four capabilities — tool use, persistent memory, agent coordination, and human-in-the-loop escalation — with PostgreSQL for structured state, ChromaDB for semantic memory, and Redis/Celery for distributed background execution.

```
                                  +--> math_agent (calculator tool) -----+
POST /tasks -> retrieve_memory -> supervisor +--> research_agent (KB search tool) -+--> finalize -> done
   (Celery task,                  +--> general_agent --------------------+     ^
   Postgres checkpoint)           +--> human_escalation (interrupt) ------------+
                                            |  graph pauses; Postgres holds the checkpoint
                                            v
                            POST /escalations/{id}/respond -> Command(resume=...) -> resumes exactly where it paused
```

## Why it's built this way

Same philosophy as the sibling `rag-hybrid-search` project: every backend sits behind an interface with an **offline default** (no API key, no external services beyond a database) and a **real production implementation**.

| Capability | How it's implemented |
|---|---|
| Tool use | A safe AST-based calculator (no `eval`) and a knowledge-base search tool that calls the `rag-hybrid-search` project's `/retrieval/search` over HTTP — the two portfolio projects compose |
| Persistent memory | **Two separate stores**, each used for what it's good at: PostgreSQL for exact, queryable `tasks`/`escalations` records; ChromaDB for fuzzy similarity-search recall of past interactions, gated by a relevance threshold so unrelated memories don't leak in |
| Agent coordination | A supervisor node classifies each request and routes to a specialist agent — a standard LangGraph supervisor pattern |
| Human-in-the-loop | `interrupt()` pauses the graph mid-run (state persisted via the Postgres checkpointer); `POST /escalations/{id}/respond` resumes it with `Command(resume=...)` from the exact checkpoint, potentially in a completely different process |
| Background execution | `POST /tasks` enqueues a Celery task and returns immediately (202); a separate worker process (Redis broker/backend) actually runs the graph, decoupling interactive request handling from agent execution time |
| Reasoning backend | `LocalRuleBasedProvider` (default): keyword-heuristic routing + direct tool-output formatting, no API key needed. `LLM_PROVIDER=openai`/`anthropic` swaps in GPT-4o / Claude Sonnet for real reasoning — nothing else in `app/graph` changes |

This means the entire graph — routing, tool use, memory read/write, and **both** escalation outcomes (approve/reject) — is fully exercisable with zero API keys, and was verified two ways during development: in-process (`CELERY_TASK_ALWAYS_EAGER=true`) and against a genuinely separate Celery worker process consuming from a real Redis broker.

## Running it

**Docker (production-shaped: real Postgres, Redis, Chroma server, and a separate worker process):**
```bash
docker compose up --build
# API at http://localhost:8000/docs
```

**Locally, no Docker** (needs a Postgres reachable at `DATABASE_URL`, or set `CHECKPOINTER_BACKEND=sqlite` to skip Postgres entirely for local dev):
```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload &
celery -A app.celery_app worker --loglevel=info &     # skip if CELERY_TASK_ALWAYS_EAGER=true
```

**End-to-end smoke test** (all four routes, both escalation outcomes, works whether Celery is eager or backed by a real worker):
```bash
python scripts/smoke_test.py --base-url http://localhost:8000
```

**Unit tests** (safe-math evaluator, routing heuristics, semantic memory + relevance threshold, ORM models — no live infrastructure needed):
```bash
pytest tests/ -v
```

To use real GPT-4o / Claude Sonnet, set `LLM_PROVIDER=openai` or `anthropic` and the matching API key in `.env`. To let the research agent actually query documents instead of returning "not configured", run the `rag-hybrid-search` project alongside this one and set `RAG_SERVICE_URL=http://localhost:8000` (its port, not this project's).

## API

| Endpoint | Purpose |
|---|---|
| `POST /tasks` | Submit `{"input": "..."}`; enqueues the graph run, returns `202` with the task immediately (`pending`/`completed`/`awaiting_human` depending on timing) |
| `GET /tasks/{id}` | Poll task status: `pending` → `running` → `completed` \| `awaiting_human` \| `failed` |
| `GET /tasks` | List recent tasks |
| `GET /escalations?status=pending` | List escalations awaiting a human response |
| `POST /escalations/{id}/respond` | Submit `{"response": "approve"}` (or `"reject"`, or free-text guidance) to resume the paused graph |

Full interactive schema at `/docs` once running.

## Design notes / what a next iteration would add

- **Escalation granularity**: the demo escalates at the *whole-request* level (the supervisor decides up front). The same `interrupt()`/`Command(resume=...)` pattern applies identically to a *single tool call* mid-agent (e.g., "approve this specific `send_email` call") — same primitive, finer placement.
- **The offline `LocalRuleBasedProvider`'s routing is keyword-based**, not learned - it's a real, testable substitute for exercising the graph's control flow, not a claim about language understanding. One regression worth knowing about: an earlier, looser regex misrouted "90-minute" into the math agent (digit immediately followed by a hyphen); the fix requires a digit on *both* sides of an operator and is covered by a test (`test_hyphenated_compound_does_not_false_positive_to_math`).
- **Semantic memory's relevance threshold** (`min_similarity=0.3`) is tuned for the offline hashing embedder, which has no learned semantics — recalibrate (likely upward) when `EMBEDDING_BACKEND=openai`, since real embeddings separate unrelated topics far more cleanly and the current threshold was chosen empirically against the crude baseline, not derived from first principles.
- **Checkpointer connections are opened fresh per graph invocation** (one Celery task = one `with PostgresSaver.from_conn_string(...)` block) rather than pooled across the worker's lifetime - simple and correct for this task-per-job execution model; a very high-throughput deployment would want a shared connection pool instead.
- Not included: auth/authz, retry/backoff policy for failed Celery tasks, a TTL/expiry sweep for escalations nobody ever responds to, structured tracing across the FastAPI → Celery → LangGraph boundary. All straightforward additions on top of the existing module boundaries.

## A note on debugging this one

The Postgres-checkpointer + FastAPI + eager-Celery combination initially hung indefinitely on the very first request. Tracked down with `faulthandler.dump_traceback_later()` (dumps every thread's stack if the process is still alive after N seconds) to a specific line: the request handler was still holding its own DB session open (via FastAPI's `Depends`) while synchronously running the entire graph in-process. Fixed by closing the router's session before triggering task execution, plus forcing IPv4 and adding explicit connection timeouts so any future connection issue fails fast and loud instead of hanging silently. Verified fixed against both in-process (eager) execution and a genuinely separate Celery worker process.
