# Agent Orchestration System

[![CI](https://github.com/shehzadahmed-gh/agent-orchestration-system/actions/workflows/ci.yml/badge.svg)](https://github.com/shehzadahmed-gh/agent-orchestration-system/actions/workflows/ci.yml)

A multi-agent service built with LangGraph. A supervisor routes each request to a math agent (calculator tool), a research agent (knowledge-base search), a general agent, or a human reviewer. Requests that need approval pause mid-graph and resume when someone responds through the API, possibly in a different worker process. Task state lives in PostgreSQL, long-term memory in ChromaDB, and graph runs happen in Celery workers fed through Redis.

```
                                  +--> math_agent (calculator) ---------+
POST /tasks -> retrieve_memory -> supervisor +--> research_agent (KB search) -------+--> finalize -> done
   (Celery task,                  +--> general_agent -------------------+     ^
   Postgres checkpoint)           +--> human_escalation (interrupt) ------------+
                                            |  graph pauses; Postgres holds the checkpoint
                                            v
                            POST /escalations/{id}/respond -> Command(resume=...) -> continues from the pause
```

## How the pieces fit

| Part | Implementation |
|---|---|
| Routing | A supervisor node picks one of four routes. With an LLM provider the model classifies the request; the default local provider uses keyword and regex rules. |
| Tools | An AST-based calculator (no `eval`, with size limits, see below) and `knowledge_base_search`, which calls `/retrieval/search` on my [rag-hybrid-search](https://github.com/shehzadahmed-gh/rag-hybrid-search) service when `RAG_SERVICE_URL` is set. |
| Human review | `interrupt()` pauses the graph and the Postgres checkpointer stores its state. `POST /escalations/{id}/respond` resumes it with `Command(resume=...)`. Replying "approve" hands the request to the general agent; anything else ends it with the reviewer's reply. |
| Task records | PostgreSQL tables for tasks and escalations, used for status queries. |
| Memory | ChromaDB. Every finished interaction is embedded and stored, and similar past interactions are fetched before routing. Results below a cosine similarity of 0.3 are dropped so unrelated memories don't leak into the context. |
| Background work | `POST /tasks` stores the task, enqueues a Celery job and returns 202. A separate worker runs the graph and writes the result back. |
| Model backends | `LLM_PROVIDER=local` (default, no API key), `openai` or `anthropic`. Nothing in `app/graph` changes between them. |

## Running it

Locally, without Postgres, Redis or a worker:

```bash
pip install -r requirements.txt
cp .env.example .env              # defaults to SQLite checkpoints and in-process Celery
uvicorn app.main:app --reload     # API docs at http://localhost:8000/docs
```

With Docker (Postgres, Redis, a Chroma server and a separate worker):

```bash
docker compose up --build
```

Tests:

```bash
pytest -q                                              # 45 tests, no services needed
python scripts/smoke_test.py --base-url http://localhost:8000   # against a running stack
```

The unit tests cover the calculator, routing rules, memory threshold, database models and the escalation claim. `tests/test_graph.py` compiles the real graph with an in-memory checkpointer and runs the math, escalation-approve and escalation-reject paths end to end. CI runs both suites, and the smoke job starts Postgres and Redis as services with the API and a Celery worker as separate processes.

To use a real model, set `LLM_PROVIDER=openai` or `anthropic` and the matching key in `.env`. To make the research agent query documents, run rag-hybrid-search alongside this service and set `RAG_SERVICE_URL` to its address.

## API

| Endpoint | Purpose |
|---|---|
| `POST /tasks` | Submit `{"input": "..."}`. Returns 202 with the task record. |
| `GET /tasks/{id}` | Status: `pending`, `running`, `completed`, `awaiting_human` or `failed` |
| `GET /tasks` | Most recent 50 tasks |
| `GET /escalations?status=pending` | Escalations waiting for a reviewer |
| `POST /escalations/{id}/respond` | `{"response": "approve"}`, `"reject"` or free text. Returns 409 if already resolved. |

## Design decisions

**Calculator limits.** Python integers have no size limit, so `9**9**9` never finishes, and the math agent turns "What is 2^9^9^9?" into exactly that. The evaluator rejects expressions over 200 characters, exponents over 1,000, and any integer result over 4,096 bits, and it checks the size of a power before computing it. Division by zero and float overflow come back as a message instead of failing the task.

**Responding to an escalation exactly once.** The first version read the escalation, checked it was pending, then updated it. Two responses arriving together could both see "pending" and resume the graph twice. The claim is now a single `UPDATE ... WHERE status = 'pending'` and the handler checks that it changed one row (`app/db/escalations.py`). `tests/test_escalations.py` reproduces the stale-read case with two sessions.

**Escalation keywords.** The local provider escalates when a word starts with one of `ESCALATION_KEYWORDS`, so "deleted" and "payments" match but "undelete" doesn't. It still has no language understanding: "What does delete do in SQL?" is escalated.

**Two stores for two jobs.** Task status needs exact queries and transactions, so it lives in Postgres. Memory recall is a similarity search, so it lives in Chroma. The LangGraph checkpointer is separate again: it holds one paused run's state so that run can resume.

## Debugging notes

With the Postgres checkpointer and eager Celery, the first request hung with no error. `faulthandler.dump_traceback_later()` dumped every thread's stack after a timeout and showed the request handler holding its own database session (from FastAPI's `Depends`) while it ran the whole graph inline. Closing the router's session before calling `.delay()` fixed it. I also forced IPv4 and added connection timeouts so connection problems fail fast instead of hanging.

## Limitations

- Escalation happens once per request, decided by the supervisor before any agent runs. The same `interrupt()` call could gate a single tool call instead (for example, approving one `send_email`).
- The 0.3 memory threshold was tuned by trial against the hashing embedder, which has no learned semantics. It would need re-tuning for OpenAI embeddings.
- Each graph run opens its own checkpointer connection. That's fine at this scale, but a busy deployment would want a shared pool.
- Not included: authentication, retry policy for failed Celery jobs, expiry for escalations nobody answers, tracing across the API, Celery and LangGraph boundary.
