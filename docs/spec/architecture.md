# Spec: Architecture

If this conflicts with PRD, stop and ask. Library APIs change: verify against pinned versions' docs; log deviations in `docs/DECISIONS.md`.

## Decisions

| #   | Decision                                                                  | Why                                                                                               |
| --- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| D1  | LangGraph core; hand-written `StateGraph` (not only prebuilt helpers)     | Learning goal: understand the ReAct loop, state, interrupt. CrewAI/AutoGen only in `experiments/` |
| D2  | One PostgreSQL (+pgvector) for transactions, vector memory, checkpoints   | Simple ops on one VPS. Checkpoint tables are created by `checkpointer.setup()`, not Alembic       |
| D3  | Tools get `user_id`/`timezone` from `RunnableConfig`, never from LLM args | Data isolation                                                                                    |
| D4  | All numbers computed in SQL/Python, never by the LLM                      | Accuracy                                                                                          |
| D5  | Money = `BIGINT` integer rupiah                                           | IDR has no practical minor unit; no floats. Foreign currency stored separately with rate          |
| D6  | Soft delete for expenses                                                  | Reversible, auditable. Default queries filter `deleted_at IS NULL`                                |
| D7  | HITL via LangGraph `interrupt()` for risky actions                        | Safety (see `agent.md`)                                                                           |
| D8  | Interfaces (Telegram/CLI) separated from agent                            | New channels (WhatsApp) are cheap to add                                                          |

## Stack

| Layer             | Choice                                                                                                                                                                                                                           |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language / pkg    | Python 3.12, `uv`                                                                                                                                                                                                                |
| Agent             | `langgraph`, `langchain-core`, `langchain-openai`, `openai` (pin versions at setup)                                                                                                                                                     |
| Checkpoint        | `langgraph-checkpoint-postgres` (`AsyncPostgresSaver`), psycopg v3 + pool                                                                                                                                                        |
| LLM               | Any OpenAI-compatible endpoint (`ChatOpenAI(base_url=…)`). Env: `LLM_BASE_URL`, `LLM_API_KEY`, `MODEL_AGENT`, `MODEL_LIGHT`, `MODEL_VISION` (see `.env.example`). Never hardcode                                                                                                                       |
| DB                | PostgreSQL 18 + pgvector (`pgvector/pgvector:pg18`)                                                                                                                                                                              |
| ORM/migrations    | SQLAlchemy 2.x async (Core for aggregates) + Alembic                                                                                                                                                                             |
| Embeddings        | `sentence-transformers`, default `intfloat/multilingual-e5-small` (384-dim). E5 needs `"query: "` / `"passage: "` prefixes. Alt: `bge-m3` (1024-dim, heavier). Dim is tied to `vector(N)`: changing model = migration + re-embed |
| Telegram          | `python-telegram-bot` ≥ 21 (async), long polling first, webhook later                                                                                                                                                            |
| Config/validation | `pydantic` v2, `pydantic-settings`                                                                                                                                                                                               |
| HTTP              | `httpx` async                                                                                                                                                                                                                    |
| Observability     | `opentelemetry-sdk` + OTLP → SigNoz                                                                                                                                                                                              |
| Quality           | `pytest`, `pytest-asyncio`, `ruff`, `pyright`                                                                                                                                                                                    |

## Repo layout

```
duitku/
├── CLAUDE.md  pyproject.toml  docker-compose.yml  .env.example  alembic.ini
├── docs/            # PRD, TASKS, EVALS, DECISIONS, spec/
├── experiments/     # 01_manual_loop.py, 02_crewai/, 03_autogen/ (outside core app)
├── migrations/
├── src/duitku/
│   ├── config.py                 # only place that reads env
│   ├── db/        engine.py, models.py, repositories/{expenses,budgets,memories,users,audit}.py
│   ├── domain/    money.py (parse_idr_amount, format_idr), periods.py, categories.py, budgets.py   # pure, no I/O
│   ├── memory/    embeddings.py, store.py
│   ├── tools/     base.py, registry.py, expenses.py, periods.py, budgets.py, receipts.py, fx.py, memory.py
│   ├── agent/     state.py, prompts.py, nodes.py, routing.py, trimming.py, graph.py
│   ├── interfaces/ runner.py, cli.py, telegram/{bot,handlers,keyboards,formatting}.py
│   └── observability/tracing.py
└── tests/  unit/  integration/  evals/{cases.yaml,runner.py,seeds/}
```

## Dependency rules

- `domain` imports nothing from the project.
- `db`, `memory` may import `domain`, `config`.
- `tools` may import `domain`, `db`, `memory`; NOT `agent`/`interfaces`.
- `agent` may import `tools`, `memory`; NOT `interfaces`/`telegram`.
- `interfaces` import `agent` only via `runner.py`.
- Files ~<300 lines. One tool = one public function with a Pydantic input schema.

## Testing

| Level       | Scope                                                                                                  | Tools                           |
| ----------- | ------------------------------------------------------------------------------------------------------ | ------------------------------- |
| Unit        | `parse_idr_amount` (table-driven), `resolve_period` (month/year edges, Monday week), routing, trimming | pytest, no I/O                  |
| Integration | repositories + tools against real Postgres/pgvector; cross-user isolation; `add_expense` idempotency   | pytest-asyncio + docker compose |
| Graph       | wiring + HITL with a fake chat model: approve, reject, idempotent resume, ToolMessage invariant        | pytest, no API key              |
| Eval        | real LLM, cases in `EVALS.md`; assert DB effects + tool calls, not reply text                          | `pytest -m eval`                |

Changing prompt, model, or tool descriptions → rerun evals, log result in `DECISIONS.md`.
