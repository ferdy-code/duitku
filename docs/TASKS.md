# TASKS

Work sequentially by phase, one task at a time. Tick `[x]` only when ALL acceptance criteria (AC) pass and the Definition of Done holds. `Read:` lists the only docs needed for that task (under `docs/spec/`).

## Definition of Done (every task)

- `ruff check`, `ruff format --check`, `pyright`, `pytest` all green
- Relevant unit/integration tests written
- Dependency rules from `spec/architecture.md` respected; no secrets in code/logs
- Prompt/model/tool-description change → rerun evals, log result in `docs/DECISIONS.md`
- One small commit per task

---

## Phase 0 — Foundation

- [x] **T-001 Project init.** `uv init`, folders per `architecture.md`, `pyproject.toml` (pin versions after checking latest stable), `ruff` + `pyright` config, `.gitignore` (`.env`, `data/`).
  Read: `architecture.md`. AC: `uv run pytest` runs (0 tests OK); ruff + pyright green.
- [x] **T-002 Config.** `config.py` with `pydantic-settings` covering every var in `.env.example`; fail fast with a clear message on missing required vars.
  AC: unit test loads config from dummy env; no `os.environ` outside `config.py`.
- [x] **T-003 Database & migrations.** Prerequisite: local PostgreSQL 18 with the pgvector extension available (no Docker; see DECISIONS 2026-10-12); async engine; SQLAlchemy models + Alembic for all tables (except checkpoint tables).
  Read: `data-model.md`. AC: `upgrade head` from empty DB works; `downgrade base` works; `vector` extension enabled.
- [x] **T-004 Repositories.** `db/repositories/*`, `user_id` mandatory everywhere; seed default categories on user creation.
  Read: `data-model.md`. AC: CRUD integration tests; **isolation test**: user A cannot read/update/delete user B's rows.

## Phase 1 — Pure domain + tool calling without a framework

Goal: understand tool calling from first principles before using a framework.

- [x] **T-101 `parse_idr_amount`.** `domain/money.py` per `tools.md`. ≥25 parametrized cases: `35rb`, `35k`, `35 ribu`, `1,5 juta`, `2.5jt`, `Rp 1.250.000`, `35.000`, `1.5` (ambiguous), `0`, `-5rb`, `5m` (error), garbage.
  Read: `tools.md`. AC: all pass; ambiguous input returns `AMBIGUOUS_AMOUNT`, never a guess.
- [ ] **T-102 `resolve_period`.** `domain/periods.py`, TZ-aware, Monday-start weeks, injected `now`.
  Read: `tools.md`. AC: tests cover month/year rollover, leap February, no direct `datetime.now()` in logic.
- [ ] **T-103 Tool envelope + decorator.** `tools/base.py`: ok/error envelope, context extraction from `RunnableConfig`, audit decorator (`tool_audit_log`, redaction when `LOG_CONTENT=false`).
  Read: `tools.md`. AC: unexpected exception → generic `INTERNAL_ERROR`; validation failure → `INVALID_ARGUMENT` with `hint`.
- [ ] **T-104 Core tools.** `add_expense`, `query_expenses`, `resolve_period` (tool), including `idempotency_key`.
  Read: `tools.md`, `data-model.md`. AC: same `tool_call_id` twice → one row; every `group_by` matches reference SQL.
- [ ] **T-105 Manual loop experiment.** `experiments/01_manual_loop.py`: tool-calling loop with the OpenAI-compatible Chat Completions API directly (`openai` SDK, no LangGraph, `base_url` from config) using the T-104 tools: send → read `message.tool_calls` → execute → send `role:"tool"` message with `tool_call_id` → repeat until `finish_reason != "tool_calls"`.
  AC: handles "makan siang 35rb di warteg" then "total hari ini berapa?" correctly; `DECISIONS.md` notes message format, `tool_calls`↔`role:"tool"` pairing, stop condition.

## Phase 2 — LangGraph ReAct + short-term memory + CLI

- [ ] **T-201 State + prompt builder.** `agent/state.py`, `agent/prompts.py`.
  Read: `agent.md`. AC: no placeholders left unfilled; empty memories still valid.
- [ ] **T-202 Graph v1.** `agent/{graph,nodes,routing}.py`: `agent ↔ tools` only (no approval/memory yet), `MAX_AGENT_STEPS`, `GraphRecursionError` handling.
  Read: `agent.md`. AC: fake-model test: loop stops at step cap; ToolMessage invariant holds.
- [ ] **T-203 Postgres checkpointer.** `AsyncPostgresSaver` + pool, `setup()` at startup.
  Read: `agent.md`. AC: second message in same thread sees the first; different threads isolated; history survives process restart.
- [ ] **T-204 History trimming.** `agent/trimming.py`.
  Read: `agent.md`. AC: trimmed view never starts with an orphan ToolMessage or splits a tool_calls↔ToolMessage pair; stored state unchanged.
- [ ] **T-205 CLI.** `interfaces/runner.py` + `interfaces/cli.py`.
  Read: `interfaces-ops.md`. AC: multi-turn chat works; `--verbose` shows tool calls + observations.
- [ ] **T-206 More tools.** `update_expense`, `get_exchange_rate` (external API, cache, timeout, retry-once). *`get_exchange_rate` is P1 in PRD but built here to practice calling external APIs.*
  Read: `tools.md`. AC: HTTP-mocked tests (success, timeout, malformed); "change the last one to 40rb" updates the correct expense (R-05).
- [ ] **T-207 Eval harness v0.** `tests/evals/runner.py` + cases E-01…E-10 via `pytest -m eval`; report pass rate, avg steps, tokens.
  Read: `EVALS.md`. AC: fresh DB per case; asserts on DB effects + tool-call order; baseline score logged in `DECISIONS.md`.

**Gate (end of Phase 2):** baseline ≥ 80% on E-01…E-10; otherwise fix tool descriptions/prompt first.

## Phase 3 — Long-term memory

- [ ] **T-301 Embedding service.** `memory/embeddings.py` per `memory.md` (lazy load, E5 prefixes, executor).
  Read: `memory.md`. AC: dimension matches `EMBEDDING_DIM`; Indonesian paraphrases score higher than unrelated sentences.
- [ ] **T-302 Memory store.** `memory/store.py` + repository: save (with dedupe), recall, forget.
  Read: `memory.md`, `data-model.md`. AC: two near-duplicate saves → one row (updated); recall filtered by `user_id`.
- [ ] **T-303 Threshold calibration.** `experiments/calibrate_memory_threshold.py`: ~30 relevant/irrelevant Indonesian pairs; pick recall + dedupe thresholds; compare `multilingual-e5-small` vs `bge-m3`.
  Read: `memory.md`. AC: decision + summary data in `DECISIONS.md`; thresholds in config.
- [ ] **T-304 Memory tools.** `save_memory`, `recall_memory`, `forget_memory`.
  Read: `tools.md`, `memory.md`. AC: sensitive data (card/ID numbers) refused by prompt AND a regex validator; tested.
- [ ] **T-305 `load_context` node.** Auto-recall top-3 into `recalled_memories`.
  Read: `agent.md`. AC: memory saved in thread A appears in thread B (same user); simulated embedding failure → run still succeeds.
- [ ] **T-306 Memory evals.** Add E-15…E-19.
  Read: `EVALS.md`. AC: ≥ 80% pass; sensitive-data case (E-19) 100%.

## Phase 4 — Telegram, HITL, receipts, observability

- [ ] **T-401 Telegram basics.** `interfaces/telegram/`: long polling, text handler, allowlist middleware, rate limit, per-thread lock, HTML formatting, `/start`, `/bantuan`.
  Read: `interfaces-ops.md`. AC: non-allowlisted ID never reaches the agent (test); handler errors don't crash the bot.
- [ ] **T-402 Approval node + `REQUIRES_APPROVAL`.** Including `delete_expense`.
  Read: `agent.md`. AC: fake-model tests: approve → tool runs; reject → no DB effect and agent responds sensibly; ToolMessage invariant holds.
- [ ] **T-403 Approval in Telegram.** Inline buttons, callback → `resume_turn`, handling of messages while pending.
  Read: `interfaces-ops.md`. AC: delete flow works end-to-end on real Telegram (manual) + handler tests with fake updates.
- [ ] **T-404 Receipts.** Photo handler → file + `receipts` row; `parse_receipt` (structured output + `warnings`); `add_expense(source="receipt")` triggers approval.
  Read: `tools.md`, `interfaces-ops.md`. AC: tested on ≥20 real receipts (minimarket, restaurant, e-commerce); accuracy of total/merchant logged (target ≥ 85%, G3); images never enter checkpoints.
- [ ] **T-405 `interrupt` idempotency test.** Prove `approval` has no side effects before `interrupt()` (resuming twice duplicates nothing).
  Read: `agent.md`. AC: test green and documented.
- [ ] **T-406 Observability.** `observability/tracing.py`: spans, metrics, `agent_runs` rows, redaction per `LOG_CONTENT`.
  Read: `interfaces-ops.md`. AC: one conversation's trace is complete in local SigNoz; `agent_runs` has tokens + duration.
- [ ] **T-407 Full eval.** All cases in `EVALS.md` incl. safety/injection.
  Read: `EVALS.md`. AC: ≥ 90% total (G2); section E (safety) 100%.
- [ ] **T-408 Deploy.** Multi-stage `Dockerfile` (non-root), `app` service in `docker-compose.yml`, migrations on deploy, scheduled `pg_dump` backups, daily job enforcing `RECEIPT_RETENTION_DAYS`.
  AC: bot runs 7 days on VPS without manual restart; backup restore tested once.
- [ ] **T-409 Dogfood 4 weeks.** Use daily; log each real failure in `DECISIONS.md` and add it as a new eval case.
  AC: G4 and G5 reported at week 4 (active days/week, cost per message).

**v1 gate:** G2 ≥ 90%, G3 ≥ 85%, and a go/no-go on productizing based on G4/G5 data.

## Phase 5 — Fast follow & experiments (optional)

- [ ] **T-501 Budgets.** `set_budget`, `check_budget`, 80%/100% warnings after `add_expense` (R-20).
- [ ] **T-502 Scheduled reports** weekly/monthly using existing tools (R-22).
- [ ] **T-503 Auto-summarization** of long history (replaces pure trimming).
- [ ] **T-504 CrewAI experiment.** `experiments/02_crewai/`: "analyst" + "budget advisor" agents build a monthly report from real data. AC: comparison notes vs LangGraph (control, debuggability, token cost) in `DECISIONS.md`.
- [ ] **T-505 AutoGen experiment.** `experiments/03_autogen/`: similar scenario; comparison notes.
- [ ] **T-506 Product review.** Resolve PRD open questions (UU PDP, column encryption, price vs token cost) before accepting external users.
