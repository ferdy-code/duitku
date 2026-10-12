# DECISIONS (append-only)

Format: `## YYYY-MM-DD — T-xxx — title`, then what / why.

## 2026-10-12 — T-001 — sentence-transformers deferred to T-301
Not in `pyproject.toml` yet. It pulls in torch (very large install), and nothing before the memory phase needs embeddings. Add it in T-301 and record the resulting install size.

## 2026-10-12 — T-001 — docs/ excluded from ruff
`extend-exclude = ["docs"]`. ruff 0.17 also formats code blocks inside `.md`; spec snippets are illustrative, not code to be linted.

## 2026-10-12 — T-001 — OTLP exporter added
Added `opentelemetry-exporter-otlp-proto-http` next to `opentelemetry-sdk` (the SDK alone has no exporter, and `OTEL_EXPORTER_OTLP_ENDPOINT` would be unusable). Caveat: the HTTP exporter conventionally listens on 4318; `.env.example` has 4317 (gRPC). Revisit in the observability task: either switch the port or use the gRPC exporter.

## 2026-10-12 — T-002 — config choices
- Required (fail fast at startup): `LLM_BASE_URL`, `LLM_API_KEY`, `DATABASE_URL`, `CHECKPOINT_DATABASE_URL`. DB URLs have no default so no credentials are baked into code; `.env.example` supplies the dev values.
- Optional / lazily required: `TELEGRAM_BOT_TOKEN` (None until the bot starts; CLI and tests must work without it), `MEMORY_RECALL_THRESHOLD` / `MEMORY_DEDUPE_THRESHOLD` (None until calibrated in T-303; memory code must raise if used while None).
- Blank value in `.env` == unset (`env_ignore_empty=True`), so `.env.example` can be copied verbatim. Consequence: a blank required var reports as missing.
- `TELEGRAM_ALLOWED_USER_IDS`: comma-separated -> `list[int]` (`NoDecode` + before-validator; default pydantic-settings expects JSON). Empty = nobody is allowed (default-deny, R-14).
- `SecretStr` for API key, bot token and both DB URLs (URLs contain passwords). Consumers call `.get_secret_value()` at the point of use only.
- Errors: `ValidationError` is wrapped in `ConfigError` listing env var names and pydantic messages, built with `include_input=False` so no values leak.
- `get_settings()` is `lru_cache`d; tests use `load_settings(env_file=None)` to bypass both the cache and the real `.env`.
- `DEFAULT_TIMEZONE` validated against IANA names; `LOG_LEVEL` restricted to the standard levels.
- Guard test greps `src/` for `os.environ` / `getenv` / `environ` outside `config.py` (hard rule: env read only in config).

## 2026-10-12 — T-003 — local Postgres instead of Docker; test DB separation
- Decision (Keina): no Docker for the DB. PostgreSQL 18 runs locally (here: `localhost:5433`); no `docker-compose.yml` in the repo for now. TASKS T-003 and `architecture.md` updated. T-408 (`app` service in `docker-compose.yml`) is untouched and will need revisiting at deploy time.
- Prerequisite: the server must offer the `pgvector` extension (`SELECT * FROM pg_available_extensions WHERE name='vector'`); migration 0001 runs `CREATE EXTENSION IF NOT EXISTS vector` (needs a role allowed to create it; the dev role is superuser). Verified on 18.3 with vector 0.8.2.
- Integration tests never touch the dev DB: they derive `<db>_test` from `DATABASE_URL`, and drop and recreate it before every test (guard: refuses names not ending in `_test`). The role needs CREATEDB. If Postgres is unreachable the tests skip with a message.
- Migration 0001 is raw DDL copied from `data-model.md` (exact match: partial/hnsw indexes, CHECKs, `gen_random_uuid()`); a test asserts the ORM models produce zero Alembic autogenerate diff against it. `vector(384)` is frozen in the migration (a migration must not follow runtime config); the models carry the same constant `EMBEDDING_DIM`. Downgrade drops the tables and the `vector` extension (fails safely if something else uses it).
- Alembic env.py is async, takes the URL from `duitku.config` (or `config.attributes["url"]` for tests), and ignores LangGraph checkpoint tables. Integration tests are sync (Alembic's async env uses `asyncio.run`, illegal in a running loop).
- Flaw noted: `get_settings()` requires `LLM_BASE_URL`/`LLM_API_KEY`, so `alembic upgrade head` fails on a machine where only the DB is configured. Tests set dummy LLM env vars. Consider lazily-required LLM settings (same pattern as Telegram) in a later task.

## 2026-10-12 — T-004 — repository conventions
- `user_id` is a mandatory positional parameter of every repository function and part of every WHERE (reads, updates, deletes). Not found and not yours look identical (`None` / `False` / empty), so no id can be used to probe another user's data. It is the second line of defence behind "`user_id` comes only from `RunnableConfig`": even a tool that forgets a filter cannot compile without a `user_id`.
- Foreign keys point at `categories.id` / `receipts.id` only, so Postgres would happily attach user B's category to user A's expense. Repos therefore check ownership before inserting/updating/upserting (`require_category`, `require_receipt`) and raise `domain.errors.NotOwnedError`. Tools (T-104) will map it to an observation.
- Repos take an `AsyncSession` and never commit: the caller owns the transaction. This is what makes "seed default categories in the same transaction as user creation" true (`get_or_create_user` -> `seed_defaults`; a test rolls back and finds neither). Seeding is `ON CONFLICT DO NOTHING`, so it is idempotent.
- `get_or_create_user` uses `INSERT .. ON CONFLICT DO NOTHING RETURNING`, then a SELECT on conflict: concurrent first messages cannot raise or double-seed. Returns `(user, created)`.
- Idempotency: `insert_expense` uses `INSERT .. ON CONFLICT (user_id, idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING RETURNING`; no row back -> SELECT the existing one. Chosen over catch-IntegrityError because a failed statement aborts the surrounding Postgres transaction. Returns `InsertResult(expense, created)`. A replay ignores the new arguments (first write wins); NULL keys never collide.
- Updates/deletes use `UPDATE/DELETE .. RETURNING id` instead of `rowcount` (typed, and the result doubles as the "was it yours" signal). Expense deletes are soft (`deleted_at`) and deleted rows are invisible to every expense read. `update_expense` accepts only a whitelisted set of columns (never `user_id`, `idempotency_key`, `source`).
- Upserts that RETURN an ORM entity use `populate_existing` (otherwise the identity map hands back the stale pre-update object).
- `repositories/memories.py` is plain CRUD plus the vector column; embedding, dedupe, similarity recall and `recall_count` bookkeeping are T-302. `list_expenses` is a simple filtered list for tests/tools plumbing, not the aggregation path (`query_expenses`, T-104).
- Default category list lives in `domain/categories.py` (pure data), names from `data-model.md`.
- Doc note: `architecture.md` lists `repositories/{expenses,budgets,memories,users,audit}.py`; `categories.py` and `receipts.py` were added since those tables have their own ownership rules. `audit.py` covers both `agent_runs` and `tool_audit_log` (append-only, no update/delete).

## 2026-10-12 — migration 0002 — composite FKs against cross-user references
- Problem (flagged in T-004): `expenses.category_id`, `expenses.receipt_id`, `budgets.category_id` pointed at `categories.id` / `receipts.id` only, so Postgres accepted user A referencing user B's rows; only the repo ownership checks prevented it.
- Fix: `UNIQUE (user_id, id)` on `categories` and `receipts` (FK target; redundant with the PK for uniqueness but required by Postgres), and the three single-column FKs replaced by `(user_id, category_id) -> categories(user_id, id)` and `(user_id, receipt_id) -> receipts(user_id, id)`. The same-user rule is now enforced by the database for raw SQL, bugs and future tools, not just by `require_category` / `require_receipt`, which stay as defence in depth (they give a clean `NotOwnedError` instead of an `IntegrityError`).
- The originals had no `ON DELETE` action (NO ACTION), so none is added: no `SET NULL (col)` column-list form is needed. `receipts.id` deletion while referenced is still refused. Default MATCH SIMPLE keeps `receipt_id NULL` valid (`user_id` is NOT NULL, so only `receipt_id` can opt out of the check). Deleting a user still cascades (all rows go in one statement; NO ACTION is checked at statement end).
- Cost: one extra unique index per target table. Downgrade restores the exact 0001 constraints and names. Models mirror it with `ForeignKeyConstraint`; `compare_metadata` stays empty. `data-model.md` DDL updated to the final state.
- Apply to dev DB manually: `uv run alembic upgrade head` (not run by the agent).

## 2026-10-12 — T-101 — parse_idr_amount
- API: `parse_idr_amount(text) -> int`, raises `AmountError(code, message, hint)`. tools.md says both `-> int` and "error with a hint", so we raise: `domain` stays independent of the tool envelope, and the tool layer maps the exception to `{"ok": false, ...}`.
- Two codes: `AMBIGUOUS_AMOUNT` (could mean two amounts: `1.5`, `5m`, mixed separators, fractional rupiah) and `INVALID_AMOUNT` (zero, negative, > 10 billion, empty, no digits). `INVALID_AMOUNT` is new; the spec had no code for these. The agent asks the user to confirm in both cases, never guesses.
- No floats: `Decimal`/`int` only. With a multiplier (`rb`, `k`, `juta`, `jt`) `.` and `,` are decimal separators (`1,5 juta` = 1,500,000). Without one, only groups of exactly 3 digits with one repeated separator count as thousands (`35.000`).
- Footgun: `1.500rb` reads as 1,500 (follows the spec). Mention it in the tool description or prompt.

## 2026-10-12 — T-102 — resolve_period
- `resolve_period(period, now, tz) -> Period(start_date, end_date, label)`. `now` is injected, so the logic never reads the clock and tests are deterministic. Unknown period raises `InvalidPeriodError` (with `.hint`); T-104 maps it to `INVALID_ARGUMENT`.
- `end_date` is inclusive, matching `query_expenses` in tools.md, so output feeds the query without ±1 conversion.
- `this_week`/`this_month`/`this_year` end at today ("so far"), so future-dated expenses in the current period are excluded. Switching to full calendar periods is a small change if needed.
- `last_7_days`/`last_30_days` include today. Weeks start on Monday. Labels are Indonesian (user-facing).
- tools.md gap: it does not state these choices or an error code for an unknown period. Add them to the spec.

## 2026-10-12 — T-103 — tool envelope and audit decorator
- `tools/base.py`: `ok`/`error` envelope, `get_context(config)` (user_id and timezone only from `config["configurable"]`), `audited(tool_name)` decorator writing `tool_audit_log`.
- Mapping: `ValidationError`/`ValueError` -> `INVALID_ARGUMENT` with hint; any other exception -> generic `INTERNAL_ERROR` with no exception text (only the exception type is logged; traceback only when `LOG_CONTENT=true`). Audit args are redacted unless `LOG_CONTENT=true`. A failed audit write is logged and does not change the tool result.
- Deviation from tools.md: LangChain validates args against `args_schema` BEFORE our wrapper runs, so the decorator never sees those errors. T-104 tools must be built with `StructuredTool.from_function(coroutine=audited("x")(fn), name=..., handle_validation_error=validation_error_envelope)`; the `@tool` decorator does not accept that kwarg in the pinned langchain-core 1.6.9.
- Tool results are dicts that LangChain JSON-dumps into `ToolMessage.content`, so `data` must be JSON-serializable (no `date`/`UUID`).
- Open: `tool_audit_log` has no `tool_call_id` column; the id is only logged on audit failure. OTel span from tools.md skipped until the observability task. Broad `ValueError` mapping may leak library error text to the LLM; introduce an explicit `ToolError` if it bites.

## 2026-10-12 — T-104 — core tools (add_expense, query_expenses, resolve_period)
- Built as `StructuredTool`s via `build_*_tool(sessionmaker=None)`; the sessionmaker is resolved at call time, so tests inject the test DB and importing tools needs no settings. `tools/clock.py` is the only place that reads the time; tests override `clock.now`.
- Idempotency: `idempotency_key` = the injected `tool_call_id` (hidden from the LLM schema). A replay returns the stored row with `duplicate: true` (first write wins, even if the replay has a different category).
- `add_expense.amount` is `str` (parsed by `parse_idr_amount`); a validator also accepts an int because models often send numbers, while the schema stays `string`.
- Dates: `occurred_on` is a local date; stored `occurred_at` = that date + current time of day in the user's tz. Range queries use local-midnight bounds `[start, end+1 day)`; day/week/month buckets use `timezone(tz, occurred_at)` then `date_trunc` (Postgres weeks start Monday).
- `query_expenses` output: `total_idr`/`count` cover the whole range; `limit` only trims `rows` (flag `rows_truncated`); `groups` is never capped so sums equal the total; `rows` only when `group_by=none`. All numbers computed in SQL. Verified against a plain-Python reference over a 5 group_by x 5 ranges x 3 category filters x 2 timezones matrix, plus other-user and soft-deleted rows.
- New error codes (spec only named `INVALID_ARGUMENT`): `UNKNOWN_CATEGORY` (hint lists valid slugs), `NOT_FOUND` (foreign receipt), `CONFLICT` (IntegrityError), `INVALID_AMOUNT`/`AMBIGUOUS_AMOUNT` from the parser.
- Approval for `source="receipt"` lives in the graph `approval` node (later), not in the tool.
- tools.md gaps: no code for unknown category/foreign receipt/unknown period; future `occurred_on` allowed; `receipt_id` not enforced for `source="receipt"`; `groups[]` shape undefined (we return `{key, total_idr, count, name?}`, no avg).
- Risks: `group_by=day` over a year gives ~366 groups in LLM context; no guard against a wrong-year `occurred_on`; "this_*" periods hide future-dated rows (T-102). Check provider handling of nullable `anyOf` schemas in T-105/T-207.

## 2026-10-12 — T-105 — manual tool-calling loop (experiments/01_manual_loop.py)
- Goal: tool calling from first principles with the `openai` SDK only (Chat Completions; base_url/key/model from config), reusing the T-104 tools. No LangGraph.
- Message format: a plain list of dicts, re-sent in full on every call (the API is stateless). Roles: `system` (once, first; date + tz + "reply in Indonesian, never compute numbers"), `user`, `assistant`, `tool`. The conversation state IS this list; it was kept across both scenario turns, so turn 2 saw turn 1.
- The assistant message that requests tools: `{"role":"assistant","content":null,"tool_calls":[{"id","type":"function","function":{"name","arguments"}}]}`. `arguments` is a JSON STRING (the model can send invalid JSON), not a dict. It must be appended to `messages` before the results, WITH its tool_calls; otherwise the tool messages are orphans. `content` was null when tools were requested.
- Pairing: each `tool_calls[i].id` needs exactly one `{"role":"tool","tool_call_id":<same id>,"content":<str>}` message right after the assistant message. `content` is a string (our JSON envelope). A model can issue several calls in one message; each gets its own tool message. We always answer every id: unknown tool, bad/non-object JSON args and tool crashes become error envelopes (UNKNOWN_TOOL, INVALID_ARGUMENT, INTERNAL_ERROR; no exception text), tested with a fake client. OpenAI's own API rejects a missing answer with HTTP 400 ("An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'") — from knowledge, not observed here. OBSERVED: our configured gateway did NOT reject an unanswered tool_call (it returned finish_reason=stop). Do not rely on that leniency; the invariant stays mandatory.
- Stop condition: loop until the response has no `tool_calls` (finish_reason "stop" instead of "tool_calls"). We key on the presence of tool_calls: if present we must answer them anyway (some gateways report a wrong finish_reason); if finish_reason says tool_calls but none are present there is nothing to run. Guard: at most `max_agent_steps` model calls per user turn, checked only after all tool calls of a step were answered, so history stays valid. On cap we return a fixed message.
- Schema: `tool.tool_call_schema.model_json_schema()` is the LLM-facing schema (injected `config`/`tool_call_id` stripped). `get_input_jsonschema()` leaks `tool_call_id`; do not use it. Tools run via `ainvoke({"name","args","id","type":"tool_call"}, config=...)`; passing a ToolCall (not bare args) is what makes InjectedToolCallId, and therefore the idempotency key, work.
- Provider check (answers the T-104 risk): the gateway accepted our nullable `anyOf` + `null` defaults schema unchanged (no flattening needed). The model sent only the args it needed, passed the amount as the raw string "35rb", and picked the category slug correctly.
- Observed: "total hari ini" chained 2 tools (resolve_period -> query_expenses) as the descriptions intend, 3 LLM calls in that turn. Tokens per call grow with history: 1950 -> 2580 (system prompt + 3 tool schemas are most of the ~1.9k baseline). The reply was Indonesian and quoted `total_idr` verbatim, no own arithmetic. DB check: exactly one expense of 35000; 3 `tool_audit_log` rows.
- Learning: this ~100-line loop is the whole ReAct agent. LangGraph (Phase 2) adds persistence/checkpoints, `interrupt()` for approval, and step/recursion limits, not new tool-calling mechanics.
- Notes: the script uses a fixed dev-DB experiment user (telegram id 900000105); `--reset` deletes it and its rows by cascade. `[tool.pyright] include` now covers `experiments`.
