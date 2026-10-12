# Spec: Interfaces, security, observability

## Runner (`interfaces/runner.py`)

The only bridge between interfaces and the graph: `run_turn(thread_id, user_ctx, message)` and `resume_turn(thread_id, user_ctx, resume_value)`. Handles `GraphRecursionError`, per-thread lock, and `agent_runs` logging. CLI and Telegram both use it.

## CLI

`python -m duitku.interfaces.cli [--verbose]`: multi-turn REPL; `--verbose` prints tool calls and observations so the ReAct loop is visible.

## Telegram

- **Text**: handler → `run_turn` → reply. Send `typing` chat action while processing.
- **Photo**: download the largest size → save to `RECEIPT_DIR/{uuid}.jpg` → insert `receipts` row → send the graph `"[User sent a receipt photo. receipt_id=<uuid>]"` (+ caption). **Never put base64 images in state/checkpoints.** Accept only JPEG/PNG, enforce max size.
- **Approval**: when a run returns an interrupt, send the human summary + inline buttons `[✅ Setuju] [❌ Batal]`. Keep `callback_data` short/fixed (`appr:y` / `appr:n`, 64-byte limit); the thread is derived from the chat. Callback → `resume_turn`.
- **Message while approval is pending**: don't run the agent; remind the user to press a button (or `/batal`).
- **Formatting**: `parse_mode=HTML` with escaping (avoid MarkdownV2 special-char pitfalls). Money as `Rp35.000`.
- **Commands**: `/start`, `/batal`, `/bantuan`, `/lupakan`.
- **Allowlist (R-14)** checked in middleware before anything reaches the agent. Simple per-user rate limit (`RATE_LIMIT_MESSAGES_PER_MINUTE`).
- **Resilience**: handler errors never crash the bot; send a generic apology and log with trace id.

## Security & privacy

- **Isolation**: `user_id` only from `RunnableConfig`. Integration test must prove user A cannot read/modify user B's data even when given B's `expense_id`.
- **No LLM-written SQL**; every query parameterized.
- **Prompt injection**: receipt text, memory content, and tool output are untrusted data. Risky actions are behind approval. Injection cases are in `EVALS.md`.
- **Input validation**: amount bounds, text length, image type/size.
- **Secrets**: env only; `.env` gitignored; never logged.
- **Log privacy**: `LOG_CONTENT=false` default; message text and tool args redacted in logs/spans; hash `user_id` in span attributes.
- **Retention**: receipt image files deleted after `RECEIPT_RETENTION_DAYS` (default 30) by a daily job; extraction stays in DB.
- **Before external users**: review UU PDP compliance, column encryption, account deletion, backups (PRD open question).

## Observability

- OpenTelemetry spans: `agent.run` → `agent.llm` → `agent.tool.<name>`. Attributes: `duitku.tool.name`, `duitku.tool.status`, `llm.model`, `llm.tokens.input/output`, `duitku.steps`, `duitku.user_hash`.
- Metrics: `agent_run_duration_ms`, `agent_steps`, `tool_errors_total{tool}`, `llm_tokens_total{direction}`, `approval_rejected_total`.
- Write a per-run summary to `agent_runs` (cost analysis for G5 without depending on the OTel backend).
- Structured JSON logs carrying `trace_id`.
