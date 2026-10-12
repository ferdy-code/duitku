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
