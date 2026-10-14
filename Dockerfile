# syntax=docker/dockerfile:1
# Multi-stage: uv builds the venv (CPU-only torch via [tool.uv.sources] in pyproject), the runtime
# stage copies only the venv + migrations. No .env or data/ is copied (see .dockerignore).
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first so code edits do not re-download torch (~200 MB CPU wheel).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin duitku \
    && mkdir -p /data/receipts /data/hf && chown -R duitku:duitku /data
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations ./migrations
# /data is the volume: receipts + the HuggingFace cache (E5 model, ~470 MB, downloaded on first start).
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf \
    RECEIPT_DIR=/data/receipts
USER duitku
# No HEALTHCHECK: the bot only polls Telegram outbound (no port). `restart: unless-stopped` handles
# crashes; a process-level check would add no signal.
CMD ["python", "-m", "duitku.interfaces.telegram.bot"]
