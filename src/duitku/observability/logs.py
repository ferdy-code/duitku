"""Structured JSON logs carrying `trace_id`/`span_id` of the active span.

One JSON object per line on stderr. Callers log class names only (hard rule); a traceback is
rendered only when the caller passed `exc_info`, which the code base does only for LOG_CONTENT=true.
A record may carry an explicit `trace_id` (via `extra=`) for code that logs after its span ended.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace


def active_ids() -> tuple[str, str] | None:
    """(trace_id, span_id) as hex of the current span, None if there is no valid one."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if ids := active_ids():
            out["trace_id"], out["span_id"] = ids
        if explicit := getattr(record, "trace_id", None):
            out["trace_id"] = explicit
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
