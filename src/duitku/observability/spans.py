"""Span helpers: `agent.run` -> `agent.llm` / `agent.tool.<name>` plus per-run token/step stats.

Privacy: attributes hold only names, counts, model, status, error codes and a user hash. Message
text, tool args and exception messages never go in unless LOG_CONTENT=true (tool args only).
Exceptions are NOT recorded as span events (their text can embed content); only the class name.
"""

import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage
from opentelemetry.trace import Span, Status, StatusCode

from duitku.observability import metrics
from duitku.observability.logs import active_ids
from duitku.observability.redact import content_attr, user_hash
from duitku.observability.tracing import get_tracer

ToolStatus = Literal["ok", "error", "rejected"]


@dataclass
class RunStats:
    """What ONE graph call (a `run` or a `resume` segment) did. Counted at the source, call by
    call, so a resume segment never repeats the pre-pause calls (its message list would)."""

    steps: int = 0  # model calls
    input_tokens: int | None = None  # None = provider never reported usage
    output_tokens: int | None = None
    duration_ms: int = 0
    trace_id: str | None = None


_stats: ContextVar[RunStats | None] = ContextVar("duitku_run_stats", default=None)
_last_trace_id: ContextVar[str | None] = ContextVar("duitku_last_trace_id", default=None)


def last_trace_id() -> str | None:
    """Trace id of the active span, else of the last agent.run in this task (for error logs
    written after the span ended)."""
    ids = active_ids()
    return ids[0] if ids else _last_trace_id.get()


@contextmanager
def _span(name: str, attrs: dict[str, Any]) -> Generator[Span]:
    with get_tracer().start_as_current_span(
        name, attributes=attrs, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            yield span
        except BaseException as e:  # incl. cancellation: mark, never swallow
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("duitku.error.type", type(e).__name__)
            raise


@contextmanager
def agent_run(user_id: uuid.UUID, kind: Literal["run", "resume"]) -> Generator[RunStats]:
    stats = RunStats()
    token = _stats.set(stats)
    start = time.perf_counter()
    try:
        with _span(
            "agent.run", {"duitku.user_hash": user_hash(user_id), "duitku.run.kind": kind}
        ) as span:
            ids = active_ids()
            stats.trace_id = ids[0] if ids else None
            _last_trace_id.set(stats.trace_id)
            try:
                yield stats
            finally:
                stats.duration_ms = int((time.perf_counter() - start) * 1000)
                span.set_attribute("duitku.steps", stats.steps)
                metrics.record_run(stats.duration_ms, stats.steps)
    finally:
        _stats.reset(token)


@contextmanager
def llm_call(model: str) -> Generator[Span]:
    if (stats := _stats.get()) is not None:
        stats.steps += 1  # an attempt counts, even if the call then fails
    with _span("agent.llm", {"llm.model": model}) as span:
        yield span


def record_llm_usage(span: Span, message: BaseMessage) -> None:
    usage = message.usage_metadata if isinstance(message, AIMessage) else None
    if not usage:
        return
    n_in, n_out = usage["input_tokens"], usage["output_tokens"]
    span.set_attribute("llm.tokens.input", n_in)
    span.set_attribute("llm.tokens.output", n_out)
    metrics.record_tokens(n_in, n_out)
    if (stats := _stats.get()) is not None:
        stats.input_tokens = (stats.input_tokens or 0) + n_in
        stats.output_tokens = (stats.output_tokens or 0) + n_out


@contextmanager
def tool_call(name: str) -> Generator[Span]:
    with _span(f"agent.tool.{name}", {"duitku.tool.name": name}) as span:
        yield span


def finish_tool(
    span: Span,
    name: str,
    status: ToolStatus,
    error_code: str | None = None,
    *,
    args: dict[str, Any] | None = None,
    log_content: bool = False,
) -> None:
    span.set_attribute("duitku.tool.status", status)
    if error_code:
        span.set_attribute("duitku.tool.error_code", error_code)
    if status == "error":
        span.set_status(Status(StatusCode.ERROR))
        metrics.record_tool_error(name)
    elif status == "rejected":
        metrics.record_approval_rejected()
    if args is not None and (text := content_attr(args, log_content)) is not None:
        span.set_attribute("duitku.tool.args", text)


def tool_event(name: str, status: ToolStatus, error_code: str | None = None) -> None:
    """A tool call that never reached the tool body (user rejected it / unknown tool)."""
    with tool_call(name) as span:
        finish_tool(span, name, status, error_code)
