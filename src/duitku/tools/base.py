"""Tool plumbing: result envelope, trusted context, audited-call decorator.

Contract (docs/spec/tools.md): every tool returns `{"ok":true,"data":..}` or
`{"ok":false,"error":{code,message,hint}}`. Business/validation errors are observations the
model can react to, never raised. Unexpected exceptions become a generic INTERNAL_ERROR; the
LLM never sees exception text.

Usage: `StructuredTool.from_function(coroutine=audited("add_expense")(fn), name=...,
handle_validation_error=validation_error_envelope)`. (`@tool` does not accept
`handle_validation_error`.) functools.wraps keeps fn's signature, so LangChain still injects
`config` / `tool_call_id` and hides them from the LLM schema.
"""

import functools
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.config import get_settings
from duitku.db.engine import get_sessionmaker
from duitku.db.repositories import audit
from duitku.observability.redact import redact_args
from duitku.observability.spans import finish_tool, tool_call

logger = logging.getLogger(__name__)

Envelope = dict[str, Any]
INTERNAL_ERROR = "INTERNAL_ERROR"
INVALID_ARGUMENT = "INVALID_ARGUMENT"


# --- envelope -------------------------------------------------------------------------------


def ok(data: Any = None) -> Envelope:
    return {"ok": True, "data": data}


def error(code: str, message: str, hint: str = "") -> Envelope:
    return {"ok": False, "error": {"code": code, "message": message, "hint": hint}}


def _internal_error() -> Envelope:
    return error(
        INTERNAL_ERROR, "Internal error.", "Do not retry. Tell the user something went wrong."
    )


def _invalid(message: str) -> Envelope:
    return error(INVALID_ARGUMENT, message, "Fix the arguments and call the tool again.")


def _validation_message(e: ValidationError) -> str:
    # loc + msg only: pydantic's default str() echoes input values and a docs URL.
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'args'}: {err['msg']}"
        for err in e.errors(include_input=False, include_url=False)
    )


def validation_error_envelope(e: ValidationError) -> str:
    """`handle_validation_error` for `@tool`: LangChain validates args BEFORE our wrapper runs."""
    return json.dumps(_invalid(_validation_message(e)), ensure_ascii=False)


# --- trusted context ------------------------------------------------------------------------


class ContextError(RuntimeError):
    """`config["configurable"]` lacks a valid user_id/timezone: a wiring bug, not LLM input."""


@dataclass(frozen=True)
class ToolContext:
    user_id: uuid.UUID
    timezone: ZoneInfo


def get_context(config: RunnableConfig) -> ToolContext:
    """user_id/timezone come ONLY from RunnableConfig (CLAUDE.md hard rule), never tool args."""
    cfg = config.get("configurable") or {}
    try:
        raw_uid = cfg["user_id"]
        user_id = raw_uid if isinstance(raw_uid, uuid.UUID) else uuid.UUID(str(raw_uid))
        timezone = ZoneInfo(str(cfg["timezone"]))
    except (KeyError, ValueError) as e:  # ZoneInfoNotFoundError is a KeyError
        raise ContextError(
            f"config['configurable'] needs a valid user_id and timezone ({type(e).__name__})"
        ) from None
    return ToolContext(user_id, timezone)


# --- audit decorator ------------------------------------------------------------------------


async def _write_audit(
    sm: async_sessionmaker[AsyncSession] | None,
    user_id: uuid.UUID,
    **fields: Any,
) -> None:
    async with (sm or get_sessionmaker())() as session:
        await audit.record_tool_call(session, user_id, **fields)
        await session.commit()


def audited(
    tool_name: str, *, sessionmaker: async_sessionmaker[AsyncSession] | None = None
) -> Callable[[Callable[..., Awaitable[Envelope]]], Callable[..., Awaitable[Envelope]]]:
    """Wrap an async tool: map exceptions to envelopes and write one tool_audit_log row per call.

    The wrapped function must accept `config: RunnableConfig`. `tool_call_id` (if declared)
    is logged but not stored: tool_audit_log has no such column (see DECISIONS).
    """

    def decorator(fn: Callable[..., Awaitable[Envelope]]) -> Callable[..., Awaitable[Envelope]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Envelope:
            start = time.perf_counter()
            config: RunnableConfig = kwargs.get("config") or {}
            log_content = get_settings().log_content
            try:
                ctx = get_context(config)
            except ContextError as e:
                logger.error("tool %s: %s", tool_name, e)
                return _internal_error()  # no user_id -> nothing to attribute an audit row to
            with tool_call(tool_name) as span:
                try:
                    result = await fn(*args, **kwargs)
                except ValidationError as e:
                    result = _invalid(_validation_message(e))
                except ValueError as e:  # deliberate domain validation (e.g. amount parser)
                    result = _invalid(str(e) or "Invalid argument.")
                except Exception as e:
                    # Type only unless LOG_CONTENT=true: messages/tracebacks can embed user content.
                    logger.error(
                        "tool %s failed: %s", tool_name, type(e).__name__, exc_info=log_content
                    )
                    result = _internal_error()
                err_code = None if result["ok"] else result["error"]["code"]
                tool_args = {k: v for k, v in kwargs.items() if k not in ("config", "tool_call_id")}
                finish_tool(
                    span,
                    tool_name,
                    "ok" if result["ok"] else "error",
                    err_code,
                    args=tool_args,
                    log_content=log_content,
                )

            call_id = kwargs.get("tool_call_id")
            try:
                await _write_audit(
                    sessionmaker,
                    ctx.user_id,
                    tool_name=tool_name,
                    status="ok" if result["ok"] else "error",
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    args=redact_args(tool_args, log_content),
                    error_code=err_code,
                )
            except Exception as e:  # audit must never break the tool result
                logger.error(
                    "tool %s (call %s): audit write failed: %s",
                    tool_name,
                    call_id,
                    type(e).__name__,
                )
            return result

        return wrapper

    return decorator
