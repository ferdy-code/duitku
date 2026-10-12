"""Append-only logs: agent_runs and tool_audit_log. Insert + scoped read; no update/delete.

Callers decide redaction (LOG_CONTENT) before passing `args`; this layer stores what it is given.
"""

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from duitku.db.models import AgentRun, ToolAuditLog


async def record_agent_run(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    thread_id: str,
    model: str,
    steps: int,
    duration_ms: int,
    status: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> AgentRun:
    run = AgentRun(
        user_id=user_id,
        thread_id=thread_id,
        model=model,
        steps=steps,
        duration_ms=duration_ms,
        status=status,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    session.add(run)
    await session.flush()
    return run


async def list_agent_runs(
    session: AsyncSession, user_id: uuid.UUID, limit: int = 50
) -> Sequence[AgentRun]:
    q = (
        select(AgentRun)
        .where(AgentRun.user_id == user_id)
        .order_by(AgentRun.created_at.desc())
        .limit(limit)
    )
    return (await session.execute(q)).scalars().all()


async def record_tool_call(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    tool_name: str,
    status: str,
    latency_ms: int,
    args: dict[str, Any] | None = None,
    error_code: str | None = None,
) -> ToolAuditLog:
    row = ToolAuditLog(
        user_id=user_id,
        tool_name=tool_name,
        status=status,
        latency_ms=latency_ms,
        args=args,
        error_code=error_code,
    )
    session.add(row)
    await session.flush()
    return row


async def list_tool_calls(
    session: AsyncSession, user_id: uuid.UUID, limit: int = 50
) -> Sequence[ToolAuditLog]:
    q = (
        select(ToolAuditLog)
        .where(ToolAuditLog.user_id == user_id)
        .order_by(ToolAuditLog.created_at.desc())
        .limit(limit)
    )
    return (await session.execute(q)).scalars().all()
