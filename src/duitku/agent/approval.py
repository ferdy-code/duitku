"""`approval` node: ask the user before risky tool calls run (docs/spec/agent.md).

`interrupt()` pauses the graph; on resume the node RE-RUNS from its start and each `interrupt()`
call returns the stored answers in order. So everything before an interrupt must be repeatable:
only READ-ONLY lookups happen here, all side effects live in `tools`, after approval.

One interrupt per risky call, in order. A resume answers exactly one of them; if more risky
calls remain the graph pauses again (the caller sees a new pending payload).
"""

import json
import logging
import uuid
from typing import Any, cast

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.agent.approval_policy import needs_approval
from duitku.agent.nodes import Node
from duitku.agent.state import AgentState
from duitku.db.engine import get_sessionmaker
from duitku.db.repositories import categories, expenses
from duitku.domain.money import AmountError, parse_idr_amount
from duitku.tools.base import get_context

logger = logging.getLogger(__name__)

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des")
_MAX_ARGS_CHARS = 200


def rupiah(amount: int) -> str:
    return "Rp" + f"{amount:,}".replace(",", ".")


def is_approved(answer: object) -> bool:
    """Defensive: only an explicit `{"decision": "approve"}` approves; anything else rejects."""
    return isinstance(answer, dict) and cast(dict[str, Any], answer).get("decision") == "approve"


def _args_summary(args: dict[str, Any]) -> str:
    s = json.dumps(args, ensure_ascii=False, default=str)
    return s if len(s) <= _MAX_ARGS_CHARS else s[: _MAX_ARGS_CHARS - 1] + "…"


def _receipt_summary(args: dict[str, Any]) -> str:
    raw = str(args.get("amount", "?"))
    try:
        amount = rupiah(parse_idr_amount(raw))
    except AmountError:
        amount = raw  # shown as written; the tool will report the parse problem afterwards
    who = f" di {args['merchant']}" if args.get("merchant") else ""
    return f"Catat pengeluaran dari struk: {amount}{who} ({args.get('category', '?')})?"


async def _delete_summary(
    call: ToolCall,
    config: RunnableConfig,
    sessionmaker: async_sessionmaker[AsyncSession] | None,
) -> str | None:
    """READ-ONLY, ownership-scoped lookup. None = nothing to confirm (bad id / not the user's /
    already deleted): the tools node then returns the normal error observation."""
    try:
        eid = uuid.UUID(str(call["args"].get("expense_id")))
    except ValueError:
        return None
    ctx = get_context(config)
    try:
        async with (sessionmaker or get_sessionmaker())() as session:
            row = await expenses.get_expense(session, ctx.user_id, eid)
            if row is None:
                return None
            cat = await categories.get_category(session, ctx.user_id, row.category_id)
    except SQLAlchemyError as e:
        # Still ask (never delete unasked); the summary just lacks details.
        logger.warning("approval summary lookup failed (%s)", type(e).__name__)
        return "Hapus pengeluaran ini?"
    day = row.occurred_at.astimezone(ctx.timezone)
    label = row.merchant or row.note or (cat.slug if cat else "pengeluaran")
    return (
        f"Hapus pengeluaran: {label} {rupiah(row.amount_idr)} ({day.day} {_MONTHS[day.month - 1]})?"
    )


async def _human_summary(
    call: ToolCall,
    config: RunnableConfig,
    sessionmaker: async_sessionmaker[AsyncSession] | None,
) -> str | None:
    if call["name"] == "delete_expense":
        return await _delete_summary(call, config, sessionmaker)
    return _receipt_summary(call["args"])


def make_approval_node(sessionmaker: async_sessionmaker[AsyncSession] | None = None) -> Node:
    async def approval(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        rejected: list[str] = []
        for call in last.tool_calls:
            call_id = call["id"]
            if not call_id or not needs_approval(call):
                continue
            summary = await _human_summary(call, config, sessionmaker)
            if summary is None:
                continue
            answer = interrupt(
                {
                    "tool": call["name"],
                    "args_summary": _args_summary(call["args"]),
                    "human_summary": summary,
                    "tool_call_id": call_id,
                }
            )
            if not is_approved(answer):
                rejected.append(call_id)
        return {"rejected_call_ids": rejected}  # full list, no reducer (agent.md)

    return approval
