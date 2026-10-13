"""Graph nodes (docs/spec/agent.md). Each factory closes over its injectable dependencies."""

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.agent.prompts import build_system_prompt
from duitku.agent.state import AgentState
from duitku.agent.trimming import trim_history
from duitku.db.engine import get_sessionmaker
from duitku.db.repositories import categories
from duitku.domain.categories import DEFAULT_CATEGORIES
from duitku.tools.base import Envelope, error, get_context

logger = logging.getLogger(__name__)


class Node(Protocol):
    """Shape langgraph's add_node accepts for `(state, config)` nodes."""

    async def __call__(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]: ...


SlugLoader = Callable[[RunnableConfig], Awaitable[Sequence[str]]]
Clock = Callable[[], datetime]

STEP_LIMIT_FALLBACK = (
    "Maaf, aku butuh terlalu banyak langkah untuk permintaan ini dan berhenti di tengah jalan. "
    "Coba ulangi dengan lebih singkat atau pecah jadi beberapa pesan ya."
)
_DEFAULT_SLUGS = tuple(slug for slug, _ in DEFAULT_CATEGORIES)


def repo_slug_loader(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> SlugLoader:
    """Category slugs of config's user from the DB; falls back to the defaults on any failure
    (a prompt without exact slugs is worse than one with the default list, but not fatal)."""

    async def load(config: RunnableConfig) -> Sequence[str]:
        try:
            user_id = get_context(config).user_id
            async with (sessionmaker or get_sessionmaker())() as session:
                rows = await categories.list_categories(session, user_id)
            return [c.slug for c in rows] or _DEFAULT_SLUGS
        except Exception as e:  # broad on purpose: the prompt must still be buildable
            logger.warning("category slug lookup failed (%s); using defaults", type(e).__name__)
            return _DEFAULT_SLUGS

    return load


def make_agent_node(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    slugs: SlugLoader,
    now: Clock,
    max_history_tokens: int,
) -> Node:
    bound = llm.bind_tools(list(tools))

    async def agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = get_context(config)  # user_id/timezone: config only, never LLM args
        system = build_system_prompt(
            now=now(),
            timezone=ctx.timezone.key,
            category_slugs=await slugs(config),
            recalled_memories=state.get("recalled_memories", []),  # set by load_context
        )
        # Model view only: trimmed, with the system prompt prepended. Stored state stays whole.
        view: list[BaseMessage] = [
            SystemMessage(system),
            *trim_history(state["messages"], max_history_tokens),
        ]
        return {"messages": [await bound.ainvoke(view, config)]}

    return agent


def _tool_message(content: Envelope, call_id: str, name: str | None = None) -> ToolMessage:
    return ToolMessage(
        json.dumps(content, ensure_ascii=False), tool_call_id=call_id, name=name, status="error"
    )


def make_tools_node(
    tools: Sequence[BaseTool],
) -> Node:
    by_name = {t.name: t for t in tools}

    async def run_tools(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Invariant: EVERY tool_call_id of the last AIMessage gets exactly one ToolMessage.

        Unknown tool, crashing tool and unparseable args all become error observations the model
        can read. Sequential on purpose: calls in one message may depend on each other's side
        effects, and order keeps the transcript deterministic.
        """
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        out: list[ToolMessage] = []
        for call in last.tool_calls:
            call_id = call["id"]
            if not call_id:  # providers always send ids; nothing to pair a result with otherwise
                continue
            tool = by_name.get(call["name"])
            if tool is None:
                out.append(
                    _tool_message(
                        error("UNKNOWN_TOOL", "No such tool.", f"Valid tools: {sorted(by_name)}"),
                        call_id,
                        call["name"],
                    )
                )
                continue
            try:
                # Passing the whole ToolCall (not bare args) is what makes InjectedToolCallId work.
                result = await tool.ainvoke(call, config)
            except Exception as e:  # tools map their own errors; this is the last line of defence
                logger.error("tool %s raised %s", call["name"], type(e).__name__)
                result = _tool_message(
                    error("INTERNAL_ERROR", "Internal error.", "Do not retry. Tell the user."),
                    call_id,
                    call["name"],
                )
            assert isinstance(result, ToolMessage)
            out.append(result)
        for bad in last.invalid_tool_calls:
            if bad["id"]:  # without an id there is nothing to answer
                out.append(
                    _tool_message(
                        error("INVALID_ARGUMENT", "Arguments were not valid JSON.", "Resend."),
                        bad["id"],
                        bad["name"],
                    )
                )
        return {"messages": out}

    return run_tools


def make_limit_node() -> Node:
    async def limit(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Step cap hit with tool_calls pending: answer each id (not executed), then apologise."""
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        ids = [(c["id"], c["name"]) for c in (*last.tool_calls, *last.invalid_tool_calls)]
        refusals = [
            _tool_message(
                error(
                    "STEP_LIMIT", "Step limit reached; call not executed.", "Stop and apologise."
                ),
                call_id,
                name,
            )
            for call_id, name in ids
            if call_id
        ]
        return {"messages": [*refusals, AIMessage(STEP_LIMIT_FALLBACK)]}

    return limit
