"""Graph v1: `agent <-> tools` loop (docs/spec/agent.md).

    START -> load_context -> agent --(tool_calls)--> tools -> agent ...
                               |         \\--(step cap hit)--> limit -> END
                               \\--(no tool_calls)--> END

`load_context` (T-305) recalls long-term memories once per run, before the first model call.
Extension point: `approval` goes between agent and tools (route_after_agent).
"""

# pyright: reportUnknownMemberType=false
# ^ langgraph's own signatures carry unparameterised generics (Command[Unknown], ...); this file is
#   the only boundary to the StateGraph API, so the relaxation stays local.

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.agent.context import make_load_context_node
from duitku.agent.nodes import (
    STEP_LIMIT_FALLBACK,
    Clock,
    SlugLoader,
    make_agent_node,
    make_limit_node,
    make_tools_node,
    repo_slug_loader,
)
from duitku.agent.routing import route_after_agent
from duitku.agent.state import AgentState
from duitku.config import get_settings
from duitku.memory.embeddings import Embedder
from duitku.tools import clock
from duitku.tools.registry import build_tools

logger = logging.getLogger(__name__)

# Backstop above the step counter in routing (which should always fire first): one agent call is
# 2 supersteps (agent + tools), plus the final limit node.
_RECURSION_SLACK = 5

type Graph = CompiledStateGraph[AgentState, Any, AgentState, AgentState]


def make_chat_model() -> ChatOpenAI:
    """OpenAI-compatible chat model; everything comes from settings."""
    s = get_settings()
    return ChatOpenAI(
        model=s.model_agent,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        timeout=s.llm_timeout_seconds,
        max_retries=1,  # agent.md: retry once on network errors
    )


def build_graph(
    *,
    llm: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    category_slugs: SlugLoader | None = None,
    now: Clock | None = None,
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    embedder: Embedder | None = None,
) -> Graph:
    """`sessionmaker`/`embedder` only feed `load_context` (None = real DB / real model, lazily).
    Tools get theirs through `build_tools(sm, embedder)`."""
    settings = get_settings()
    tools = build_tools() if tools is None else tools
    graph = StateGraph(AgentState)
    graph.add_node(
        "agent",
        make_agent_node(
            llm or make_chat_model(),
            tools,
            slugs=category_slugs or repo_slug_loader(),
            now=now or clock.now,
            max_history_tokens=settings.short_term_max_tokens,
        ),
    )
    graph.add_node("load_context", make_load_context_node(sessionmaker, embedder))
    graph.add_node("tools", make_tools_node(tools))
    graph.add_node("limit", make_limit_node())
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "agent")
    graph.add_conditional_edges(
        "agent", partial(route_after_agent, max_steps=settings.max_agent_steps)
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("limit", END)
    return graph.compile(checkpointer=checkpointer)


async def run_turn(
    graph: Graph, messages: list[AnyMessage], config: RunnableConfig
) -> dict[str, Any]:
    """Invoke the graph; a runaway loop becomes a polite reply instead of an exception (R-11).

    Normally the routing step cap ends the turn cleanly. GraphRecursionError is the backstop
    for bugs; in that case the stored thread may be left mid-loop, the caller only gets the
    fallback message (it is not written to the thread).
    """
    try:
        initial = AgentState(messages=messages, recalled_memories=[], rejected_call_ids=[])
        limit = 2 * get_settings().max_agent_steps + _RECURSION_SLACK
        return await graph.ainvoke(initial, {"recursion_limit": limit, **config})
    except GraphRecursionError:
        logger.error("graph recursion limit hit; step cap in routing did not fire")
        return {"messages": [*messages, AIMessage(STEP_LIMIT_FALLBACK)]}
