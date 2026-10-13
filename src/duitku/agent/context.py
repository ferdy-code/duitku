"""`load_context` node (docs/spec/agent.md): automatic recall of long-term memories.

Takes the last user message, recalls the top-k similar memories of config's user and puts their
text into `recalled_memories` (rendered into the prompt as DATA). Recall is an enhancement, never a
dependency: embedding failure, DB error or timeout -> log the exception class only (no message
content, LOG_CONTENT=false rule) and continue with no memories.

The embedder already runs the model on its own executor thread (memory/embeddings.py), so
`asyncio.wait_for` can really time out without the event loop being blocked. A timed-out encode
keeps running in that thread until done; only its result is dropped.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.agent.nodes import Node
from duitku.agent.state import AgentState
from duitku.config import get_settings
from duitku.db.engine import get_sessionmaker
from duitku.memory import store
from duitku.memory.embeddings import Embedder, EmbeddingError
from duitku.memory.thresholds import recall_threshold
from duitku.tools._memory import resolve_embedder
from duitku.tools.base import get_context

logger = logging.getLogger(__name__)


def last_human_text(messages: Sequence[AnyMessage]) -> str:
    """Text of the LAST HumanMessage ('' if none). List-style content: join its text parts."""
    for m in reversed(messages):
        if not isinstance(m, HumanMessage):
            continue
        if isinstance(m.content, str):
            return m.content
        parts = (p if isinstance(p, str) else p.get("text", "") for p in m.content)
        return " ".join(p for p in parts if isinstance(p, str))
    return ""


def make_load_context_node(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    embedder: Embedder | None = None,
    timeout_seconds: float | None = None,
) -> Node:
    async def recall(user_query: str, config: RunnableConfig) -> list[str]:
        settings = get_settings()
        async with (sessionmaker or get_sessionmaker())() as session:
            hits = await store.recall(
                session,
                get_context(config).user_id,  # config only, never LLM args
                user_query,
                resolve_embedder(embedder),
                top_k=settings.memory_recall_top_k,
                min_similarity=recall_threshold(settings),
            )
            await session.commit()  # persists recall_count / last_recalled_at
        return [h.content for h in hits]

    async def load_context(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        # rejected_call_ids is reset on every run: a stale list from a finished turn must not
        # block the next turn's tool calls (T-201 note). Plain-overwrite state field.
        out: dict[str, Any] = {"recalled_memories": [], "rejected_call_ids": []}
        query = last_human_text(state["messages"])
        if not query.strip():
            return out
        timeout = timeout_seconds or get_settings().memory_recall_timeout_seconds
        try:
            out["recalled_memories"] = await asyncio.wait_for(recall(query, config), timeout)
        except (EmbeddingError, TimeoutError, SQLAlchemyError) as e:
            # CancelledError is a BaseException: never caught here.
            logger.warning("memory recall failed (%s); continuing without", type(e).__name__)
        return out

    return load_context
