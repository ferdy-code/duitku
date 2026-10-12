"""Interface-agnostic turn runner: the one bridge between CLI/Telegram and the graph.

Resolves the app user from an external identity, builds the trusted `RunnableConfig`
(user_id/timezone/thread_id come from the DB row, never from message text), serializes turns per
thread, and turns any failure into a friendly reply. Approval/resume is added with T-4xx.
"""

import asyncio
import logging
import uuid
import weakref
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.agent.graph import Graph, run_turn
from duitku.config import get_settings
from duitku.db.engine import get_sessionmaker
from duitku.db.repositories import users
from duitku.interfaces.trace import ModelStep, extract_trace, final_text

logger = logging.getLogger(__name__)

ERROR_REPLY = "Maaf, ada gangguan saat memproses pesanmu. Coba lagi sebentar lagi ya."


@dataclass(frozen=True)
class UserCtx:
    user_id: uuid.UUID
    timezone: str


@dataclass(frozen=True)
class TurnResult:
    text: str
    trace: list[ModelStep] = field(default_factory=lambda: list[ModelStep]())
    failed: bool = False


class TurnRunner:
    def __init__(
        self, graph: Graph, sessionmaker: async_sessionmaker[AsyncSession] | None = None
    ) -> None:
        self._graph = graph
        self._sessionmaker = sessionmaker
        # Weak values: a lock disappears once no turn holds or waits on it, so the dict does not
        # grow with every thread ever seen. Safe because a waiter keeps a strong reference.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    async def resolve_user(self, telegram_user_id: int, display_name: str | None = None) -> UserCtx:
        """Get or create the app user (new users get default categories, in one transaction)."""
        async with (self._sessionmaker or get_sessionmaker())() as session:
            user, created = await users.get_or_create_user(session, telegram_user_id, display_name)
            tz = get_settings().default_timezone
            if created and user.timezone != tz:  # DB column default is fixed; DEFAULT_TIMEZONE wins
                user = await users.update_user(session, user.id, timezone=tz) or user
            await session.commit()
            return UserCtx(user.id, user.timezone)

    def _lock(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = self._locks[thread_id] = asyncio.Lock()
        return lock

    async def run(self, thread_id: str, user: UserCtx, message: str) -> TurnResult:
        """One user message -> final reply + trace of this turn. Never raises on graph/LLM errors.

        Turns on the same thread run one after another (the checkpoint is read-modify-write, so
        interleaving would lose or duplicate messages); different threads run concurrently.
        """
        config: RunnableConfig = {
            "configurable": {
                "user_id": str(user.user_id),
                "timezone": user.timezone,
                "thread_id": thread_id,
            }
        }
        async with self._lock(thread_id):
            try:
                out = await run_turn(self._graph, [HumanMessage(message)], config)
            except Exception as e:  # boundary: an interface must always be able to answer
                logger.error(
                    "turn failed (%s)", type(e).__name__, exc_info=get_settings().log_content
                )
                return TurnResult(ERROR_REPLY, failed=True)
        messages = out["messages"]
        text = final_text(messages)
        if text is None:
            logger.error("turn produced no assistant text")
            return TurnResult(ERROR_REPLY, extract_trace(messages), failed=True)
        return TurnResult(text, extract_trace(messages))
