"""Interface-agnostic turn runner: the one bridge between CLI/Telegram and the graph.

Resolves the app user from an external identity, builds the trusted `RunnableConfig`
(user_id/timezone/thread_id come from the DB row, never from message text), serializes turns per
thread, and turns any failure into a friendly reply. A turn that hits `interrupt()` (approval)
comes back with `TurnResult.pending` set; `resume()` answers it.
"""

import asyncio
import logging
import uuid
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.agent.graph import Graph, resume_turn, run_turn
from duitku.config import get_settings
from duitku.db.engine import get_sessionmaker
from duitku.db.repositories import users
from duitku.interfaces.run_log import record_run, run_status
from duitku.interfaces.trace import ModelStep, extract_trace, final_text
from duitku.observability.spans import RunStats, agent_run, last_trace_id
from duitku.receipts.ingest import ingest_receipt

logger = logging.getLogger(__name__)

ERROR_REPLY = "Maaf, ada gangguan saat memproses pesanmu. Coba lagi sebentar lagi ya."
NOTHING_PENDING_REPLY = "Tidak ada yang perlu dikonfirmasi saat ini."


@dataclass(frozen=True)
class UserCtx:
    user_id: uuid.UUID
    timezone: str


@dataclass(frozen=True)
class TurnResult:
    text: str
    trace: list[ModelStep] = field(default_factory=lambda: list[ModelStep]())
    failed: bool = False
    # Set when the graph is paused for the user's OK: the approval payload
    # {tool, args_summary, human_summary, tool_call_id}. `text` then holds its human_summary.
    # Answer with TurnRunner.resume(); one payload at a time (several risky calls = several rounds).
    pending: dict[str, Any] | None = None


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

    async def save_receipt(self, user: UserCtx, data: bytes) -> uuid.UUID:
        """Store a receipt image for the user (file + `receipts` row); raises ReceiptRejected for
        a bad type/size. The image goes to disk only, never into the graph."""
        return await ingest_receipt(user.user_id, data, sessionmaker=self._sessionmaker)

    async def pending_approval(self, thread_id: str, user: UserCtx) -> dict[str, Any] | None:
        """The approval payload the thread is paused on, else None (read-only)."""
        snapshot = await self._graph.aget_state(self._config(thread_id, user))
        return snapshot.interrupts[0].value if snapshot.interrupts else None

    def _lock(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = self._locks[thread_id] = asyncio.Lock()
        return lock

    async def _guarded(
        self,
        thread_id: str,
        user: UserCtx,
        config: RunnableConfig,
        turn: Callable[[], Awaitable[dict[str, Any]]],
        *,
        resuming: bool,
    ) -> TurnResult:
        """Run one graph call under the thread lock and shape its output (never raises).

        Guards both ways against a paused thread: a new message must not be appended after an
        unanswered tool_call (the API would reject the history), and `Command(resume)` on a thread
        that is not paused would re-run the graph from its start. A new message while paused
        re-asks the same question instead.
        """
        run: RunStats | None = None
        async with self._lock(thread_id):
            try:
                snapshot = await self._graph.aget_state(config)
                paused = snapshot.interrupts[0].value if snapshot.interrupts else None
                if paused is not None and not resuming:
                    return TurnResult(str(paused["human_summary"]), pending=paused)
                if paused is None and resuming:
                    return TurnResult(NOTHING_PENDING_REPLY)
                with agent_run(user.user_id, "resume" if resuming else "run") as run:
                    out = await turn()
            except Exception as e:  # boundary: an interface must always be able to answer
                logger.error(
                    "turn failed (%s)",
                    type(e).__name__,
                    exc_info=get_settings().log_content,
                    extra={"trace_id": last_trace_id()},
                )
                result = TurnResult(ERROR_REPLY, failed=True)
            else:
                result = self._shape(out)
        if run is not None:  # a graph call happened: one agent_runs row for this segment
            await record_run(
                self._sessionmaker,
                user.user_id,
                thread_id,
                run,
                run_status(result.failed, result.pending is not None, result.text),
            )
        return result

    @staticmethod
    def _shape(out: dict[str, Any]) -> TurnResult:
        messages = out["messages"]
        trace = extract_trace(messages)
        if interrupts := out.get("__interrupt__"):
            payload: dict[str, Any] = interrupts[0].value
            return TurnResult(str(payload["human_summary"]), trace, pending=payload)
        text = final_text(messages)
        if text is None:
            logger.error("turn produced no assistant text", extra={"trace_id": last_trace_id()})
            return TurnResult(ERROR_REPLY, trace, failed=True)
        return TurnResult(text, trace)

    @staticmethod
    def _config(thread_id: str, user: UserCtx) -> RunnableConfig:
        return {
            "configurable": {
                "user_id": str(user.user_id),
                "timezone": user.timezone,
                "thread_id": thread_id,
            }
        }

    async def run(self, thread_id: str, user: UserCtx, message: str) -> TurnResult:
        """One user message -> final reply + trace of this turn. Never raises on graph/LLM errors.

        Turns on the same thread run one after another (the checkpoint is read-modify-write, so
        interleaving would lose or duplicate messages); different threads run concurrently.
        """
        config = self._config(thread_id, user)
        return await self._guarded(
            thread_id,
            user,
            config,
            lambda: run_turn(self._graph, [HumanMessage(message)], config),
            resuming=False,
        )

    async def resume(
        self,
        thread_id: str,
        user: UserCtx,
        decision: Literal["approve", "reject"],
        reason: str | None = None,
    ) -> TurnResult:
        """Answer the thread's pending approval; the turn continues and may pause again.

        `reason` is accepted for the interface contract but not used yet: the graph only needs
        approve/reject (a rejection reaches the model as REJECTED_BY_USER).
        """
        config = self._config(thread_id, user)
        answer = {"decision": decision, "reason": reason}
        return await self._guarded(
            thread_id,
            user,
            config,
            lambda: resume_turn(self._graph, answer, config),
            resuming=True,
        )
