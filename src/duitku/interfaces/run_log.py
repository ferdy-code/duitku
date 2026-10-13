"""`agent_runs` rows: one per graph call (a `run` or a `resume` segment).

Policy (DECISIONS): a turn that pauses for approval gets a row with status `interrupted` at the
pause; its `resume` gets its own row. Every row counts only what ITS segment did (steps and tokens
are counted per model call at the source, see observability/spans.py), so summing a thread's rows
gives the true cost and nothing is counted twice. Never raises: observability must not fail a turn.
"""

import logging
import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.agent.nodes import STEP_LIMIT_FALLBACK
from duitku.config import get_settings
from duitku.db.engine import get_sessionmaker
from duitku.db.repositories import audit
from duitku.observability.spans import RunStats

logger = logging.getLogger(__name__)


def run_status(failed: bool, pending: bool, text: str) -> str:
    """agent_runs.status: ok | interrupted | step_limit | error."""
    if failed:
        return "error"
    if pending:
        return "interrupted"
    return "step_limit" if text == STEP_LIMIT_FALLBACK else "ok"


async def record_run(
    sessionmaker: async_sessionmaker[AsyncSession] | None,
    user_id: uuid.UUID,
    thread_id: str,
    stats: RunStats,
    status: str,
) -> None:
    try:
        async with (sessionmaker or get_sessionmaker())() as session:
            await audit.record_agent_run(
                session,
                user_id,
                thread_id=thread_id,
                model=get_settings().model_agent,
                steps=stats.steps,
                duration_ms=stats.duration_ms,
                status=status,
                input_tokens=stats.input_tokens,
                output_tokens=stats.output_tokens,
            )
            await session.commit()
    except (SQLAlchemyError, OSError) as e:
        logger.error(
            "agent_runs write failed (%s)", type(e).__name__, extra={"trace_id": stats.trace_id}
        )
