"""Tool `forget_memory`: delete ONE memory of the current user by id (no embedding needed)."""

import uuid
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.db.engine import get_sessionmaker
from duitku.memory import store
from duitku.tools.base import Envelope, audited, error, get_context, ok, validation_error_envelope

DESCRIPTION = (
    "Delete ONE saved memory by id (from recall_memory or save_memory). Use it when the user "
    "asks to forget something, or to remove a stale memory before saving its replacement. "
    "Never guess an id. Example: memory_id='<uuid>'."
)

NOT_FOUND = error(
    "NOT_FOUND",
    "Memory not found.",
    "Use a memory id returned by recall_memory or save_memory.",
)


def build_forget_memory_tool(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> StructuredTool:
    async def forget_memory(
        memory_id: Annotated[str, Field(description="Id from recall_memory/save_memory.")],
        config: RunnableConfig,
    ) -> Envelope:
        ctx = get_context(config)
        try:
            mid = uuid.UUID(memory_id)
        except ValueError:
            return error(
                "INVALID_ARGUMENT",
                "memory_id is not a valid id.",
                "Use a memory id returned by recall_memory or save_memory.",
            )
        async with (sessionmaker or get_sessionmaker())() as session:
            deleted = await store.forget(session, ctx.user_id, mid)
            await session.commit()
        # missing and foreign ids look the same: no existence leak
        return ok({"memory_id": str(mid), "forgotten": True}) if deleted else NOT_FOUND

    return StructuredTool.from_function(
        coroutine=audited("forget_memory", sessionmaker=sessionmaker)(forget_memory),
        name="forget_memory",
        description=DESCRIPTION,
        handle_validation_error=validation_error_envelope,
    )
