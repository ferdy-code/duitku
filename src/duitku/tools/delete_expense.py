"""Tool `delete_expense`: soft-delete ONE of the current user's expenses, by expense_id.

The graph's `approval` node asks the user to confirm BEFORE this runs (REQUIRES_APPROVAL), so the
tool itself does not ask again. Ownership is enforced by the repository query (user_id comes
from RunnableConfig); missing, foreign and already-deleted ids all look the same (NOT_FOUND).
"""

import uuid
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.db.engine import get_sessionmaker
from duitku.db.repositories import categories, expenses
from duitku.tools.base import (
    Envelope,
    audited,
    error,
    get_context,
    ok,
    validation_error_envelope,
)

DESCRIPTION = (
    "Delete ONE expense (soft delete). The user is asked to confirm first, so just call it; do "
    "not ask for confirmation yourself. Get expense_id from add_expense or query_expenses "
    "(rows are newest first); never guess it. Example: expense_id='<uuid>'."
)

NOT_FOUND = error(
    "NOT_FOUND",
    "Expense not found.",
    "Use an expense_id from add_expense or query_expenses (group_by='none').",
)


def build_delete_expense_tool(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> StructuredTool:
    async def delete_expense(
        expense_id: Annotated[str, Field(description="Id from add_expense/query_expenses.")],
        config: RunnableConfig,
    ) -> Envelope:
        ctx = get_context(config)
        try:
            eid = uuid.UUID(expense_id)
        except ValueError:
            return error(
                "INVALID_ARGUMENT",
                "expense_id is not a valid id.",
                "Use an expense_id from add_expense or query_expenses.",
            )
        async with (sessionmaker or get_sessionmaker())() as session:
            row = await expenses.get_expense(session, ctx.user_id, eid)
            if row is None or not await expenses.soft_delete_expense(session, ctx.user_id, eid):
                return NOT_FOUND
            cat = await categories.get_category(session, ctx.user_id, row.category_id)
            await session.commit()
            return ok(
                {
                    "expense_id": str(eid),
                    "amount_idr": row.amount_idr,
                    "category": cat.slug if cat else None,
                    "merchant": row.merchant,
                    "note": row.note,
                    "occurred_on": row.occurred_at.astimezone(ctx.timezone).date().isoformat(),
                    "deleted": True,
                }
            )

    return StructuredTool.from_function(
        coroutine=audited("delete_expense", sessionmaker=sessionmaker)(delete_expense),
        name="delete_expense",
        description=DESCRIPTION,
        handle_validation_error=validation_error_envelope,
    )
