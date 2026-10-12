"""Tool `update_expense`: partial edit of ONE of the current user's expenses, by expense_id.

How the agent finds the id ("change the last one to 40rb"): from the add_expense result still in
the conversation, or from `query_expenses` rows (newest first). There is deliberately no
"last" shortcut: a tool that guesses which row to edit can silently change the wrong one.
"""

import uuid
from datetime import date, datetime
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.db.engine import get_sessionmaker
from duitku.db.models import Expense
from duitku.db.repositories import categories, expenses
from duitku.domain.errors import NotOwnedError
from duitku.domain.money import AmountError, parse_idr_amount
from duitku.tools._common import CategorySlug, unknown_category
from duitku.tools.add_expense import AmountText
from duitku.tools.base import (
    Envelope,
    audited,
    error,
    get_context,
    ok,
    validation_error_envelope,
)

DESCRIPTION = (
    "Change fields of ONE existing expense; only the fields you pass are changed. Get "
    "expense_id from add_expense or query_expenses (rows are newest first); never guess it. "
    "Pass amount as the user wrote it ('40rb'). merchant/note='' clears the field. "
    "Example: expense_id='<uuid>', amount='40rb'."
)

NOT_FOUND = error(
    "NOT_FOUND",
    "Expense not found.",
    "Use an expense_id from add_expense or query_expenses (group_by='none').",
)


def _snapshot(e: Expense, slug: str, tz: ZoneInfo) -> dict[str, Any]:
    return {
        "amount_idr": e.amount_idr,
        "category": slug,
        "merchant": e.merchant,
        "note": e.note,
        "occurred_on": e.occurred_at.astimezone(tz).date().isoformat(),
    }


def build_update_expense_tool(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> StructuredTool:
    async def update_expense(
        expense_id: Annotated[str, Field(description="Id from add_expense/query_expenses.")],
        config: RunnableConfig,
        amount: Annotated[AmountText | None, Field(description="New amount, as written.")] = None,
        category: Annotated[CategorySlug | None, Field(description="New category slug.")] = None,
        merchant: Annotated[str | None, Field(max_length=200)] = None,
        note: Annotated[str | None, Field(max_length=500)] = None,
        occurred_on: Annotated[date | None, Field(description="YYYY-MM-DD, user's tz.")] = None,
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
        changes: dict[str, Any] = {}
        if amount is not None:
            try:
                changes["amount_idr"] = parse_idr_amount(amount)
            except AmountError as e:
                return error(e.code, e.message, e.hint)
        if merchant is not None:
            changes["merchant"] = merchant.strip() or None
        if note is not None:
            changes["note"] = note.strip() or None
        if not (changes or category or occurred_on):
            return error(
                "INVALID_ARGUMENT",
                "Nothing to change.",
                "Pass at least one of amount, category, merchant, note, occurred_on.",
            )

        async with (sessionmaker or get_sessionmaker())() as session:
            row = await expenses.get_expense(session, ctx.user_id, eid)
            if row is None:  # missing, foreign and soft-deleted look the same: no existence leak
                return NOT_FOUND
            old_cat = await categories.get_category(session, ctx.user_id, row.category_id)
            old = _snapshot(row, old_cat.slug if old_cat else "", ctx.timezone)
            new_cat = old_cat
            if category:
                new_cat = await categories.get_by_slug(session, ctx.user_id, category)
                if new_cat is None:
                    return await unknown_category(session, ctx.user_id, category)
                changes["category_id"] = new_cat.id
            if occurred_on:  # keep the time of day so same-day ordering stays stable
                t = row.occurred_at.astimezone(ctx.timezone).time()
                changes["occurred_at"] = datetime.combine(occurred_on, t, ctx.timezone)
            try:
                updated = await expenses.update_expense(session, ctx.user_id, eid, changes)
                if updated is None:
                    return NOT_FOUND
                await session.commit()
            except NotOwnedError:
                return await unknown_category(session, ctx.user_id, category or "")
            except IntegrityError:
                return error(
                    "CONFLICT",
                    "The expense could not be updated.",
                    "Check the arguments and try once more.",
                )
            await session.refresh(updated)
            new = _snapshot(updated, new_cat.slug if new_cat else "", ctx.timezone)
        return ok({"expense_id": str(eid), "old": old, "new": new})

    return StructuredTool.from_function(
        coroutine=audited("update_expense", sessionmaker=sessionmaker)(update_expense),
        name="update_expense",
        description=DESCRIPTION,
        handle_validation_error=validation_error_envelope,
    )
