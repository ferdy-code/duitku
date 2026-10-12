"""Tool `add_expense`: parse amount text, validate category, save once per tool_call_id."""

import uuid
from datetime import date, datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, StructuredTool
from pydantic import BeforeValidator, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.db.engine import get_sessionmaker
from duitku.db.models import Expense
from duitku.db.repositories import categories, expenses
from duitku.domain.errors import NotOwnedError
from duitku.domain.money import AmountError, parse_idr_amount
from duitku.tools import clock
from duitku.tools._common import CategorySlug, unknown_category
from duitku.tools.base import (
    Envelope,
    audited,
    error,
    get_context,
    ok,
    validation_error_envelope,
)

# Models sometimes send 35000 instead of "35000"; accept it but keep the schema `string`.
AmountText = Annotated[
    str,
    BeforeValidator(lambda v: str(v) if isinstance(v, int) and not isinstance(v, bool) else v),
]

DESCRIPTION = (
    "Save ONE expense. Pass the amount exactly as the user wrote it ('35rb', '1,5 juta', "
    "'Rp 35.000'); it is parsed for you, never convert it yourself. category is a slug "
    "(makan-minum, transport, belanja, tagihan, hiburan, kesehatan, pendidikan, lainnya). "
    "occurred_on (YYYY-MM-DD) defaults to today in the user's timezone. "
    "Example: amount='35rb', category='makan-minum', merchant='Warteg Bahari'."
)


def _view(e: Expense, slug: str, tz_key: str, duplicate: bool) -> dict[str, Any]:
    return {
        "expense_id": str(e.id),
        "amount_idr": e.amount_idr,
        "category": slug,
        "merchant": e.merchant,
        "note": e.note,
        "occurred_on": e.occurred_at.astimezone(ZoneInfo(tz_key)).date().isoformat(),
        "source": e.source,
        # True => this tool_call_id was already saved; the stored row is returned unchanged.
        "duplicate": duplicate,
    }


def build_add_expense_tool(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> StructuredTool:
    async def add_expense(
        amount: Annotated[AmountText, Field(description="Amount as the user wrote it.")],
        category: Annotated[CategorySlug, Field(description="Category slug.")],
        config: RunnableConfig,
        tool_call_id: Annotated[str, InjectedToolCallId],
        merchant: Annotated[str | None, Field(max_length=200)] = None,
        note: Annotated[str | None, Field(max_length=500)] = None,
        occurred_on: Annotated[date | None, Field(description="YYYY-MM-DD, user's tz.")] = None,
        source: Literal["text", "receipt"] = "text",
        receipt_id: Annotated[str | None, Field(description="Only for source='receipt'.")] = None,
        items: Annotated[list[dict[str, Any]] | None, Field(max_length=100)] = None,
    ) -> Envelope:
        ctx = get_context(config)
        try:
            amount_idr = parse_idr_amount(amount)
        except AmountError as e:
            return error(e.code, e.message, e.hint)
        rid: uuid.UUID | None = None
        if receipt_id is not None:
            try:
                rid = uuid.UUID(receipt_id)
            except ValueError:
                return error(
                    "INVALID_ARGUMENT",
                    "receipt_id is not a valid id.",
                    "Use the receipt_id given in the conversation.",
                )

        now_local = clock.now().astimezone(ctx.timezone)
        day = occurred_on or now_local.date()
        # Keep the current time of day so same-day expenses sort in the order they were added.
        occurred_at = datetime.combine(day, now_local.time(), ctx.timezone)

        async with (sessionmaker or get_sessionmaker())() as session:
            cat = await categories.get_by_slug(session, ctx.user_id, category)
            if cat is None:
                return await unknown_category(session, ctx.user_id, category)
            try:
                res = await expenses.insert_expense(
                    session,
                    ctx.user_id,
                    amount_idr=amount_idr,
                    category_id=cat.id,
                    occurred_at=occurred_at,
                    source=source,
                    merchant=merchant,
                    note=note,
                    receipt_id=rid,
                    items=items,
                    idempotency_key=tool_call_id,
                )
                await session.commit()
            except NotOwnedError:
                return error(
                    "NOT_FOUND", "Receipt not found.", "Use a receipt_id from this conversation."
                )
            except IntegrityError:
                return error(
                    "CONFLICT",
                    "The expense could not be saved.",
                    "Check the arguments and try once more.",
                )
            # A replay returns the STORED row, whose category may differ from this call's.
            stored = (
                cat
                if res.created
                else await categories.get_category(session, ctx.user_id, res.expense.category_id)
            )
        slug = stored.slug if stored else cat.slug
        return ok(_view(res.expense, slug, ctx.timezone.key, duplicate=not res.created))

    return StructuredTool.from_function(
        coroutine=audited("add_expense", sessionmaker=sessionmaker)(add_expense),
        name="add_expense",
        description=DESCRIPTION,
        handle_validation_error=validation_error_envelope,
    )
