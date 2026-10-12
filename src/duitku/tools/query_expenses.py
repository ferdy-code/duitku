"""Tool `query_expenses`: the ONLY read path for expenses. Aggregation is done in SQL."""

from datetime import date, datetime, time, timedelta
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.db.engine import get_sessionmaker
from duitku.db.repositories import categories, expense_queries
from duitku.db.repositories.expense_queries import GroupBy
from duitku.tools._common import CategorySlug, unknown_category
from duitku.tools.base import (
    Envelope,
    audited,
    error,
    get_context,
    ok,
    validation_error_envelope,
)

DESCRIPTION = (
    "Summarize the user's expenses between two dates (inclusive, user's timezone). Never add up "
    "numbers yourself; use total_idr/count/groups. group_by: 'none' returns the newest "
    "rows (up to limit); 'category', 'day', 'week' (Monday start) or 'month' returns per-group "
    "totals. For relative periods call resolve_period first. "
    "Example: start_date='2026-10-01', end_date='2026-10-31', group_by='category'."
)


def _bounds(start: date, end: date, tz: Any) -> tuple[datetime, datetime]:
    """Local-midnight [start, end+1day) as aware datetimes, so DST/offsets are handled by tz."""
    return (
        datetime.combine(start, time.min, tz),
        datetime.combine(end + timedelta(days=1), time.min, tz),
    )


def build_query_expenses_tool(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> StructuredTool:
    async def query_expenses(
        start_date: Annotated[date, Field(description="First day, YYYY-MM-DD, inclusive.")],
        end_date: Annotated[date, Field(description="Last day, YYYY-MM-DD, inclusive.")],
        config: RunnableConfig,
        category: Annotated[CategorySlug | None, Field(description="Only this category.")] = None,
        group_by: GroupBy = "none",
        limit: Annotated[int, Field(ge=1, le=50, description="Max rows (group_by='none').")] = 20,
    ) -> Envelope:
        ctx = get_context(config)
        if end_date < start_date:
            return error(
                "INVALID_ARGUMENT",
                "end_date is before start_date.",
                "Swap them, or call resolve_period for relative periods.",
            )
        try:
            start, end = _bounds(start_date, end_date, ctx.timezone)
        except OverflowError:
            return error("INVALID_ARGUMENT", "Date out of range.", "Use realistic dates.")
        tz = ctx.timezone.key

        async with (sessionmaker or get_sessionmaker())() as session:
            category_id = None
            if category is not None:
                cat = await categories.get_by_slug(session, ctx.user_id, category)
                if cat is None:
                    return await unknown_category(session, ctx.user_id, category)
                category_id = cat.id
            total, count = await expense_queries.totals(
                session, ctx.user_id, start, end, category_id
            )
            groups: list[dict[str, Any]] = []
            rows: list[dict[str, Any]] = []
            if group_by == "none":
                rows = [
                    {
                        "expense_id": str(r.id),
                        "occurred_on": r.occurred_on.isoformat(),
                        "amount_idr": r.amount_idr,
                        "category": r.category_slug,
                        "merchant": r.merchant,
                        "note": r.note,
                    }
                    for r in await expense_queries.rows(
                        session, ctx.user_id, start, end, tz, limit, category_id
                    )
                ]
            else:
                for g in await expense_queries.grouped(
                    session, ctx.user_id, start, end, group_by, tz, category_id
                ):
                    item: dict[str, Any] = {
                        "key": g.key,
                        "total_idr": g.total_idr,
                        "count": g.count,
                    }
                    if g.name is not None:
                        item["name"] = g.name
                    groups.append(item)
        return ok(
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "category": category,
                "group_by": group_by,
                "total_idr": total,
                "count": count,
                "groups": groups,
                "rows": rows,
                # rows only: total_idr/count always cover the whole range.
                "rows_truncated": group_by == "none" and count > len(rows),
            }
        )

    return StructuredTool.from_function(
        coroutine=audited("query_expenses", sessionmaker=sessionmaker)(query_expenses),
        name="query_expenses",
        description=DESCRIPTION,
        handle_validation_error=validation_error_envelope,
    )
