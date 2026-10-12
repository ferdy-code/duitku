"""Read path behind `query_expenses`: all aggregation happens in SQL, always scoped by user_id.

Dates are the user's LOCAL calendar days: `timezone(tz, occurred_at)` converts the stored
timestamptz to local wall time before truncating, so "today" means the user's today.
"""

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from sqlalchemy import ColumnElement, Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from duitku.db.models import Category, Expense

GroupBy = Literal["none", "category", "day", "week", "month"]


@dataclass(frozen=True)
class Group:
    key: str  # category slug | local date of the bucket start (day/week) | "YYYY-MM" (month)
    name: str | None  # category display name; None for time buckets
    total_idr: int
    count: int


@dataclass(frozen=True)
class ExpenseRow:
    id: uuid.UUID
    occurred_on: date  # local date
    amount_idr: int
    category_slug: str
    merchant: str | None
    note: str | None


def _filters(
    user_id: uuid.UUID, start: datetime, end: datetime, category_id: uuid.UUID | None
) -> list[ColumnElement[bool]]:
    """`start` inclusive, `end` exclusive (callers pass local-midnight bounds)."""
    conds = [
        Expense.user_id == user_id,
        Expense.deleted_at.is_(None),
        Expense.occurred_at >= start,
        Expense.occurred_at < end,
    ]
    if category_id is not None:
        conds.append(Expense.category_id == category_id)
    return conds


async def totals(
    session: AsyncSession,
    user_id: uuid.UUID,
    start: datetime,
    end: datetime,
    category_id: uuid.UUID | None = None,
) -> tuple[int, int]:
    """(sum of amount_idr, row count) over the whole range, independent of any row limit."""
    q = select(func.coalesce(func.sum(Expense.amount_idr), 0), func.count(Expense.id)).where(
        *_filters(user_id, start, end, category_id)
    )
    total, count = (await session.execute(q)).one()
    return int(total), int(count)


async def grouped(
    session: AsyncSession,
    user_id: uuid.UUID,
    start: datetime,
    end: datetime,
    group_by: Literal["category", "day", "week", "month"],
    tz: str,
    category_id: uuid.UUID | None = None,
) -> list[Group]:
    """Category groups: biggest total first. Time buckets: chronological (weeks start Monday)."""
    total = func.sum(Expense.amount_idr).label("total")
    count = func.count(Expense.id).label("n")
    conds = _filters(user_id, start, end, category_id)
    if group_by == "category":
        q = (
            select(Category.slug.label("key"), Category.name.label("name"), total, count)
            .join(
                Category,
                (Category.user_id == Expense.user_id) & (Category.id == Expense.category_id),
            )
            .where(*conds)
            .group_by(Category.slug, Category.name)
            .order_by(total.desc(), Category.slug)
        )
    else:
        # Grouped by the label (not the expression): the tz bind parameter would otherwise be
        # a different $n in SELECT and GROUP BY and Postgres rejects the query.
        bucket = cast(func.date_trunc(group_by, func.timezone(tz, Expense.occurred_at)), Date)
        key = bucket.label("key")
        q = select(key, total, count).where(*conds).group_by(key).order_by(key)
    rows = (await session.execute(q)).all()
    if group_by == "category":
        return [Group(r.key, r.name, int(r.total), int(r.n)) for r in rows]
    fmt = "%Y-%m" if group_by == "month" else "%Y-%m-%d"
    return [Group(r.key.strftime(fmt), None, int(r.total), int(r.n)) for r in rows]


async def rows(
    session: AsyncSession,
    user_id: uuid.UUID,
    start: datetime,
    end: datetime,
    tz: str,
    limit: int,
    category_id: uuid.UUID | None = None,
) -> list[ExpenseRow]:
    """Newest first (id breaks ties so the order is stable)."""
    q = (
        select(
            Expense.id,
            cast(func.timezone(tz, Expense.occurred_at), Date).label("on"),
            Expense.amount_idr,
            Category.slug,
            Expense.merchant,
            Expense.note,
        )
        .join(
            Category, (Category.user_id == Expense.user_id) & (Category.id == Expense.category_id)
        )
        .where(*_filters(user_id, start, end, category_id))
        .order_by(Expense.occurred_at.desc(), Expense.id)
        .limit(limit)
    )
    return [ExpenseRow(*r) for r in (await session.execute(q)).all()]
