"""Expenses. Money is int rupiah. Soft delete only; deleted rows are invisible to every read."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from duitku.db.models import Expense
from duitku.db.repositories.categories import require_category
from duitku.db.repositories.receipts import require_receipt

UPDATABLE = frozenset({"amount_idr", "category_id", "merchant", "note", "occurred_at"})


@dataclass(frozen=True)
class InsertResult:
    expense: Expense
    created: bool  # False => an earlier call with the same idempotency_key already stored it


async def insert_expense(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    amount_idr: int,
    category_id: uuid.UUID,
    occurred_at: datetime,
    source: str,
    merchant: str | None = None,
    note: str | None = None,
    receipt_id: uuid.UUID | None = None,
    items: list[Any] | None = None,
    idempotency_key: str | None = None,
) -> InsertResult:
    """Insert once per (user_id, idempotency_key); a retry returns the stored row, no exception.

    Note: a replay ignores the new arguments (the first write wins).
    """
    if amount_idr <= 0:
        raise ValueError("amount_idr must be > 0")
    await require_category(session, user_id, category_id)
    if receipt_id is not None:
        await require_receipt(session, user_id, receipt_id)
    stmt = (
        pg_insert(Expense)
        .values(
            user_id=user_id,
            amount_idr=amount_idr,
            category_id=category_id,
            occurred_at=occurred_at,
            source=source,
            merchant=merchant,
            note=note,
            receipt_id=receipt_id,
            items=items,
            idempotency_key=idempotency_key,
        )
        .on_conflict_do_nothing(
            index_elements=["user_id", "idempotency_key"],
            index_where=Expense.idempotency_key.is_not(None),
        )
        .returning(Expense)
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is not None:
        return InsertResult(row, True)
    q = select(Expense).where(
        Expense.user_id == user_id, Expense.idempotency_key == idempotency_key
    )
    return InsertResult((await session.execute(q)).scalar_one(), False)


async def get_expense(
    session: AsyncSession, user_id: uuid.UUID, expense_id: uuid.UUID
) -> Expense | None:
    q = select(Expense).where(
        Expense.user_id == user_id, Expense.id == expense_id, Expense.deleted_at.is_(None)
    )
    return (await session.execute(q)).scalar_one_or_none()


async def list_expenses(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    category_id: uuid.UUID | None = None,
    limit: int = 50,
) -> Sequence[Expense]:
    """Newest first; `start` inclusive, `end` exclusive. Aggregations belong to query_expenses."""
    q = select(Expense).where(Expense.user_id == user_id, Expense.deleted_at.is_(None))
    if start is not None:
        q = q.where(Expense.occurred_at >= start)
    if end is not None:
        q = q.where(Expense.occurred_at < end)
    if category_id is not None:
        q = q.where(Expense.category_id == category_id)
    q = q.order_by(Expense.occurred_at.desc()).limit(limit)
    return (await session.execute(q)).scalars().all()


async def update_expense(
    session: AsyncSession, user_id: uuid.UUID, expense_id: uuid.UUID, changes: dict[str, Any]
) -> Expense | None:
    """Apply whitelisted `changes`; None when the row is not this user's (or is deleted)."""
    unknown = changes.keys() - UPDATABLE
    if unknown or not changes:
        raise ValueError(f"changes must be a non-empty subset of {sorted(UPDATABLE)}")
    if "amount_idr" in changes and changes["amount_idr"] <= 0:
        raise ValueError("amount_idr must be > 0")
    if "category_id" in changes:
        await require_category(session, user_id, changes["category_id"])
    q = (
        update(Expense)
        .where(Expense.user_id == user_id, Expense.id == expense_id, Expense.deleted_at.is_(None))
        .values(**changes, updated_at=func.now())
        .returning(Expense)
    )
    return (await session.execute(q)).scalar_one_or_none()


async def soft_delete_expense(
    session: AsyncSession, user_id: uuid.UUID, expense_id: uuid.UUID
) -> bool:
    q = (
        update(Expense)
        .where(Expense.user_id == user_id, Expense.id == expense_id, Expense.deleted_at.is_(None))
        .values(deleted_at=func.now(), updated_at=func.now())
        .returning(Expense.id)
    )
    return (await session.execute(q)).scalar_one_or_none() is not None
