"""Budgets: one monthly limit (int rupiah) per (user, category)."""

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from duitku.db.models import Budget
from duitku.db.repositories.categories import require_category


async def upsert_budget(
    session: AsyncSession, user_id: uuid.UUID, category_id: uuid.UUID, monthly_limit_idr: int
) -> Budget:
    if monthly_limit_idr <= 0:
        raise ValueError("monthly_limit_idr must be > 0")
    await require_category(session, user_id, category_id)
    stmt = (
        pg_insert(Budget)
        .values(user_id=user_id, category_id=category_id, monthly_limit_idr=monthly_limit_idr)
        .on_conflict_do_update(
            index_elements=["user_id", "category_id"],
            set_={"monthly_limit_idr": monthly_limit_idr},
        )
        .returning(Budget)
        .execution_options(populate_existing=True)  # refresh an already-loaded identity
    )
    return (await session.execute(stmt)).scalar_one()


async def get_budget(
    session: AsyncSession, user_id: uuid.UUID, category_id: uuid.UUID
) -> Budget | None:
    q = select(Budget).where(Budget.user_id == user_id, Budget.category_id == category_id)
    return (await session.execute(q)).scalar_one_or_none()


async def list_budgets(session: AsyncSession, user_id: uuid.UUID) -> Sequence[Budget]:
    return (await session.execute(select(Budget).where(Budget.user_id == user_id))).scalars().all()


async def delete_budget(session: AsyncSession, user_id: uuid.UUID, category_id: uuid.UUID) -> bool:
    q = (
        delete(Budget)
        .where(Budget.user_id == user_id, Budget.category_id == category_id)
        .returning(Budget.id)
    )
    return (await session.execute(q)).scalar_one_or_none() is not None
