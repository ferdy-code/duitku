"""Categories. Seeded per user on creation; every query is scoped by `user_id`."""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from duitku.db.models import Category
from duitku.domain.categories import DEFAULT_CATEGORIES
from duitku.domain.errors import NotOwnedError


async def seed_defaults(session: AsyncSession, user_id: uuid.UUID) -> None:
    """Idempotent: re-running leaves existing rows untouched."""
    stmt = (
        pg_insert(Category)
        .values([{"user_id": user_id, "slug": s, "name": n} for s, n in DEFAULT_CATEGORIES])
        .on_conflict_do_nothing(index_elements=["user_id", "slug"])
    )
    await session.execute(stmt)


async def list_categories(session: AsyncSession, user_id: uuid.UUID) -> Sequence[Category]:
    q = select(Category).where(Category.user_id == user_id).order_by(Category.slug)
    return (await session.execute(q)).scalars().all()


async def get_category(
    session: AsyncSession, user_id: uuid.UUID, category_id: uuid.UUID
) -> Category | None:
    q = select(Category).where(Category.user_id == user_id, Category.id == category_id)
    return (await session.execute(q)).scalar_one_or_none()


async def get_by_slug(session: AsyncSession, user_id: uuid.UUID, slug: str) -> Category | None:
    q = select(Category).where(Category.user_id == user_id, Category.slug == slug)
    return (await session.execute(q)).scalar_one_or_none()


async def require_category(
    session: AsyncSession, user_id: uuid.UUID, category_id: uuid.UUID
) -> Category:
    """FKs only point at `categories.id`, so ownership must be checked before attaching one."""
    cat = await get_category(session, user_id, category_id)
    if cat is None:
        raise NotOwnedError("category")
    return cat
