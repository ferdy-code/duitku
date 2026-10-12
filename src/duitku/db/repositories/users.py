"""Users. The one aggregate keyed by its own id: `user_id` here IS `users.id`."""

import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from duitku.db.models import User
from duitku.db.repositories.categories import seed_defaults


async def get_user(session: AsyncSession, user_id: uuid.UUID) -> User | None:
    return (await session.execute(select(User).where(User.id == user_id))).scalar_one_or_none()


async def get_by_telegram_id(session: AsyncSession, telegram_user_id: int) -> User | None:
    q = select(User).where(User.telegram_user_id == telegram_user_id)
    return (await session.execute(q)).scalar_one_or_none()


async def get_or_create_user(
    session: AsyncSession, telegram_user_id: int, display_name: str | None = None
) -> tuple[User, bool]:
    """Return (user, created). Defaults are seeded in the caller's transaction.

    ON CONFLICT DO NOTHING makes concurrent first messages safe: exactly one insert wins
    (and seeds); the loser just reads the winner's row.
    """
    stmt = (
        pg_insert(User)
        .values(telegram_user_id=telegram_user_id, display_name=display_name)
        .on_conflict_do_nothing(index_elements=["telegram_user_id"])
        .returning(User)
    )
    user = (await session.execute(stmt)).scalar_one_or_none()
    if user is not None:
        await seed_defaults(session, user.id)
        return user, True
    existing = await get_by_telegram_id(session, telegram_user_id)
    assert existing is not None  # conflict implies the row exists (we never delete users here)
    return existing, False


async def update_user(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    timezone: str | None = None,
    display_name: str | None = None,
) -> User | None:
    values: dict[str, str] = {}
    if timezone is not None:
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(f"unknown IANA timezone: {timezone}") from e
        values["timezone"] = timezone
    if display_name is not None:
        values["display_name"] = display_name
    if not values:
        return await get_user(session, user_id)
    q = update(User).where(User.id == user_id).values(**values).returning(User)
    return (await session.execute(q)).scalar_one_or_none()
