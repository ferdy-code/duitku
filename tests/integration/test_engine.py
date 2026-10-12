"""Async engine + ORM models against the migrated test DB (server defaults, vector column)."""

import asyncio

from alembic import command
from alembic.config import Config
from sqlalchemy import select

from duitku.db.engine import make_engine, make_sessionmaker
from duitku.db.models import EMBEDDING_DIM, Memory, User


async def test_orm_roundtrip(alembic_cfg: Config, fresh_db: str) -> None:
    # Sync Alembic cannot run inside this loop (asyncio.run) -> hop to a thread.
    await asyncio.to_thread(command.upgrade, alembic_cfg, "head")
    engine = make_engine(fresh_db)
    try:
        async with make_sessionmaker(engine)() as s:
            user = User(telegram_user_id=42)
            s.add(user)
            await s.flush()
            s.add(
                Memory(user_id=user.id, kind="fact", content="x", embedding=[0.1] * EMBEDDING_DIM)
            )
            await s.commit()
        async with make_sessionmaker(engine)() as s:
            u = (await s.execute(select(User))).scalar_one()
            m = (await s.execute(select(Memory))).scalar_one()
            assert u.timezone == "Asia/Jakarta" and u.created_at is not None
            assert m.recall_count == 0 and len(m.embedding) == EMBEDDING_DIM
    finally:
        await engine.dispose()
