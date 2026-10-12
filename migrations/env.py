"""Alembic env (async). DB URL: `config.attributes["url"]` (tests) or duitku.config DATABASE_URL."""

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from duitku.config import get_settings
from duitku.db.models import Base

config = context.config
target_metadata = Base.metadata

# Owned by AsyncPostgresSaver.setup(); never autogenerate/drop them (data-model.md).
CHECKPOINT_TABLES = {
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
}


def include_object(obj: object, name: str | None, type_: str, reflected: bool, compare_to: object):
    return not (type_ == "table" and name in CHECKPOINT_TABLES)


def _url() -> str:
    return config.attributes.get("url") or get_settings().database_url.get_secret_value()


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata, include_object=include_object
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    engine = create_async_engine(_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(_run_async())
