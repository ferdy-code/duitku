"""Postgres checkpointer: `AsyncPostgresSaver` over a small psycopg pool (docs/spec/agent.md).

The checkpoint tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`,
`checkpoint_migrations`) are created by `saver.setup()`, not by Alembic. They live in the same
database as the app tables; `migrations/env.py` ignores them, so `alembic upgrade/downgrade`
and autogenerate are unaffected.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from duitku.config import get_settings


def tg_thread_id(chat_id: int) -> str:
    return f"tg:{chat_id}"


def cli_thread_id(name: str) -> str:
    return f"cli:{name}"


@asynccontextmanager
async def open_checkpointer(url: str | None = None) -> AsyncGenerator[AsyncPostgresSaver]:
    """Open pool + saver, run `setup()` (idempotent), close the pool on exit.

    `url` (libpq) defaults to CHECKPOINT_DATABASE_URL; tests pass the test DB. Never logged.
    """
    conninfo = url or get_settings().checkpoint_database_url.get_secret_value()
    pool = AsyncConnectionPool(
        conninfo,
        min_size=1,
        max_size=5,
        timeout=10,
        # Same kwargs as AsyncPostgresSaver.from_conn_string: the saver manages its own
        # transactions (autocommit), reads rows by column name, and uses no server-side
        # prepared statements.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        check=AsyncConnectionPool.check_connection,  # a Postgres restart must not poison the pool
        open=False,
    )
    await pool.open(wait=True, timeout=10)
    try:
        saver = AsyncPostgresSaver(pool)  # pyright: ignore[reportArgumentType]
        await saver.setup()
        yield saver
    finally:
        await pool.close()
