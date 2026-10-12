"""Async SQLAlchemy engine + session factory. URL comes from config only."""

from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from duitku.config import get_settings


def make_engine(url: str) -> AsyncEngine:
    # pool_pre_ping: a Postgres restart must not poison pooled connections.
    return create_async_engine(url, pool_pre_ping=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@lru_cache
def get_engine() -> AsyncEngine:
    return make_engine(get_settings().database_url.get_secret_value())


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return make_sessionmaker(get_engine())
