"""Integration fixtures: a SEPARATE `<db>_test` database, recreated empty for every test.

The dev database is never touched. If Postgres is unreachable, tests skip with a clear reason.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic.config import Config
from psycopg import sql
from sqlalchemy.engine import make_url

# Config requires LLM vars at load time; DB tests do not use them. Real values win if set.
os.environ.setdefault("LLM_BASE_URL", "http://unused.invalid/v1")
os.environ.setdefault("LLM_API_KEY", "unused")

from duitku.config import get_settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """SQLAlchemy URL (with password) of `<db>_test`, derived from DATABASE_URL."""
    dev = make_url(get_settings().database_url.get_secret_value())
    assert dev.database, "DATABASE_URL has no database name"
    return dev.set(database=f"{dev.database}_test").render_as_string(hide_password=False)


@pytest.fixture
def fresh_db(test_db_url: str) -> Iterator[str]:
    """Drop + recreate the empty test DB; yields its SQLAlchemy URL."""
    url = make_url(test_db_url)
    name = url.database
    assert name and name.endswith("_test"), "refusing to drop a non-test database"
    admin = url.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    try:
        conn = psycopg.connect(admin, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres unreachable at DATABASE_URL host/port: {type(e).__name__}")
    with conn:
        ident = sql.Identifier(name)
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(ident))
        conn.execute(sql.SQL("CREATE DATABASE {}").format(ident))
    yield test_db_url


@pytest.fixture
def alembic_cfg(fresh_db: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["url"] = fresh_db  # env.py prefers this over DATABASE_URL
    return cfg
