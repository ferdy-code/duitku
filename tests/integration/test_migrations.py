"""T-003 AC: upgrade head from empty, downgrade base, vector enabled, models == migration.

Sync tests on purpose: Alembic's async env.py calls asyncio.run(), illegal inside a running loop.
"""

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from duitku.db.models import Base

TABLES = {
    "users",
    "categories",
    "receipts",
    "expenses",
    "budgets",
    "memories",
    "agent_runs",
    "tool_audit_log",
}


def _tables_and_exts(url: str) -> tuple[set[str], set[str]]:
    engine = create_engine(url)
    with engine.connect() as c:
        tables = set(inspect(c).get_table_names())
        exts = {r[0] for r in c.execute(text("SELECT extname FROM pg_extension"))}
    engine.dispose()
    return tables, exts


def test_upgrade_head_from_empty(alembic_cfg: Config, fresh_db: str) -> None:
    command.upgrade(alembic_cfg, "head")
    tables, exts = _tables_and_exts(fresh_db)
    assert tables >= TABLES
    assert not tables & {"checkpoints", "checkpoint_blobs", "checkpoint_writes"}
    assert "vector" in exts


def test_downgrade_base(alembic_cfg: Config, fresh_db: str) -> None:
    command.upgrade(alembic_cfg, "head")
    command.downgrade(alembic_cfg, "base")
    tables, exts = _tables_and_exts(fresh_db)
    assert tables <= {"alembic_version"}
    assert "vector" not in exts
    command.upgrade(alembic_cfg, "head")  # round trip


def _skip_version_table(obj: object, name: str | None, type_: str, *_: object) -> bool:
    return not (type_ == "table" and name == "alembic_version")


def test_models_match_migration(alembic_cfg: Config, fresh_db: str) -> None:
    command.upgrade(alembic_cfg, "head")
    engine = create_engine(fresh_db)
    with engine.connect() as c:
        ctx = MigrationContext.configure(c, opts={"include_object": _skip_version_table})
        assert compare_metadata(ctx, Base.metadata) == []
    engine.dispose()


def test_idempotency_key_unique_per_user_but_null_allowed(
    alembic_cfg: Config, fresh_db: str
) -> None:
    command.upgrade(alembic_cfg, "head")
    engine = create_engine(fresh_db)
    with engine.begin() as c:
        uid = c.execute(
            text("INSERT INTO users (telegram_user_id) VALUES (1) RETURNING id")
        ).scalar()
        cid = c.execute(
            text(
                "INSERT INTO categories (user_id, slug, name)"
                " VALUES (:u,'lainnya','Lainnya') RETURNING id"
            ),
            {"u": uid},
        ).scalar()
    ins = text(
        "INSERT INTO expenses"
        " (user_id, amount_idr, category_id, occurred_at, source, idempotency_key)"
        " VALUES (:u, 35000, :c, now(), 'text', :k)"
    )
    with engine.begin() as c:
        c.execute(ins, {"u": uid, "c": cid, "k": None})
        c.execute(ins, {"u": uid, "c": cid, "k": None})  # NULL keys never collide
        c.execute(ins, {"u": uid, "c": cid, "k": "call_1"})
    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(ins, {"u": uid, "c": cid, "k": "call_1"})
    bad = text(
        "INSERT INTO expenses (user_id, amount_idr, category_id, occurred_at, source)"
        " VALUES (:u, 0, :c, now(), 'text')"
    )
    with pytest.raises(IntegrityError), engine.begin() as c:  # CHECK amount_idr > 0
        c.execute(bad, {"u": uid, "c": cid})
    engine.dispose()
