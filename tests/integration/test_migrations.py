"""T-003 AC: upgrade head from empty, downgrade base, vector enabled, models == migration.

Sync tests on purpose: Alembic's async env.py calls asyncio.run(), illegal inside a running loop.
"""

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Connection, create_engine, inspect, text
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


def test_downgrade_to_0001_and_back(alembic_cfg: Config, fresh_db: str) -> None:
    command.upgrade(alembic_cfg, "head")
    command.downgrade(alembic_cfg, "0001")
    engine = create_engine(fresh_db)
    with engine.connect() as c:
        fks = {
            (t, tuple(fk["constrained_columns"]), fk["name"])
            for t in ("expenses", "budgets")
            for fk in inspect(c).get_foreign_keys(t)
            if fk["referred_table"] in ("categories", "receipts")
        }
        uniques = {
            u["name"]
            for t in ("categories", "receipts")
            for u in inspect(c).get_unique_constraints(t)
        }
    engine.dispose()
    assert fks == {
        ("expenses", ("category_id",), "expenses_category_id_fkey"),
        ("expenses", ("receipt_id",), "expenses_receipt_id_fkey"),
        ("budgets", ("category_id",), "budgets_category_id_fkey"),
    }
    assert not uniques & {"categories_user_id_id_key", "receipts_user_id_id_key"}
    command.upgrade(alembic_cfg, "head")


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


def _two_users(c: Connection) -> dict[str, object]:
    """User A and B, each with one category and one receipt. Returns their ids."""
    ids: dict[str, object] = {}
    for n, tg in (("a", 1), ("b", 2)):
        u = c.execute(
            text("INSERT INTO users (telegram_user_id) VALUES (:t) RETURNING id"), {"t": tg}
        ).scalar()
        ids[f"u{n}"] = u
        ids[f"c{n}"] = c.execute(
            text("INSERT INTO categories (user_id, slug, name) VALUES (:u,'x','X') RETURNING id"),
            {"u": u},
        ).scalar()
        ids[f"r{n}"] = c.execute(
            text("INSERT INTO receipts (user_id, file_path) VALUES (:u,'f') RETURNING id"),
            {"u": u},
        ).scalar()
    return ids


_INS_EXP = text(
    "INSERT INTO expenses (user_id, amount_idr, category_id, receipt_id, occurred_at, source)"
    " VALUES (:u, 1000, :c, :r, now(), 'text') RETURNING id"
)
_INS_BUD = text(
    "INSERT INTO budgets (user_id, category_id, monthly_limit_idr) VALUES (:u, :c, 1000)"
)


def test_db_rejects_cross_user_references(alembic_cfg: Config, fresh_db: str) -> None:
    """Raw SQL, no repos: the composite FKs alone must stop user A pointing at B's rows."""
    command.upgrade(alembic_cfg, "head")
    engine = create_engine(fresh_db)
    with engine.begin() as c:
        i = _two_users(c)
        # Own refs and NULL receipt (MATCH SIMPLE) are fine.
        own = {"u": i["ua"], "c": i["ca"], "r": i["ra"]}
        eid = c.execute(_INS_EXP, own).scalar()
        c.execute(_INS_EXP, {**own, "r": None})
        c.execute(_INS_BUD, {"u": i["ua"], "c": i["ca"]})

    def fails(sql: str, params: dict[str, object]) -> None:
        with pytest.raises(IntegrityError, match="foreign key"), engine.begin() as c:
            c.execute(text(sql), params)

    # INSERT with another user's category / receipt.
    fails(_INS_EXP.text, {"u": i["ua"], "c": i["cb"], "r": None})
    fails(_INS_EXP.text, {"u": i["ua"], "c": i["ca"], "r": i["rb"]})
    fails(_INS_BUD.text, {"u": i["ua"], "c": i["cb"]})
    # UPDATE re-pointing an existing row at another user's category / receipt.
    fails("UPDATE expenses SET category_id = :c WHERE id = :e", {"c": i["cb"], "e": eid})
    fails("UPDATE expenses SET receipt_id = :r WHERE id = :e", {"r": i["rb"], "e": eid})
    fails("UPDATE budgets SET category_id = :c WHERE user_id = :u", {"c": i["cb"], "u": i["ua"]})
    # UPDATE moving the row to another user while keeping its refs.
    fails("UPDATE expenses SET user_id = :u WHERE id = :e", {"u": i["ub"], "e": eid})
    engine.dispose()


def test_user_delete_still_cascades(alembic_cfg: Config, fresh_db: str) -> None:
    command.upgrade(alembic_cfg, "head")
    engine = create_engine(fresh_db)
    with engine.begin() as c:
        i = _two_users(c)
        c.execute(_INS_EXP, {"u": i["ua"], "c": i["ca"], "r": i["ra"]})
        c.execute(_INS_BUD, {"u": i["ua"], "c": i["ca"]})
        c.execute(text("DELETE FROM users WHERE id = :u"), {"u": i["ua"]})
        n = c.execute(text("SELECT count(*) FROM expenses")).scalar()
    engine.dispose()
    assert n == 0
