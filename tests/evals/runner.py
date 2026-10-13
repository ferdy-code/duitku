"""Run one eval case: fresh `<db>_eval_test` DB, seeded data, the REAL graph and LLM from settings.

The dev database is never touched: every DB handle here points at the eval DB (sessionmaker,
checkpointer, tools, slug loader are all injected).
"""

import asyncio
import logging
import time
import uuid
from datetime import UTC, date, datetime
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg
from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.evals.cases import Case
from tests.evals.checks import DbRow, Observed, check_case
from tests.evals.report import CaseResult
from tests.evals.seeds import sample_month

from duitku.agent.checkpointer import open_checkpointer
from duitku.agent.graph import build_graph
from duitku.agent.nodes import repo_slug_loader
from duitku.config import get_settings
from duitku.db.engine import make_engine, make_sessionmaker
from duitku.db.repositories import categories, expenses
from duitku.interfaces.runner import TurnRunner
from duitku.interfaces.trace import ModelStep
from duitku.tools import clock
from duitku.tools.registry import build_tools

ROOT = Path(__file__).resolve().parents[2]
EVAL_TZ = ZoneInfo("Asia/Jakarta")
# docs/EVALS.md: every case runs at Monday 2026-10-12 10:00 Asia/Jakarta.
FIXED_NOW = datetime(2026, 10, 12, 10, 0, tzinfo=EVAL_TZ)
EVAL_TELEGRAM_ID = 700_000_001
SEED_FIXTURES = {"sample_month": sample_month.EXPENSES}


def eval_db_url(dev_url: str) -> str:
    """SQLAlchemy URL of `<dev db>_eval_test`; the `_test` suffix is the drop-safety marker."""
    dev = make_url(dev_url)
    if not dev.database:
        raise ValueError("DATABASE_URL has no database name")
    return dev.set(database=f"{dev.database}_eval_test").render_as_string(hide_password=False)


def reset_eval_db(url: str) -> None:
    """Drop + recreate the empty eval DB, then alembic head (sync: Alembic runs its own loop)."""
    u = make_url(url)
    name = u.database
    if not name or not name.endswith("_test"):
        raise ValueError("refusing to drop a database whose name does not end in _test")
    admin = u.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    with psycopg.connect(admin, autocommit=True, connect_timeout=5) as conn:
        ident = sql.Identifier(name)
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(ident))
        conn.execute(sql.SQL("CREATE DATABASE {}").format(ident))
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["url"] = url
    command.upgrade(cfg, "head")


async def _seed(sm: async_sessionmaker[AsyncSession], user_id: uuid.UUID, case: Case) -> None:
    async with sm() as s:
        for slug in case.seed.remove_categories:
            await s.execute(
                text("DELETE FROM categories WHERE user_id = :u AND slug = :s"),
                {"u": user_id, "s": slug},
            )
        if case.seed.fixture:
            for e in SEED_FIXTURES[case.seed.fixture]:
                cat = await categories.get_by_slug(s, user_id, e.category)
                assert cat is not None, f"seed category {e.category} missing"
                await expenses.insert_expense(
                    s,
                    user_id,
                    amount_idr=e.amount_idr,
                    category_id=cat.id,
                    occurred_at=datetime.combine(e.day, dtime(12, 0), EVAL_TZ),
                    source="text",
                    merchant=e.merchant,
                )
        await s.commit()


async def _read_rows(sm: async_sessionmaker[AsyncSession], user_id: uuid.UUID) -> list[DbRow]:
    q = text(
        "SELECT e.amount_idr, c.slug, e.merchant, e.occurred_at FROM expenses e "
        "JOIN categories c ON c.id = e.category_id AND c.user_id = e.user_id "
        "WHERE e.user_id = :u AND e.deleted_at IS NULL ORDER BY e.created_at, e.id"
    )
    async with sm() as s:
        rows = (await s.execute(q, {"u": user_id})).all()
    return [DbRow(r[0], r[1], r[2], r[3].astimezone(EVAL_TZ).date()) for r in rows]


async def _range_total(
    sm: async_sessionmaker[AsyncSession], user_id: uuid.UUID, start: date, end: date
) -> int:
    q = text(
        "SELECT COALESCE(SUM(amount_idr), 0) FROM expenses "
        "WHERE user_id = :u AND deleted_at IS NULL "
        "AND (occurred_at AT TIME ZONE 'Asia/Jakarta')::date BETWEEN :a AND :b"
    )
    async with sm() as s:
        return int((await s.execute(q, {"u": user_id, "a": start, "b": end})).scalar_one())


class _Collect(logging.Handler):
    """Keeps runner log lines (exception TYPE only, never content) to explain a failed turn."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


async def run_case(case: Case) -> CaseResult:
    dev_url = get_settings().database_url.get_secret_value()
    url = eval_db_url(dev_url)
    await asyncio.to_thread(reset_eval_db, url)
    engine = make_engine(url)
    sm = make_sessionmaker(engine)
    pg_url = make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)
    collect = _Collect()
    runner_log = logging.getLogger("duitku.interfaces.runner")
    runner_log.addHandler(collect)
    original_now = clock.now
    clock.now = lambda: FIXED_NOW.astimezone(UTC)  # tools read the clock via this module attribute
    try:
        async with open_checkpointer(pg_url) as saver:
            graph = build_graph(
                tools=build_tools(sm),
                checkpointer=saver,
                category_slugs=repo_slug_loader(sm),
                now=lambda: FIXED_NOW,
                sessionmaker=sm,  # auto-recall must read the eval DB, not the dev DB
            )
            runner = TurnRunner(graph, sm)
            user = await runner.resolve_user(EVAL_TELEGRAM_ID, f"eval-{case.id}")
            await _seed(sm, user.user_id, case)
            steps: list[ModelStep] = []
            replies: list[str] = []
            failures: list[str] = []
            started = time.perf_counter()
            for turn in case.turns:
                res = await runner.run(f"eval:{case.id}", user, turn.user)
                steps += res.trace
                replies.append(res.text)
                if res.failed:
                    failures.append(f"turn failed: {'; '.join(collect.lines) or 'unknown'}")
            latency = time.perf_counter() - started
        total = None
        if case.expect.reply_total_of_range:
            total = await _range_total(sm, user.user_id, *case.expect.reply_total_of_range)
        obs = Observed(steps, replies, await _read_rows(sm, user.user_id), total)
    finally:
        clock.now = original_now
        runner_log.removeHandler(collect)
        await engine.dispose()
    failures += check_case(case, obs)
    return CaseResult(
        id=case.id,
        tags=case.tags,
        passed=not failures,
        failures=failures,
        steps=len(steps),
        input_tokens=sum(s.input_tokens or 0 for s in steps),
        output_tokens=sum(s.output_tokens or 0 for s in steps),
        latency_s=round(latency, 2),
        calls=[f"{c.name}({c.args}) -> {c.observation}" for s in steps for c in s.calls],
    )
