"""T-206: update_expense through the real StructuredTool (partial update, isolation, R-05)."""

import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.db.repositories import audit, categories, expenses, users
from duitku.tools import clock
from duitku.tools.registry import build_tools

SM = async_sessionmaker[AsyncSession]
Run = Callable[..., Awaitable[dict[str, Any]]]
JKT = "Asia/Jakarta"
T0 = datetime(2026, 10, 12, 3, 0, tzinfo=UTC)  # 10:00 Jakarta, 2026-10-12
WIDE = {"start_date": "2026-10-01", "end_date": "2026-10-31"}


@pytest.fixture
def tick(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Move the fake clock to T0 + n minutes."""
    monkeypatch.setattr(clock, "now", lambda: T0)

    def _set(minutes: int) -> None:
        monkeypatch.setattr(clock, "now", lambda: T0 + timedelta(minutes=minutes))

    return _set


@pytest.fixture
def run(sm: SM) -> Run:
    tools = {t.name: t for t in build_tools(sm)}

    async def _run(name: str, args: dict[str, Any], uid: uuid.UUID, cid: str = "c") -> Any:
        config: RunnableConfig = {"configurable": {"user_id": str(uid), "timezone": JKT}}
        call = {"type": "tool_call", "name": name, "args": args, "id": cid}
        return json.loads((await tools[name].ainvoke(call, config)).content)

    return _run


async def make_user(sm: SM, tg: int) -> uuid.UUID:
    async with sm() as s:
        user, _ = await users.get_or_create_user(s, tg)
        await s.commit()
        return user.id


async def add(run: Run, uid: uuid.UUID, amount: str, cid: str, **kw: Any) -> str:
    res = await run("add_expense", {"amount": amount, "category": "makan-minum", **kw}, uid, cid)
    assert res["ok"], res
    return res["data"]["expense_id"]


def err_code(res: dict[str, Any]) -> str:
    assert res["ok"] is False and res["error"]["message"] and res["error"]["hint"]
    return res["error"]["code"]


async def stored(sm: SM, uid: uuid.UUID, eid: str) -> Any:
    async with sm() as s:
        return await expenses.get_expense(s, uid, uuid.UUID(eid))


async def test_partial_update_returns_old_and_new(sm: SM, run: Run, tick: Any) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "35rb", "a", merchant="Warteg", note="siang")
    before = await stored(sm, uid, eid)
    res = await run("update_expense", {"expense_id": eid, "amount": "40rb"}, uid)
    assert res["ok"] is True
    d = res["data"]
    assert d["old"]["amount_idr"] == 35_000 and d["new"]["amount_idr"] == 40_000
    assert d["new"]["merchant"] == "Warteg" and d["new"]["note"] == "siang"  # untouched
    assert d["new"]["category"] == "makan-minum"
    after = await stored(sm, uid, eid)
    assert after.amount_idr == 40_000 and after.occurred_at == before.occurred_at
    assert after.updated_at >= before.updated_at


async def test_update_every_field(sm: SM, run: Run, tick: Any) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "35rb", "a", merchant="Warteg", note="siang")
    args = {
        "expense_id": eid,
        "amount": "1,5 juta",
        "category": " Transport ",
        "merchant": "Gojek",
        "note": "",  # clears
        "occurred_on": "2026-10-10",
    }
    d = (await run("update_expense", args, uid))["data"]
    assert d["new"] == {
        "amount_idr": 1_500_000,
        "category": "transport",
        "merchant": "Gojek",
        "note": None,
        "occurred_on": "2026-10-10",
    }
    assert d["old"]["occurred_on"] == "2026-10-12" and d["old"]["category"] == "makan-minum"
    row = await stored(sm, uid, eid)
    # date moved, time of day kept (10:00 Jakarta)
    assert row.occurred_at == datetime(2026, 10, 10, 3, 0, tzinfo=UTC)


async def test_bad_inputs_change_nothing(sm: SM, run: Run, tick: Any) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "35rb", "a")
    cases: list[tuple[dict[str, Any], str | None]] = [
        ({"amount": "5 m"}, "AMBIGUOUS_AMOUNT"),
        ({"amount": "abc"}, "INVALID_AMOUNT"),
        ({"amount": "0"}, None),
        ({"amount": "40rb", "category": "nope"}, "UNKNOWN_CATEGORY"),  # no partial apply
        ({}, "INVALID_ARGUMENT"),
        ({"merchant": "x" * 201}, "INVALID_ARGUMENT"),
    ]
    for extra, code in cases:
        res = await run("update_expense", {"expense_id": eid, **extra}, uid)
        assert res["ok"] is False
        if code:
            assert res["error"]["code"] == code
    res = await run("update_expense", {"expense_id": "not-a-uuid", "amount": "1rb"}, uid)
    assert err_code(res) == "INVALID_ARGUMENT"
    row = await stored(sm, uid, eid)
    assert row.amount_idr == 35_000 and row.category_id is not None
    async with sm() as s:
        cat = await categories.get_by_slug(s, uid, "makan-minum")
    assert cat and row.category_id == cat.id


async def test_not_found_cases_are_indistinguishable(sm: SM, run: Run, tick: Any) -> None:
    a, b = await make_user(sm, 1), await make_user(sm, 2)
    foreign = await add(run, b, "50rb", "b")
    deleted = await add(run, a, "20rb", "a")
    async with sm() as s:
        assert await expenses.soft_delete_expense(s, a, uuid.UUID(deleted))
        await s.commit()
    results = [
        await run("update_expense", {"expense_id": eid, "amount": "99rb"}, a)
        for eid in (foreign, deleted, str(uuid.uuid4()))
    ]
    assert [err_code(r) for r in results] == ["NOT_FOUND"] * 3
    assert results[0] == results[1] == results[2]  # nothing reveals that the foreign id exists
    assert (await stored(sm, b, foreign)).amount_idr == 50_000
    async with sm() as s:  # soft-deleted row untouched
        amt = (
            await s.execute(text("select amount_idr from expenses where id=:i"), {"i": deleted})
        ).scalar_one()
    assert amt == 20_000


async def test_idempotent_same_update_twice(sm: SM, run: Run, tick: Any) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "35rb", "a")
    first = await run("update_expense", {"expense_id": eid, "amount": "40rb"}, uid, "u1")
    second = await run("update_expense", {"expense_id": eid, "amount": "40rb"}, uid, "u2")
    assert first["ok"] and second["ok"]
    assert second["data"]["old"]["amount_idr"] == second["data"]["new"]["amount_idr"] == 40_000


async def test_audit_rows_written(sm: SM, run: Run, tick: Any) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "35rb", "a")
    await run("update_expense", {"expense_id": eid, "amount": "40rb"}, uid)
    await run("update_expense", {"expense_id": str(uuid.uuid4()), "amount": "40rb"}, uid)
    async with sm() as s:
        rows = [r for r in await audit.list_tool_calls(s, uid) if r.tool_name == "update_expense"]
    assert sorted((r.status, r.error_code) for r in rows) == [("error", "NOT_FOUND"), ("ok", None)]


async def test_r05_change_the_last_one_to_40rb(sm: SM, run: Run, tick: Any) -> None:
    """The agent flow: 'last one' = newest row of query_expenses (or add_expense's id)."""
    a, b = await make_user(sm, 1), await make_user(sm, 2)
    tick(0)
    ids = [await add(run, a, "10rb", "a1")]
    tick(30)
    ids.append(await add(run, a, "20rb", "a2"))
    tick(60)
    ids.append(await add(run, a, "35rb", "a3"))  # <- the user's last expense
    tick(90)
    other = await add(run, b, "77rb", "b1")  # another user's NEWER expense
    tick(120)
    trashed = await add(run, a, "5rb", "a4")  # newest of all, but then deleted
    async with sm() as s:
        assert await expenses.soft_delete_expense(s, a, uuid.UUID(trashed))
        await s.commit()

    rows = (await run("query_expenses", {**WIDE, "group_by": "none"}, a))["data"]["rows"]
    last = rows[0]["expense_id"]
    assert last == ids[2] and last != other
    res = await run("update_expense", {"expense_id": last, "amount": "40rb"}, a)
    assert res["data"]["old"]["amount_idr"] == 35_000 and res["data"]["new"]["amount_idr"] == 40_000

    amounts = {i: (await stored(sm, a, i)).amount_idr for i in ids}
    assert amounts == {ids[0]: 10_000, ids[1]: 20_000, ids[2]: 40_000}
    assert (await stored(sm, b, other)).amount_idr == 77_000
    total = (await run("query_expenses", {**WIDE}, a))["data"]["total_idr"]
    assert total == 70_000  # 10 + 20 + 40
