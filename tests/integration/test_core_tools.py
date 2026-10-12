"""T-104: add_expense / query_expenses / resolve_period through the real StructuredTools.

Calls go through `ainvoke` with a tool-call dict, so tool_call_id injection and
handle_validation_error are exercised. group_by results are checked against a plain-Python
reference computed from the seeded rows (never from the tool's own SQL).
"""

import json
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from langchain_core.runnables import RunnableConfig
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.db.models import Expense
from duitku.db.repositories import audit, categories, expenses, receipts, users
from duitku.tools import clock
from duitku.tools.registry import build_tools

SM = async_sessionmaker[AsyncSession]
Run = Callable[..., Awaitable[dict[str, Any]]]
JKT = "Asia/Jakarta"
NYC = "America/New_York"
# 2026-10-12 23:30 UTC == 2026-10-13 06:30 Jakarta == 2026-10-12 19:30 New York
NOW = datetime(2026, 10, 12, 23, 30, tzinfo=UTC)
SLUGS = ["makan-minum", "transport", "belanja", "hiburan"]
ADD, QUERY, PERIOD = "add_expense", "query_expenses", "resolve_period"
WIDE = {"start_date": "2026-08-01", "end_date": "2026-12-31"}


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clock, "now", lambda: NOW)


@pytest.fixture
def run(sm: SM) -> Run:
    """run(tool_name, args, uid, tz=JKT, cid="c1") -> parsed envelope (real ainvoke)."""
    tools = {t.name: t for t in build_tools(sm)}

    async def _run(
        name: str, args: dict[str, Any], uid: uuid.UUID, tz: str = JKT, cid: str = "c1"
    ) -> dict[str, Any]:
        config: RunnableConfig = {"configurable": {"user_id": str(uid), "timezone": tz}}
        call = {"type": "tool_call", "name": name, "args": args, "id": cid}
        msg = await tools[name].ainvoke(call, config)
        assert msg.tool_call_id == cid
        return json.loads(msg.content)

    return _run


async def make_user(sm: SM, tg: int) -> uuid.UUID:
    async with sm() as s:
        user, _ = await users.get_or_create_user(s, tg)
        await s.commit()
        return user.id


async def count_rows(sm: SM, uid: uuid.UUID) -> int:
    async with sm() as s:
        q = select(func.count()).select_from(Expense).where(Expense.user_id == uid)
        return (await s.execute(q)).scalar_one()


def err(res: dict[str, Any]) -> dict[str, Any]:
    assert res["ok"] is False
    assert res["error"]["message"] and res["error"]["hint"]
    return res["error"]


# --- add_expense ------------------------------------------------------------------------------


async def test_add_expense_basic_and_defaults(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    args = {"amount": "35rb", "category": "makan-minum", "merchant": "Warteg"}
    d = (await run(ADD, args, uid))["data"]
    assert (d["amount_idr"], d["category"], d["merchant"]) == (35_000, "makan-minum", "Warteg")
    assert d["occurred_on"] == "2026-10-13"  # today in Jakarta, not in UTC
    assert d["duplicate"] is False and d["source"] == "text"
    uuid.UUID(d["expense_id"])

    other = await make_user(sm, 2)  # same instant, New York user: still the 12th there
    res = await run(ADD, {"amount": "10000", "category": "transport"}, other, NYC)
    assert res["data"]["occurred_on"] == "2026-10-12"

    args = {"amount": "1,5 juta", "category": "belanja", "occurred_on": "2026-09-30"}
    d = (await run(ADD, args, uid, cid="c2"))["data"]
    assert (d["occurred_on"], d["amount_idr"]) == ("2026-09-30", 1_500_000)

    # int amounts (models sometimes send numbers) and sloppy slugs are accepted
    res = await run(ADD, {"amount": 35000, "category": " Transport "}, uid, cid="c3")
    assert res["data"]["category"] == "transport" and res["data"]["amount_idr"] == 35_000


async def test_same_tool_call_id_twice_is_one_row(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    first = await run(ADD, {"amount": "35rb", "category": "makan-minum"}, uid, cid="same")
    second = await run(ADD, {"amount": "99rb", "category": "transport"}, uid, cid="same")
    assert first["data"]["duplicate"] is False and second["data"]["duplicate"] is True
    assert second["data"]["expense_id"] == first["data"]["expense_id"]
    # the replay reports the STORED row (first write wins), not its own arguments
    assert second["data"]["amount_idr"] == 35_000
    assert second["data"]["category"] == "makan-minum"
    assert await count_rows(sm, uid) == 1
    await run(ADD, {"amount": "35rb", "category": "makan-minum"}, uid, cid="different")
    assert await count_rows(sm, uid) == 2


async def test_same_tool_call_id_for_two_users_is_two_rows(sm: SM, run: Run) -> None:
    a, b = await make_user(sm, 1), await make_user(sm, 2)
    for uid in (a, b):
        res = await run(ADD, {"amount": "5rb", "category": "lainnya"}, uid)
        assert res["data"]["duplicate"] is False


@pytest.mark.parametrize(
    ("amount", "code"),
    [
        ("1.5", "AMBIGUOUS_AMOUNT"),
        ("5m", "AMBIGUOUS_AMOUNT"),
        ("0", "INVALID_AMOUNT"),
        ("-5rb", "INVALID_AMOUNT"),
        ("abc", "INVALID_AMOUNT"),
    ],
)
async def test_bad_amount_is_envelope_and_writes_nothing(
    sm: SM, run: Run, amount: str, code: str
) -> None:
    uid = await make_user(sm, 1)
    res = await run(ADD, {"amount": amount, "category": "makan-minum"}, uid)
    assert err(res)["code"] == code
    assert await count_rows(sm, uid) == 0


@pytest.mark.parametrize(
    "extra",
    [
        {"amount": "5rb"},  # category missing
        {"category": "transport"},  # amount missing
        {"amount": "5rb", "category": "transport", "source": "voice"},
        {"amount": "5rb", "category": "transport", "occurred_on": "kemarin"},
        {"amount": "5rb", "category": "transport", "items": "nope"},
        {"amount": "5rb", "category": "transport", "merchant": "x" * 201},
    ],
)
async def test_add_expense_invalid_arguments(sm: SM, run: Run, extra: dict[str, Any]) -> None:
    uid = await make_user(sm, 1)
    assert err(await run(ADD, extra, uid))["code"] == "INVALID_ARGUMENT"
    assert await count_rows(sm, uid) == 0


async def test_unknown_category_lists_valid_slugs(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    e = err(await run(ADD, {"amount": "5rb", "category": "nongkrong"}, uid))
    assert e["code"] == "UNKNOWN_CATEGORY" and "makan-minum" in e["hint"]
    assert await count_rows(sm, uid) == 0


async def test_user_id_from_llm_args_is_ignored(sm: SM, run: Run) -> None:
    a, b = await make_user(sm, 1), await make_user(sm, 2)
    args = {"amount": "5rb", "category": "lainnya", "user_id": str(b)}
    assert (await run(ADD, args, a))["ok"] is True
    assert await count_rows(sm, a) == 1 and await count_rows(sm, b) == 0


async def test_receipt_ownership(sm: SM, run: Run) -> None:
    a, b = await make_user(sm, 1), await make_user(sm, 2)
    async with sm() as s:
        theirs = await receipts.create_receipt(s, b, "data/receipts/x.jpg")
        mine = await receipts.create_receipt(s, a, "data/receipts/y.jpg")
        await s.commit()
    base = {"amount": "50rb", "category": "belanja", "source": "receipt"}
    res = await run(ADD, {**base, "receipt_id": str(mine.id)}, a, cid="r1")
    assert res["ok"] is True and res["data"]["source"] == "receipt"
    for rid, cid in ((str(theirs.id), "r2"), (str(uuid.uuid4()), "r3")):
        res = await run(ADD, {**base, "receipt_id": rid}, a, cid=cid)
        assert err(res)["code"] == "NOT_FOUND"
    res = await run(ADD, {**base, "receipt_id": "nope"}, a, cid="r4")
    assert err(res)["code"] == "INVALID_ARGUMENT"
    assert await count_rows(sm, a) == 1


async def test_calls_are_audited(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    await run(ADD, {"amount": "5rb", "category": "lainnya"}, uid, cid="a1")
    await run(ADD, {"amount": "1.5", "category": "lainnya"}, uid, cid="a2")
    async with sm() as s:
        rows = await audit.list_tool_calls(s, uid)
    assert sorted((r.tool_name, r.status, r.error_code or "") for r in rows) == [
        (ADD, "error", "AMBIGUOUS_AMOUNT"),
        (ADD, "ok", ""),
    ]


# --- query_expenses ---------------------------------------------------------------------------

Seed = tuple[int, str, datetime]  # amount, category slug, occurred_at (UTC)


def _gen(seed: int) -> list[Seed]:
    """Deterministic varied rows over ~8 weeks / 3 months, odd hours so tz bucketing matters."""
    out: list[Seed] = []
    day0 = datetime(2026, 8, 24, tzinfo=UTC)
    for _ in range(70):
        seed = (seed * 1103515245 + 12345) % 2**31
        when = day0 + timedelta(days=seed % 55, hours=(seed >> 8) % 24, minutes=(seed >> 16) % 60)
        out.append((1000 * (1 + seed % 97), SLUGS[(seed >> 4) % len(SLUGS)], when))
    # boundary rows: 23:30 vs 00:30 Jakarta around the month turn; Sunday night / Monday morning
    for amt, h in ((111_000, 16), (222_000, 17)):  # 2026-09-30 23:30 / 2026-10-01 00:30 Jakarta
        out.append((amt, "makan-minum", datetime(2026, 9, 30, h, 30, tzinfo=UTC)))
    for amt, h in ((333_000, 16), (444_000, 17)):  # Sun 10-04 23:30 / Mon 10-05 00:30 Jakarta
        out.append((amt, "transport", datetime(2026, 10, 4, h, 30, tzinfo=UTC)))
    return out


async def _seed(sm: SM, uid: uuid.UUID, data: list[Seed]) -> list[uuid.UUID]:
    ids: list[uuid.UUID] = []
    async with sm() as s:
        slug_id = {c.slug: c.id for c in await categories.list_categories(s, uid)}
        for amount, slug, when in data:
            res = await expenses.insert_expense(
                s, uid, amount_idr=amount, category_id=slug_id[slug], occurred_at=when,
                source="text",
            )  # fmt: skip
            ids.append(res.expense.id)
        await s.commit()
    return ids


@pytest.fixture
async def seeded(sm: SM) -> tuple[uuid.UUID, uuid.UUID, list[Seed]]:
    """User A's live rows (returned), plus a soft-deleted A row and a pile of B rows."""
    a, b = await make_user(sm, 1), await make_user(sm, 2)
    live = _gen(1)
    await _seed(sm, a, live)
    (doomed,) = await _seed(sm, a, [(777_777, "belanja", datetime(2026, 9, 10, 5, tzinfo=UTC))])
    async with sm() as s:
        assert await expenses.soft_delete_expense(s, a, doomed)
        await s.commit()
    b_rows = [*_gen(2), (5_000_000, "makan-minum", datetime(2026, 9, 30, 16, 45, tzinfo=UTC))]
    await _seed(sm, b, b_rows)
    return a, b, live


def _local(when: datetime, tz: str) -> date:
    return when.astimezone(ZoneInfo(tz)).date()


def _bucket(d: date, group_by: str) -> str:
    if group_by == "day":
        return d.isoformat()
    if group_by == "week":
        return (d - timedelta(days=d.weekday())).isoformat()
    return d.strftime("%Y-%m")


def _reference(data: list[Seed], tz: str, start: date, end: date, cat: str | None) -> list[Seed]:
    return [r for r in data if start <= _local(r[2], tz) <= end and cat in (None, r[1])]


RANGES = [
    (date(2026, 8, 1), date(2026, 12, 31)),  # everything
    (date(2026, 9, 1), date(2026, 9, 30)),  # month, ends on the boundary rows
    (date(2026, 10, 1), date(2026, 10, 1)),  # single day
    (date(2026, 9, 28), date(2026, 10, 6)),  # crosses month + the Sunday/Monday boundary
    (date(2027, 1, 1), date(2027, 1, 31)),  # empty
]


@pytest.mark.parametrize("tz", [JKT, NYC])
@pytest.mark.parametrize("category", [None, "makan-minum", "transport"])
@pytest.mark.parametrize("group_by", ["none", "category", "day", "week", "month"])
@pytest.mark.parametrize(("start", "end"), RANGES)
async def test_query_matches_reference(
    run: Run,
    seeded: tuple[uuid.UUID, uuid.UUID, list[Seed]],
    tz: str,
    category: str | None,
    group_by: str,
    start: date,
    end: date,
) -> None:
    a, _b, live = seeded
    args: dict[str, Any] = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "group_by": group_by,
        "limit": 50,
    }
    if category:
        args["category"] = category
    res = await run(QUERY, args, a, tz)
    assert res["ok"] is True
    d = res["data"]
    ref = _reference(live, tz, start, end, category)
    assert d["total_idr"] == sum(r[0] for r in ref)  # B's rows and the deleted row never count
    assert d["count"] == len(ref)

    if group_by == "none":
        assert d["groups"] == []
        want = sorted(ref, key=lambda r: r[2], reverse=True)[:50]
        assert [r["amount_idr"] for r in d["rows"]] == [r[0] for r in want]
        assert [r["occurred_on"] for r in d["rows"]] == [_local(r[2], tz).isoformat() for r in want]
        assert [r["category"] for r in d["rows"]] == [r[1] for r in want]
        assert d["rows_truncated"] is (len(ref) > 50)
        return

    assert d["rows"] == []
    want_groups: dict[str, list[int]] = defaultdict(list)
    for amount, slug, when in ref:
        key = slug if group_by == "category" else _bucket(_local(when, tz), group_by)
        want_groups[key].append(amount)
    keys = [g["key"] for g in d["groups"]]
    assert len(keys) == len(set(keys))
    got = {g["key"]: (g["total_idr"], g["count"]) for g in d["groups"]}
    assert got == {k: (sum(v), len(v)) for k, v in want_groups.items()}
    if group_by == "category":
        totals = [g["total_idr"] for g in d["groups"]]
        assert totals == sorted(totals, reverse=True)
        assert all(g["name"] for g in d["groups"])
    else:
        assert keys == sorted(keys)
    assert sum(g["total_idr"] for g in d["groups"]) == d["total_idr"]


async def test_limit_truncates_rows_not_totals(
    run: Run, seeded: tuple[uuid.UUID, uuid.UUID, list[Seed]]
) -> None:
    a, _b, live = seeded
    d = (await run(QUERY, {**WIDE, "limit": 5}, a))["data"]
    assert len(d["rows"]) == 5 and d["rows_truncated"] is True
    assert d["count"] == len(live) and d["total_idr"] == sum(r[0] for r in live)
    newest = sorted(live, key=lambda r: r[2], reverse=True)[:5]
    assert [r["amount_idr"] for r in d["rows"]] == [r[0] for r in newest]


async def test_tz_day_boundary_and_other_user_isolation(
    run: Run, seeded: tuple[uuid.UUID, uuid.UUID, list[Seed]]
) -> None:
    a, b, _live = seeded
    # 23:30 Jakarta on Sep 30 vs 00:30 Jakarta on Oct 1 land on different local days.
    args = {"start_date": "2026-09-30", "end_date": "2026-09-30", "category": "makan-minum"}
    amounts = [
        r["amount_idr"] for r in (await run(QUERY, {**args, "limit": 50}, a))["data"]["rows"]
    ]
    assert 111_000 in amounts and 222_000 not in amounts
    assert 5_000_000 not in amounts  # B's row
    mine = await run(QUERY, {**WIDE, "group_by": "category"}, b)
    assert mine["data"]["total_idr"] >= 5_000_000  # sanity: B sees B's own data


@pytest.mark.parametrize(
    "args",
    [
        {"start_date": "2026-10-01"},
        {"start_date": "2026-10-01", "end_date": "2026-10-31", "group_by": "year"},
        {"start_date": "2026-10-01", "end_date": "2026-10-31", "limit": 51},
        {"start_date": "2026-10-01", "end_date": "2026-10-31", "limit": 0},
        {"start_date": "bulan ini", "end_date": "2026-10-31"},
        {"start_date": "2026-10-31", "end_date": "2026-10-01"},  # reversed
        {"start_date": "2026-10-01", "end_date": "2026-10-31", "group_by": "day; DROP TABLE x"},
    ],
)
async def test_query_invalid_arguments(sm: SM, run: Run, args: dict[str, Any]) -> None:
    uid = await make_user(sm, 1)
    assert err(await run(QUERY, args, uid))["code"] == "INVALID_ARGUMENT"


async def test_query_unknown_category(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    res = await run(QUERY, {**WIDE, "category": "x"}, uid)
    assert err(res)["code"] == "UNKNOWN_CATEGORY"


async def test_query_empty_user(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    d = (await run(QUERY, {**WIDE, "group_by": "month"}, uid))["data"]
    assert (d["total_idr"], d["count"], d["groups"], d["rows"]) == (0, 0, [], [])


async def test_add_then_query_roundtrip(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    await run(ADD, {"amount": "35rb", "category": "makan-minum"}, uid, cid="x1")
    await run(ADD, {"amount": "20rb", "category": "transport"}, uid, cid="x2")
    p = (await run(PERIOD, {"period": "today"}, uid))["data"]
    d = (await run(QUERY, {"start_date": p["start_date"], "end_date": p["end_date"]}, uid))["data"]
    assert d["total_idr"] == 55_000 and d["count"] == 2


# --- resolve_period ---------------------------------------------------------------------------


async def test_resolve_period_tool(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    res = await run(PERIOD, {"period": "this_week"}, uid)
    # NOW is Tue 2026-10-13 in Jakarta -> week starts Mon 10-12
    want = {"start_date": "2026-10-12", "end_date": "2026-10-13", "label": "Minggu ini"}
    assert res == {"ok": True, "data": want}
    ny = await run(PERIOD, {"period": "today"}, uid, NYC)
    assert ny["data"]["start_date"] == "2026-10-12"
    assert err(await run(PERIOD, {"period": "bulan depan"}, uid))["code"] == "INVALID_ARGUMENT"
