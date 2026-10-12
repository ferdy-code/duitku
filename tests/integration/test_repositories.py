"""T-004: CRUD per repository, default category seeding, idempotency, cross-user isolation."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.db.models import Category
from duitku.db.repositories import audit, budgets, categories, expenses, memories, receipts, users
from duitku.domain.categories import DEFAULT_CATEGORIES
from duitku.domain.errors import NotOwnedError

SM = async_sessionmaker[AsyncSession]
T0 = datetime(2026, 10, 12, 5, 0, tzinfo=UTC)
VEC = [0.1] * 384


async def _user(sm: SM, tg: int) -> uuid.UUID:
    async with sm() as s:
        u, _ = await users.get_or_create_user(s, tg)
        await s.commit()
        return u.id


async def _cat(sm: SM, uid: uuid.UUID, slug: str = "makan-minum") -> uuid.UUID:
    async with sm() as s:
        c = await categories.get_by_slug(s, uid, slug)
        assert c is not None
        return c.id


async def _expense(sm: SM, uid: uuid.UUID, key: str | None = None, amount: int = 35_000):
    async with sm() as s:
        r = await expenses.insert_expense(
            s,
            uid,
            amount_idr=amount,
            category_id=await _cat(sm, uid),
            occurred_at=T0,
            source="text",
            merchant="warteg",
            idempotency_key=key,
        )
        await s.commit()
        return r


# --- users + categories -------------------------------------------------------------------


async def test_get_or_create_user_seeds_default_categories(sm: SM) -> None:
    async with sm() as s:
        u, created = await users.get_or_create_user(s, 111, "Keina")
        await s.commit()
    assert created and u.timezone == "Asia/Jakarta" and u.display_name == "Keina"

    async with sm() as s:
        cats = await categories.list_categories(s, u.id)
    assert {(c.slug, c.name) for c in cats} == set(DEFAULT_CATEGORIES)


async def test_get_or_create_user_is_idempotent_and_does_not_reseed(sm: SM) -> None:
    uid = await _user(sm, 111)
    async with sm() as s:
        u2, created = await users.get_or_create_user(s, 111)
        await s.commit()
        n = await s.scalar(select(func.count()).select_from(Category))
    assert not created and u2.id == uid and n == len(DEFAULT_CATEGORIES)


async def test_user_seed_rolls_back_with_user(sm: SM) -> None:
    async with sm() as s:
        await users.get_or_create_user(s, 222)
        await s.rollback()
    async with sm() as s:
        assert await users.get_by_telegram_id(s, 222) is None
        assert await s.scalar(select(func.count()).select_from(Category)) == 0


async def test_update_user(sm: SM) -> None:
    uid = await _user(sm, 111)
    async with sm() as s:
        u = await users.update_user(s, uid, timezone="Asia/Makassar", display_name="K")
        assert u is not None and u.timezone == "Asia/Makassar" and u.display_name == "K"
        with pytest.raises(ValueError):
            await users.update_user(s, uid, timezone="Not/AZone")
        assert await users.update_user(s, uuid.uuid4(), timezone="UTC") is None


async def test_category_lookup(sm: SM) -> None:
    uid = await _user(sm, 111)
    cid = await _cat(sm, uid, "transport")
    async with sm() as s:
        c = await categories.get_category(s, uid, cid)
        assert c is not None and c.name == "Transport"
        assert await categories.get_by_slug(s, uid, "nope") is None


# --- expenses -------------------------------------------------------------------------------


async def test_expense_crud(sm: SM) -> None:
    uid = await _user(sm, 111)
    r = await _expense(sm, uid)
    assert r.created and r.expense.amount_idr == 35_000 and isinstance(r.expense.amount_idr, int)
    eid = r.expense.id
    transport = await _cat(sm, uid, "transport")

    async with sm() as s:
        got = await expenses.get_expense(s, uid, eid)
        assert got is not None and got.merchant == "warteg" and got.deleted_at is None
        upd = await expenses.update_expense(
            s, uid, eid, {"amount_idr": 40_000, "category_id": transport, "note": "x"}
        )
        assert upd is not None and upd.amount_idr == 40_000 and upd.category_id == transport
        assert upd.updated_at >= upd.created_at
        listed = await expenses.list_expenses(s, uid, category_id=transport)
        assert [e.id for e in listed] == [eid]
        assert await expenses.soft_delete_expense(s, uid, eid) is True
        assert await expenses.soft_delete_expense(s, uid, eid) is False  # already deleted
        assert await expenses.get_expense(s, uid, eid) is None
        assert await expenses.list_expenses(s, uid) == []
        assert await expenses.update_expense(s, uid, eid, {"note": "y"}) is None


async def test_list_expenses_range_and_order(sm: SM) -> None:
    uid = await _user(sm, 111)
    cid = await _cat(sm, uid)
    async with sm() as s:
        for day in (1, 2, 3):
            await expenses.insert_expense(
                s,
                uid,
                amount_idr=1000 * day,
                category_id=cid,
                occurred_at=datetime(2026, 10, day, tzinfo=UTC),
                source="text",
            )
        await s.commit()
        rows = await expenses.list_expenses(
            s, uid, start=datetime(2026, 10, 2, tzinfo=UTC), end=datetime(2026, 10, 3, tzinfo=UTC)
        )
        assert [e.amount_idr for e in rows] == [2000]  # start inclusive, end exclusive
        assert [e.amount_idr for e in await expenses.list_expenses(s, uid)] == [3000, 2000, 1000]


async def test_expense_validation(sm: SM) -> None:
    uid = await _user(sm, 111)
    cid = await _cat(sm, uid)
    async with sm() as s:
        with pytest.raises(ValueError):
            await expenses.insert_expense(
                s, uid, amount_idr=0, category_id=cid, occurred_at=T0, source="text"
            )
        with pytest.raises(ValueError):
            await expenses.update_expense(s, uid, uuid.uuid4(), {"user_id": uuid.uuid4()})


async def test_idempotency_key_returns_existing_row(sm: SM) -> None:
    uid = await _user(sm, 111)
    first = await _expense(sm, uid, key="call_1")
    second = await _expense(sm, uid, key="call_1", amount=99_000)  # replay: first write wins
    assert first.created and not second.created
    assert second.expense.id == first.expense.id and second.expense.amount_idr == 35_000
    await _expense(sm, uid, key="call_2")
    await _expense(sm, uid, key=None)
    await _expense(sm, uid, key=None)  # NULL keys never collide
    async with sm() as s:
        assert len(await expenses.list_expenses(s, uid)) == 4


async def test_same_idempotency_key_for_different_users_is_independent(sm: SM) -> None:
    a, b = await _user(sm, 1), await _user(sm, 2)
    ra, rb = await _expense(sm, a, key="k"), await _expense(sm, b, key="k")
    assert ra.created and rb.created and ra.expense.id != rb.expense.id


# --- receipts / budgets / memories / audit CRUD ----------------------------------------------


async def test_receipt_crud(sm: SM) -> None:
    uid = await _user(sm, 111)
    async with sm() as s:
        r = await receipts.create_receipt(s, uid, "data/receipts/a.jpg")
        assert r.extraction is None
        got = await receipts.set_extraction(s, uid, r.id, {"total": 35000})
        assert got is not None and got.extraction == {"total": 35000}
        exp = await expenses.insert_expense(
            s,
            uid,
            amount_idr=35000,
            category_id=await _cat(sm, uid),
            occurred_at=T0,
            source="receipt",
            receipt_id=r.id,
        )
        assert exp.expense.receipt_id == r.id
        await s.rollback()
    async with sm() as s:
        r = await receipts.create_receipt(s, uid, "data/receipts/b.jpg")
        assert await receipts.delete_receipt(s, uid, r.id) is True
        assert await receipts.get_receipt(s, uid, r.id) is None


async def test_budget_crud(sm: SM) -> None:
    uid = await _user(sm, 111)
    cid = await _cat(sm, uid)
    async with sm() as s:
        b = await budgets.upsert_budget(s, uid, cid, 500_000)
        b2 = await budgets.upsert_budget(s, uid, cid, 700_000)  # one row per category
        assert b2.id == b.id and b2.monthly_limit_idr == 700_000
        got = await budgets.get_budget(s, uid, cid)
        assert got is not None and got.monthly_limit_idr == 700_000
        assert len(await budgets.list_budgets(s, uid)) == 1
        assert await budgets.delete_budget(s, uid, cid) is True
        assert await budgets.get_budget(s, uid, cid) is None
        with pytest.raises(ValueError):
            await budgets.upsert_budget(s, uid, cid, 0)


async def test_memory_crud(sm: SM) -> None:
    uid = await _user(sm, 111)
    async with sm() as s:
        m = await memories.add_memory(s, uid, "preference", "kopi = makan-minum", VEC)
        assert m.recall_count == 0 and len(m.embedding) == 384
        upd = await memories.update_memory(s, uid, m.id, content="kopi = hiburan", embedding=VEC)
        assert upd is not None and upd.content == "kopi = hiburan"
        assert [x.id for x in await memories.list_memories(s, uid)] == [m.id]
        assert await memories.delete_memory(s, uid, m.id) is True
        assert await memories.get_memory(s, uid, m.id) is None


async def test_audit_logs(sm: SM) -> None:
    uid = await _user(sm, 111)
    async with sm() as s:
        await audit.record_tool_call(
            s, uid, tool_name="add_expense", status="ok", latency_ms=5, args={"amount": "35rb"}
        )
        await audit.record_agent_run(
            s, uid, thread_id="t1", model="m", steps=2, duration_ms=10, status="ok"
        )
        assert [r.tool_name for r in await audit.list_tool_calls(s, uid)] == ["add_expense"]
        assert [r.thread_id for r in await audit.list_agent_runs(s, uid)] == ["t1"]


# --- isolation ------------------------------------------------------------------------------


async def test_user_a_cannot_touch_user_b_rows(sm: SM) -> None:
    a, b = await _user(sm, 1), await _user(sm, 2)
    b_cat = await _cat(sm, b)
    a_cat = await _cat(sm, a)
    b_exp = (await _expense(sm, b)).expense.id
    async with sm() as s:
        b_rec = (await receipts.create_receipt(s, b, "b.jpg")).id
        b_bud = (await budgets.upsert_budget(s, b, b_cat, 500_000)).id
        b_mem = (await memories.add_memory(s, b, "fact", "b secret", VEC)).id
        await audit.record_tool_call(s, b, tool_name="t", status="ok", latency_ms=1)
        await audit.record_agent_run(
            s, b, thread_id="t", model="m", steps=1, duration_ms=1, status="ok"
        )
        await s.commit()

    async with sm() as s:
        # reads -> None / empty
        assert await expenses.get_expense(s, a, b_exp) is None
        assert await expenses.list_expenses(s, a) == []
        assert await categories.get_category(s, a, b_cat) is None
        assert await receipts.get_receipt(s, a, b_rec) is None
        assert await budgets.get_budget(s, a, b_cat) is None
        assert await memories.get_memory(s, a, b_mem) is None
        assert await memories.list_memories(s, a) == []
        assert await audit.list_tool_calls(s, a) == []
        assert await audit.list_agent_runs(s, a) == []
        # updates -> None (0 rows affected)
        assert await expenses.update_expense(s, a, b_exp, {"amount_idr": 1}) is None
        assert await receipts.set_extraction(s, a, b_rec, {"hacked": True}) is None
        assert (await memories.update_memory(s, a, b_mem, content="pwned", embedding=VEC)) is None
        assert await users.update_user(s, a, display_name="mine") is not None  # own row only
        # deletes -> False (0 rows affected)
        assert await expenses.soft_delete_expense(s, a, b_exp) is False
        assert await receipts.delete_receipt(s, a, b_rec) is False
        assert await budgets.delete_budget(s, a, b_cat) is False
        assert await memories.delete_memory(s, a, b_mem) is False
        # B's category / receipt cannot be attached to A's rows
        with pytest.raises(NotOwnedError):
            await expenses.insert_expense(
                s, a, amount_idr=1, category_id=b_cat, occurred_at=T0, source="text"
            )
        with pytest.raises(NotOwnedError):
            await expenses.insert_expense(
                s, a, amount_idr=1, category_id=a_cat, occurred_at=T0, source="receipt",
                receipt_id=b_rec,
            )  # fmt: skip
        with pytest.raises(NotOwnedError):
            await expenses.update_expense(s, a, uuid.uuid4(), {"category_id": b_cat})
        with pytest.raises(NotOwnedError):
            await budgets.upsert_budget(s, a, b_cat, 1)
        await s.rollback()

    async with sm() as s:  # B's data is unchanged
        e = await expenses.get_expense(s, b, b_exp)
        assert e is not None and e.amount_idr == 35_000 and e.category_id == b_cat
        r = await receipts.get_receipt(s, b, b_rec)
        assert r is not None and r.extraction is None
        m = await memories.get_memory(s, b, b_mem)
        assert m is not None and m.content == "b secret"
        bud = await budgets.get_budget(s, b, b_cat)
        assert bud is not None and bud.id == b_bud and bud.monthly_limit_idr == 500_000
        assert len(await audit.list_tool_calls(s, b)) == 1
        assert len(await audit.list_agent_runs(s, b)) == 1
        ub = await users.get_user(s, b)
        assert ub is not None and ub.display_name is None


async def test_idempotency_key_does_not_leak_across_users(sm: SM) -> None:
    a, b = await _user(sm, 1), await _user(sm, 2)
    rb = await _expense(sm, b, key="shared")
    ra = await _expense(sm, a, key="shared")
    assert ra.created and ra.expense.user_id == a and rb.expense.user_id == b
