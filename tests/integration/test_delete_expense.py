"""T-402: delete_expense tool + approval flow through graph and runner (real DB, scripted model)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.fake_embedder import FakeEmbedder
from tests.unit.test_graph import assert_paired

from duitku.agent import graph as graph_mod
from duitku.agent.checkpointer import open_checkpointer
from duitku.agent.graph import Graph, build_graph, resume_turn, run_turn
from duitku.agent.nodes import repo_slug_loader
from duitku.db.repositories import audit, expenses, users
from duitku.interfaces.runner import NOTHING_PENDING_REPLY, TurnRunner, UserCtx
from duitku.tools import clock
from duitku.tools.registry import build_tools

if TYPE_CHECKING:
    from tests.conftest import Responder, ScriptedChat

SM = async_sessionmaker[AsyncSession]
Run = Callable[..., Awaitable[dict[str, Any]]]
type Llm = Callable[[Responder], ScriptedChat]
JKT = "Asia/Jakarta"
T0 = datetime(2026, 10, 12, 3, 0, tzinfo=UTC)  # 2026-10-12 10:00 Jakarta
APPROVE = {"decision": "approve", "reason": None}
REJECT = {"decision": "reject", "reason": "nggak jadi"}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clock, "now", lambda: T0)
    fake = SimpleNamespace(max_agent_steps=8, short_term_max_tokens=6000, log_content=False)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)


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


async def live_amounts(sm: SM, uid: uuid.UUID) -> list[int]:
    q = text("SELECT amount_idr FROM expenses WHERE user_id = :u AND deleted_at IS NULL ORDER BY 1")
    async with sm() as s:
        return [r[0] for r in (await s.execute(q, {"u": uid})).all()]


def delete_call(eid: str, i: int = 1) -> dict[str, Any]:
    return {"id": f"d{i}", "name": "delete_expense", "args": {"expense_id": eid}}


def script(*msgs: AIMessage) -> Callable[[Any], AIMessage]:
    it = iter(msgs)
    return lambda _: next(it)


def codes(messages: list[AnyMessage]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            body = json.loads(str(m.content))
            out[m.tool_call_id] = None if body["ok"] else body["error"]["code"]
    return out


def pending(out: dict[str, Any]) -> list[dict[str, Any]]:
    return [i.value for i in out.get("__interrupt__", [])]


def make_graph(sm: SM, llm: ScriptedChat, saver: Any = None) -> Graph:
    return build_graph(
        llm=llm,
        tools=build_tools(sm),
        checkpointer=saver or InMemorySaver(),
        category_slugs=repo_slug_loader(sm),
        sessionmaker=sm,
        embedder=FakeEmbedder({}, fail=True),
    )


def cfg(uid: uuid.UUID, thread: str = "t") -> RunnableConfig:
    return {"configurable": {"user_id": str(uid), "timezone": JKT, "thread_id": thread}}


# --- (i) tool unit tests -------------------------------------------------------------------


async def test_delete_soft_deletes_and_returns_summary(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a", merchant="Kopi Kenangan")
    res = await run("delete_expense", {"expense_id": eid}, uid)
    assert res["ok"] is True
    assert res["data"] == {
        "expense_id": eid,
        "amount_idr": 18_000,
        "category": "makan-minum",
        "merchant": "Kopi Kenangan",
        "note": None,
        "occurred_on": "2026-10-12",
        "deleted": True,
    }
    assert await live_amounts(sm, uid) == []
    async with sm() as s:  # soft: the row is still there, marked
        n = await s.execute(text("SELECT count(*) FROM expenses WHERE deleted_at IS NOT NULL"))
        assert n.scalar_one() == 1
        assert "delete_expense" in {r.tool_name for r in await audit.list_tool_calls(s, uid)}


async def test_delete_twice_and_unknown_id_are_not_found(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a")
    assert (await run("delete_expense", {"expense_id": eid}, uid))["ok"] is True
    for target in (eid, str(uuid.uuid4())):
        res = await run("delete_expense", {"expense_id": target}, uid)
        assert res["error"]["code"] == "NOT_FOUND" and res["error"]["hint"]


async def test_delete_foreign_expense_is_not_found_and_row_intact(sm: SM, run: Run) -> None:
    owner, other = await make_user(sm, 1), await make_user(sm, 2)
    eid = await add(run, owner, "18rb", "a")
    res = await run("delete_expense", {"expense_id": eid}, other)
    assert res["error"]["code"] == "NOT_FOUND"
    assert await live_amounts(sm, owner) == [18_000]


@pytest.mark.parametrize("bad", [{}, {"expense_id": "not-a-uuid"}, {"expense_id": 5}])
async def test_delete_validation_errors_are_envelopes(
    sm: SM, run: Run, bad: dict[str, Any]
) -> None:
    uid = await make_user(sm, 1)
    res = await run("delete_expense", bad, uid)
    assert res["ok"] is False and res["error"]["code"] == "INVALID_ARGUMENT"


# --- graph flows ---------------------------------------------------------------------------


async def test_approve_deletes_and_agent_replies(sm: SM, run: Run, scripted_llm: Llm) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a", merchant="Kopi")
    graph = make_graph(
        sm,
        scripted_llm(script(AIMessage("", tool_calls=[delete_call(eid)]), AIMessage("Dihapus."))),
    )
    out = await run_turn(graph, [HumanMessage("hapus yang tadi")], cfg(uid))

    (p,) = pending(out)
    assert p["tool"] == "delete_expense" and p["tool_call_id"] == "d1"
    assert p["human_summary"] == "Hapus pengeluaran: Kopi Rp18.000 (12 Okt)?"
    # (c) nothing changed while paused, and the state shows the run waiting at approval
    assert await live_amounts(sm, uid) == [18_000]
    assert (await graph.aget_state(cfg(uid))).next == ("approval",)

    out = await resume_turn(graph, APPROVE, cfg(uid))
    assert await live_amounts(sm, uid) == []
    assert codes(out["messages"]) == {"d1": None}
    assert out["messages"][-1].content == "Dihapus."
    assert_paired(out["messages"])


async def test_reject_leaves_db_untouched_and_agent_responds(
    sm: SM, run: Run, scripted_llm: Llm
) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a")
    llm = scripted_llm(
        script(AIMessage("", tool_calls=[delete_call(eid)]), AIMessage("Oke, dibatalkan."))
    )
    graph = make_graph(sm, llm)
    await run_turn(graph, [HumanMessage("hapus")], cfg(uid))
    out = await resume_turn(graph, REJECT, cfg(uid))

    assert await live_amounts(sm, uid) == [18_000]
    assert codes(out["messages"]) == {"d1": "REJECTED_BY_USER"}
    assert out["messages"][-1].content == "Oke, dibatalkan."
    async with sm() as s:  # the tool never ran, so it left no audit row either
        assert [r.tool_name for r in await audit.list_tool_calls(s, uid)] == ["add_expense"]
    assert_paired(out["messages"])


async def test_mixed_message_safe_call_runs_risky_waits(
    sm: SM, run: Run, scripted_llm: Llm
) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a")
    add_call = {
        "id": "n1",
        "name": "add_expense",
        "args": {"amount": "5rb", "category": "transport"},
    }
    graph = make_graph(
        sm,
        scripted_llm(
            script(AIMessage("", tool_calls=[add_call, delete_call(eid, 2)]), AIMessage("ok"))
        ),
    )
    out = await run_turn(graph, [HumanMessage("x")], cfg(uid))
    assert [p["tool_call_id"] for p in pending(out)] == ["d2"]
    assert await live_amounts(sm, uid) == [18_000]  # the safe add waits for the whole message
    out = await resume_turn(graph, REJECT, cfg(uid))
    assert await live_amounts(sm, uid) == [5_000, 18_000]
    assert codes(out["messages"]) == {"n1": None, "d2": "REJECTED_BY_USER"}
    assert_paired(out["messages"])


async def test_two_deletes_approve_one_reject_other(sm: SM, run: Run, scripted_llm: Llm) -> None:
    uid = await make_user(sm, 1)
    e1 = await add(run, uid, "10rb", "a", merchant="Satu")
    e2 = await add(run, uid, "20rb", "b", merchant="Dua")
    e3 = await add(run, uid, "30rb", "c", merchant="Tiga")
    graph = make_graph(
        sm,
        scripted_llm(
            script(
                AIMessage("", tool_calls=[delete_call(e1, 1), delete_call(e2, 2)]),
                AIMessage("ok"),
            )
        ),
    )
    out = await run_turn(graph, [HumanMessage("hapus dua")], cfg(uid))
    assert [p["human_summary"] for p in pending(out)] == [
        "Hapus pengeluaran: Satu Rp10.000 (12 Okt)?"
    ]
    out = await resume_turn(graph, APPROVE, cfg(uid))
    assert [p["human_summary"] for p in pending(out)] == [
        "Hapus pengeluaran: Dua Rp20.000 (12 Okt)?"
    ]
    assert await live_amounts(sm, uid) == [10_000, 20_000, 30_000]  # still nothing deleted
    out = await resume_turn(graph, REJECT, cfg(uid))
    assert await live_amounts(sm, uid) == [20_000, 30_000]
    assert codes(out["messages"]) == {"d1": None, "d2": "REJECTED_BY_USER"}
    assert e3  # untouched third row
    assert_paired(out["messages"])


async def test_foreign_expense_id_skips_the_interrupt(sm: SM, run: Run, scripted_llm: Llm) -> None:
    owner, other = await make_user(sm, 1), await make_user(sm, 2)
    eid = await add(run, owner, "18rb", "a")
    graph = make_graph(
        sm,
        scripted_llm(
            script(AIMessage("", tool_calls=[delete_call(eid)]), AIMessage("Tidak ketemu."))
        ),
    )
    out = await run_turn(graph, [HumanMessage("hapus")], cfg(other))
    assert pending(out) == []  # no question about someone else's row, no existence leak
    assert codes(out["messages"]) == {"d1": "NOT_FOUND"}
    assert await live_amounts(sm, owner) == [18_000]
    assert_paired(out["messages"])


async def test_bad_or_missing_id_skips_the_interrupt(sm: SM, run: Run, scripted_llm: Llm) -> None:
    uid = await make_user(sm, 1)
    calls: list[dict[str, Any]] = [
        {"id": "d1", "name": "delete_expense", "args": {"expense_id": "nope"}},
        {"id": "d2", "name": "delete_expense", "args": {}},
    ]
    graph = make_graph(sm, scripted_llm(script(AIMessage("", tool_calls=calls), AIMessage("ok"))))
    out = await run_turn(graph, [HumanMessage("hapus")], cfg(uid))
    assert pending(out) == []
    assert codes(out["messages"]) == {"d1": "INVALID_ARGUMENT", "d2": "INVALID_ARGUMENT"}


async def test_deleted_between_pause_and_resume_is_handled_by_the_tool(
    sm: SM, run: Run, scripted_llm: Llm
) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a")
    graph = make_graph(
        sm, scripted_llm(script(AIMessage("", tool_calls=[delete_call(eid)]), AIMessage("ok")))
    )
    await run_turn(graph, [HumanMessage("hapus")], cfg(uid))
    async with sm() as s:  # e.g. deleted from another chat in the meantime
        await expenses.soft_delete_expense(s, uid, uuid.UUID(eid))
        await s.commit()
    out = await resume_turn(graph, APPROVE, cfg(uid))
    assert codes(out["messages"]) == {"d1": "NOT_FOUND"}


# --- (j) runner ----------------------------------------------------------------------------


async def test_runner_pending_resume_and_guards(sm: SM, run: Run, scripted_llm: Llm) -> None:
    user = UserCtx(await make_user(sm, -77), JKT)
    eid = await add(run, user.user_id, "18rb", "a", merchant="Kopi")
    runner = TurnRunner(
        make_graph(
            sm,
            scripted_llm(
                script(
                    AIMessage("", tool_calls=[delete_call(eid)]),
                    AIMessage("Dihapus ya."),
                    AIMessage("Siap."),
                )
            ),
        ),
        sm,
    )

    assert (await runner.resume("cli:t", user, "approve")).text == NOTHING_PENDING_REPLY

    r1 = await runner.run("cli:t", user, "hapus yang tadi")
    assert r1.pending is not None and r1.pending["tool"] == "delete_expense"
    assert r1.text == "Hapus pengeluaran: Kopi Rp18.000 (12 Okt)?" and not r1.failed
    assert await live_amounts(sm, user.user_id) == [18_000]

    r2 = await runner.run(
        "cli:t", user, "eh sebentar"
    )  # message while paused: re-asks, no new turn
    assert r2.pending == r1.pending

    r3 = await runner.resume("cli:t", user, "approve")
    assert r3.pending is None and r3.text == "Dihapus ya."
    assert [c.name for s in r3.trace for c in s.calls] == ["delete_expense"]
    assert await live_amounts(sm, user.user_id) == []

    r4 = await runner.run("cli:t", user, "makasih")  # thread is healthy again
    assert r4.text == "Siap."


async def test_pending_approval_survives_a_restart(
    sm: SM, run: Run, scripted_llm: Llm, test_db_url: str
) -> None:
    """Postgres checkpointer: pause, drop everything in memory, rebuild, resume."""
    from sqlalchemy.engine import make_url

    url = make_url(test_db_url).set(drivername="postgresql").render_as_string(hide_password=False)
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a")
    user = UserCtx(uid, JKT)
    async with open_checkpointer(url) as saver:
        first = TurnRunner(
            make_graph(
                sm, scripted_llm(script(AIMessage("", tool_calls=[delete_call(eid)]))), saver
            ),
            sm,
        )
        assert (await first.run("tg:1", user, "hapus")).pending is not None
    async with open_checkpointer(url) as saver:  # "bot restarted"
        second = TurnRunner(make_graph(sm, scripted_llm(script(AIMessage("Dihapus."))), saver), sm)
        res = await second.resume("tg:1", user, "approve")
    assert res.text == "Dihapus." and await live_amounts(sm, uid) == []
