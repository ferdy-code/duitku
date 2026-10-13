"""T-405: the `approval` node is idempotent around `interrupt()` (real Postgres, scripted model).

Invariant. langgraph re-runs a node from its first line when it resumes after `interrupt()`, so
everything the `approval` node does before (or between) its `interrupt()` calls executes once per
pause AND once per resume. That is only safe if that code has no side effects. Here it is a
READ-ONLY, ownership-scoped lookup (`_human_summary`); every write (the soft delete, the audit
row) lives in the `tools` node, which runs once, after the final answer. Hence: N resumes of the
same pending question can never delete twice.

How the tests prove it:
- SQL spy: a `before_cursor_execute` listener on the engine records every statement issued WHILE
  the approval node body runs (the node is wrapped at build time). Each re-entry must issue only
  SELECTs, and at least one (so the spy is live).
- Table snapshots (rows in tool_audit_log / expenses / soft-deleted expenses / memories /
  agent_runs) are equal before the turn, at the pause, and after a rejection.
- Body-run spy: the node body is counted. 1 risky call = 2 runs (pause + resume); k risky calls =
  k + 1 runs, yet the effect is exactly k deletes / k audit rows, one ToolMessage per tool_call_id.
- Second resume: at the runner it is stopped by the "nothing pending" guard; on the raw graph
  `Command(resume=...)` against a FINISHED thread is a no-op in langgraph 1.2.14 (returns the
  stored final state, no node runs, no model call, no write). Both are asserted, not assumed.
- Restart: the pause lives in the Postgres checkpointer; a freshly built graph (new process
  stand-in) resumes it once and deletes once. The scripted-model response is consumed only by
  the final agent call, so a re-run of an earlier node would show up as an extra model call.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from tests.integration.test_delete_expense import (
    APPROVE,
    JKT,
    REJECT,
    add,
    cfg,
    codes,
    delete_call,
    live_amounts,
    make_graph,
    make_user,
    pending,
    script,
)
from tests.unit.test_graph import assert_paired

from duitku.agent import approval as approval_mod
from duitku.agent import graph as graph_mod
from duitku.agent.checkpointer import open_checkpointer
from duitku.agent.graph import Graph, resume_turn, run_turn
from duitku.agent.nodes import Node
from duitku.agent.state import AgentState
from duitku.interfaces.runner import NOTHING_PENDING_REPLY, TurnRunner, UserCtx
from duitku.tools import clock
from duitku.tools.registry import build_tools

if TYPE_CHECKING:
    from tests.conftest import Responder, ScriptedChat

SM = async_sessionmaker[AsyncSession]
Run = Callable[..., Awaitable[dict[str, Any]]]
type Llm = Callable[[Responder], ScriptedChat]
T0 = datetime(2026, 10, 12, 3, 0, tzinfo=UTC)
TABLES = ("tool_audit_log", "expenses", "memories", "agent_runs")


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:  # same stand-ins as test_delete_expense
    monkeypatch.setattr(clock, "now", lambda: T0)
    fake = SimpleNamespace(max_agent_steps=8, short_term_max_tokens=6000, log_content=False)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)


@pytest.fixture
def run(sm: SM) -> Run:
    tools = {t.name: t for t in build_tools(sm)}

    async def _run(name: str, args: dict[str, Any], uid: uuid.UUID, cid: str = "c") -> Any:
        config: Any = {"configurable": {"user_id": str(uid), "timezone": JKT}}
        call = {"type": "tool_call", "name": name, "args": args, "id": cid}
        return json.loads((await tools[name].ainvoke(call, config)).content)

    return _run


# --- spies ---------------------------------------------------------------------------------


@dataclass
class Probe:
    in_node: bool = False
    body_runs: int = 0
    summary_calls: list[str] = field(default_factory=lambda: list[str]())
    node_sql: list[str] = field(default_factory=lambda: list[str]())  # statements run in the node

    def verbs(self) -> set[str]:
        return {s.lstrip().split(None, 1)[0].upper() for s in self.node_sql}


@pytest.fixture
def probe(sm: SM, monkeypatch: pytest.MonkeyPatch) -> Probe:
    p = Probe()
    engine = cast(AsyncEngine, sm.kw["bind"]).sync_engine

    @event.listens_for(engine, "before_cursor_execute")
    def _record(_c: Any, _cur: Any, statement: str, *_a: Any) -> None:
        if p.in_node:
            p.node_sql.append(statement)

    real_summary = approval_mod._human_summary  # pyright: ignore[reportPrivateUsage]

    async def spy_summary(call: Any, config: Any, sessionmaker: Any) -> str | None:
        p.summary_calls.append(call["id"])
        return await real_summary(call, config, sessionmaker)

    monkeypatch.setattr(approval_mod, "_human_summary", spy_summary)

    real_make = graph_mod.make_approval_node

    def make_spied(sessionmaker: Any = None) -> Node:
        real_node = real_make(sessionmaker)

        async def node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            p.in_node, p.body_runs = True, p.body_runs + 1
            try:
                return await real_node(state, config)
            finally:  # GraphInterrupt propagates through here on every pause
                p.in_node = False

        return node

    monkeypatch.setattr(graph_mod, "make_approval_node", make_spied)
    return p


async def snapshot(sm: SM) -> dict[str, int]:
    async with sm() as s:
        out = {t: (await s.execute(text(f"SELECT count(*) FROM {t}"))).scalar_one() for t in TABLES}
        q = text("SELECT count(*) FROM expenses WHERE deleted_at IS NOT NULL")
        out["soft_deleted"] = (await s.execute(q)).scalar_one()
    return out


async def delete_audit_rows(sm: SM) -> int:
    q = text("SELECT count(*) FROM tool_audit_log WHERE tool_name = 'delete_expense'")
    async with sm() as s:
        return (await s.execute(q)).scalar_one()


async def n_messages(graph: Graph, uid: uuid.UUID, thread: str = "t") -> int:
    return len((await graph.aget_state(cfg(uid, thread))).values["messages"])


def tool_msgs(messages: list[Any]) -> list[str]:
    return [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]


# --- 1. nothing but read-only SQL before the answer ----------------------------------------


async def test_node_only_reads_until_decided_and_tables_are_unchanged(
    sm: SM, run: Run, scripted_llm: Llm, probe: Probe
) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a", merchant="Kopi")
    before = await snapshot(sm)
    graph = make_graph(
        sm, scripted_llm(script(AIMessage("", tool_calls=[delete_call(eid)]), AIMessage("ok")))
    )

    out = await run_turn(graph, [HumanMessage("hapus")], cfg(uid))
    assert [p["tool_call_id"] for p in pending(out)] == ["d1"]
    assert probe.body_runs == 1 and probe.node_sql  # spy is live: the lookup did run
    assert probe.verbs() == {"SELECT"}
    assert await snapshot(sm) == before  # paused: not one row differs

    out = await resume_turn(graph, REJECT, cfg(uid))  # re-entry: read again, still no writes
    assert probe.body_runs == 2 and probe.verbs() == {"SELECT"}
    assert await snapshot(sm) == before  # rejected: the tool never ran, no audit row either
    assert codes(out["messages"]) == {"d1": "REJECTED_BY_USER"}


async def test_approve_writes_happen_after_the_node_only(
    sm: SM, run: Run, scripted_llm: Llm, probe: Probe
) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a")
    before = await snapshot(sm)
    graph = make_graph(
        sm, scripted_llm(script(AIMessage("", tool_calls=[delete_call(eid)]), AIMessage("ok")))
    )
    await run_turn(graph, [HumanMessage("hapus")], cfg(uid))
    await resume_turn(graph, APPROVE, cfg(uid))

    assert probe.verbs() == {"SELECT"}  # writes exist now, but none came from inside the node
    after = await snapshot(sm)
    assert after["soft_deleted"] == before["soft_deleted"] + 1
    assert after["tool_audit_log"] == before["tool_audit_log"] + 1
    assert after["expenses"] == before["expenses"]  # soft delete: no row removed
    assert after["memories"] == before["memories"] and after["agent_runs"] == before["agent_runs"]


# --- 2. resume twice -----------------------------------------------------------------------


async def test_second_resume_is_stopped_by_the_runner_guard(
    sm: SM, run: Run, scripted_llm: Llm, probe: Probe
) -> None:
    user = UserCtx(await make_user(sm, 1), JKT)
    eid = await add(run, user.user_id, "18rb", "a")
    llm = scripted_llm(
        script(AIMessage("", tool_calls=[delete_call(eid)]), AIMessage("Dihapus."), AIMessage("!"))
    )
    graph = make_graph(sm, llm)
    runner = TurnRunner(graph, sm)

    assert (await runner.run("cli:t", user, "hapus")).pending is not None
    assert (await runner.resume("cli:t", user, "approve")).text == "Dihapus."
    done = (await snapshot(sm), await n_messages(graph, user.user_id, "cli:t"), len(llm.seen))
    runs = probe.body_runs
    assert await live_amounts(sm, user.user_id) == []

    for decision in ("approve", "approve", "reject"):
        assert (await runner.resume("cli:t", user, decision)).text == NOTHING_PENDING_REPLY  # type: ignore[arg-type]
    assert (
        await snapshot(sm),
        await n_messages(graph, user.user_id, "cli:t"),
        len(llm.seen),
    ) == done
    assert probe.body_runs == runs  # the node did not even re-enter
    assert (await delete_audit_rows(sm)) == 1


async def test_second_raw_resume_on_a_finished_thread_is_a_noop(
    sm: SM, run: Run, scripted_llm: Llm, probe: Probe
) -> None:
    """Finding (not a bug): langgraph 1.2.14 ignores `Command(resume=...)` on a thread that ran to
    END. It returns the stored final state; no node runs. The runner guard is still needed for a
    thread that never started (empty state would run from START), but not for this case."""
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a")
    llm = scripted_llm(
        script(AIMessage("", tool_calls=[delete_call(eid)]), AIMessage("ok"), AIMessage("2nd"))
    )
    graph = make_graph(sm, llm)
    await run_turn(graph, [HumanMessage("hapus")], cfg(uid))
    first = await resume_turn(graph, APPROVE, cfg(uid))
    done = (await snapshot(sm), await n_messages(graph, uid), len(llm.seen), probe.body_runs)

    for answer in (APPROVE, REJECT):  # via resume_turn and via a hand-built Command
        again = await resume_turn(graph, answer, cfg(uid))
        assert again["messages"] == first["messages"]
    resume = Command(resume={"decision": "approve"})  # bypasses resume_turn on purpose
    raw = await graph.ainvoke(resume, cfg(uid))  # pyright: ignore[reportUnknownMemberType]
    assert raw["messages"] == first["messages"]

    assert (
        await snapshot(sm),
        await n_messages(graph, uid),
        len(llm.seen),
        probe.body_runs,
    ) == done
    assert await live_amounts(sm, uid) == [] and await delete_audit_rows(sm) == 1
    assert (await graph.aget_state(cfg(uid))).next == ()
    assert tool_msgs(first["messages"]) == ["d1"]
    assert_paired(first["messages"])


# --- 3. re-entry on resume -----------------------------------------------------------------


async def test_node_body_runs_twice_for_one_risky_call_but_deletes_once(
    sm: SM, run: Run, scripted_llm: Llm, probe: Probe
) -> None:
    uid = await make_user(sm, 1)
    eid = await add(run, uid, "18rb", "a")
    graph = make_graph(
        sm, scripted_llm(script(AIMessage("", tool_calls=[delete_call(eid)]), AIMessage("ok")))
    )
    await run_turn(graph, [HumanMessage("hapus")], cfg(uid))
    assert probe.body_runs == 1 and probe.summary_calls == ["d1"]
    out = await resume_turn(graph, APPROVE, cfg(uid))

    assert probe.body_runs == 2 and probe.summary_calls == ["d1", "d1"]  # re-ran from the top
    assert await live_amounts(sm, uid) == [] and await delete_audit_rows(sm) == 1
    assert tool_msgs(out["messages"]) == ["d1"]
    assert_paired(out["messages"])


async def test_two_risky_calls_in_one_message_delete_exactly_twice(
    sm: SM, run: Run, scripted_llm: Llm, probe: Probe
) -> None:
    uid = await make_user(sm, 1)
    e1 = await add(run, uid, "10rb", "a", merchant="Satu")
    e2 = await add(run, uid, "20rb", "b", merchant="Dua")
    e3 = await add(run, uid, "30rb", "c", merchant="Tiga")
    safe = {"id": "s1", "name": "add_expense", "args": {"amount": "5rb", "category": "transport"}}
    graph = make_graph(
        sm,
        scripted_llm(
            script(
                AIMessage("", tool_calls=[safe, delete_call(e1, 1), delete_call(e2, 2)]),
                AIMessage("ok"),
            )
        ),
    )
    out = await run_turn(graph, [HumanMessage("hapus dua")], cfg(uid))
    assert [p["tool_call_id"] for p in pending(out)] == ["d1"]

    out = await resume_turn(graph, APPROVE, cfg(uid))
    assert [p["tool_call_id"] for p in pending(out)] == ["d2"]  # second pause, same node
    assert await live_amounts(sm, uid) == [10_000, 20_000, 30_000]  # nothing ran yet, not even s1
    assert await delete_audit_rows(sm) == 0

    out = await resume_turn(graph, APPROVE, cfg(uid))
    assert pending(out) == []
    assert probe.body_runs == 3  # pause 1, pause 2, final resume
    assert probe.summary_calls == ["d1", "d1", "d2", "d1", "d2"]  # replayed lookups, no writes
    assert probe.verbs() == {"SELECT"}
    assert await live_amounts(sm, uid) == [5_000, 30_000]  # d1, d2 gone once each; s1 added once
    assert await delete_audit_rows(sm) == 2
    assert e3
    ids = tool_msgs(out["messages"])
    assert sorted(ids) == ["d1", "d2", "s1"] and len(ids) == len(set(ids))  # no duplicates
    assert_paired(out["messages"])


# --- 4. rejected path ----------------------------------------------------------------------


async def test_reject_then_resume_again_changes_nothing(
    sm: SM, run: Run, scripted_llm: Llm, probe: Probe
) -> None:
    user = UserCtx(await make_user(sm, 1), JKT)
    eid = await add(run, user.user_id, "18rb", "a")
    before = await snapshot(sm)
    llm = scripted_llm(
        script(AIMessage("", tool_calls=[delete_call(eid)]), AIMessage("Batal."), AIMessage("!"))
    )
    graph = make_graph(sm, llm)
    runner = TurnRunner(graph, sm)

    await runner.run("cli:t", user, "hapus")
    assert (await runner.resume("cli:t", user, "reject")).text == "Batal."
    settled = (len(llm.seen), probe.body_runs)

    # runner layer (guard) and graph layer (finished thread) both leave everything alone
    assert (await runner.resume("cli:t", user, "approve")).text == NOTHING_PENDING_REPLY
    graph_cfg = cfg(user.user_id, "cli:t")
    after_raw = await resume_turn(graph, APPROVE, graph_cfg)  # a late "approve" must not revive it
    assert codes(after_raw["messages"]) == {"d1": "REJECTED_BY_USER"}
    assert (len(llm.seen), probe.body_runs) == settled
    after = await snapshot(sm)
    # T-406: the runner logs each graph call (pause + resume); nothing else changed. The two
    # guarded calls (nothing pending / finished thread) ran no turn, so they add no rows.
    assert after["agent_runs"] == before["agent_runs"] + 2
    assert {**after, "agent_runs": 0} == {**before, "agent_runs": 0}
    assert await live_amounts(sm, user.user_id) == [18_000]


# --- 5. crash between pause and resume -----------------------------------------------------


async def test_restart_between_every_step_still_deletes_exactly_once_each(
    sm: SM, run: Run, scripted_llm: Llm, test_db_url: str, probe: Probe
) -> None:
    """Extends test_pending_approval_survives_a_restart: two risky calls, a brand-new graph and
    checkpointer connection before EVERY step (run, resume #1, resume #2, resume #3 = late dup)."""
    from sqlalchemy.engine import make_url

    url = make_url(test_db_url).set(drivername="postgresql").render_as_string(hide_password=False)
    user = UserCtx(await make_user(sm, 1), JKT)
    e1 = await add(run, user.user_id, "10rb", "a")
    e2 = await add(run, user.user_id, "20rb", "b")
    thread = "tg:1"

    async def step(do: Callable[[TurnRunner], Awaitable[Any]], *msgs: AIMessage) -> Any:
        async with open_checkpointer(url) as saver:  # "process start"
            llm = scripted_llm(script(*msgs))
            return await do(TurnRunner(make_graph(sm, llm, saver), sm))

    r = await step(
        lambda t: t.run(thread, user, "hapus dua"),
        AIMessage("", tool_calls=[delete_call(e1, 1), delete_call(e2, 2)]),
    )
    assert r.pending is not None and r.pending["tool_call_id"] == "d1"
    assert await live_amounts(sm, user.user_id) == [10_000, 20_000]

    r = await step(lambda t: t.resume(thread, user, "approve"))  # no model call needed: pauses
    assert r.pending is not None and r.pending["tool_call_id"] == "d2"
    assert await live_amounts(sm, user.user_id) == [10_000, 20_000]

    r = await step(lambda t: t.resume(thread, user, "approve"), AIMessage("Dua-duanya dihapus."))
    assert r.pending is None and r.text == "Dua-duanya dihapus."
    assert await live_amounts(sm, user.user_id) == [] and await delete_audit_rows(sm) == 2
    done = await snapshot(sm)

    # the user taps "approve" again after yet another restart: nothing happens
    r = await step(lambda t: t.resume(thread, user, "approve"))
    assert r.text == NOTHING_PENDING_REPLY
    assert await snapshot(sm) == done and probe.verbs() == {"SELECT"}
