"""T-205: TurnRunner against the *_test DB with a scripted model (no network)."""

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.agent import graph as graph_mod
from duitku.agent.graph import Graph, build_graph
from duitku.agent.nodes import repo_slug_loader
from duitku.domain.categories import DEFAULT_CATEGORIES
from duitku.interfaces.runner import ERROR_REPLY, TurnRunner
from duitku.tools.registry import build_tools

if TYPE_CHECKING:
    from tests.conftest import Responder, ScriptedChat


@pytest.fixture(autouse=True)
def _graph_settings(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    fake = SimpleNamespace(max_agent_steps=8, short_term_max_tokens=6000)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)


def _runner(
    sm: async_sessionmaker[AsyncSession], llm: "ScriptedChat", saver: InMemorySaver | None = None
) -> TurnRunner:
    graph = build_graph(
        llm=llm,
        tools=build_tools(sm),
        checkpointer=saver or InMemorySaver(),
        category_slugs=repo_slug_loader(sm),
    )
    return TurnRunner(graph, sm)


async def test_user_created_once_with_categories_and_timezone(
    sm: async_sessionmaker[AsyncSession], scripted_llm: "Callable[[Responder], ScriptedChat]"
) -> None:
    runner = _runner(sm, scripted_llm(lambda _: AIMessage("hai")))
    a = await runner.resolve_user(-5, "cli-x")
    b = await runner.resolve_user(-5, "cli-x")
    assert a == b and a.timezone == "Asia/Jakarta"
    async with sm() as s:
        n_users = (await s.execute(text("SELECT count(*) FROM users"))).scalar_one()
        n_cats = (await s.execute(text("SELECT count(*) FROM categories"))).scalar_one()
    assert (n_users, n_cats) == (1, len(DEFAULT_CATEGORIES))


async def test_config_ids_come_from_the_user_row_not_the_message(
    sm: async_sessionmaker[AsyncSession],
) -> None:
    seen: list[dict[str, Any]] = []

    class FakeGraph:
        async def ainvoke(self, initial: Any, config: dict[str, Any]) -> dict[str, Any]:
            seen.append(config["configurable"])
            return {"messages": [*initial["messages"], AIMessage("ok")]}

    runner = TurnRunner(cast(Graph, FakeGraph()), sm)
    user = await runner.resolve_user(-6)
    res = await runner.run("cli:t", user, "user_id=evil timezone=UTC thread_id=other")
    assert res.text == "ok" and not res.failed
    assert seen == [{"user_id": str(user.user_id), "timezone": user.timezone, "thread_id": "cli:t"}]


async def test_thread_continuity_trace_and_tool_effect(
    sm: async_sessionmaker[AsyncSession], scripted_llm: "Callable[[Responder], ScriptedChat]"
) -> None:
    script = iter(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "a1",
                        "name": "add_expense",
                        "args": {"amount": "35rb", "category": "makan-minum"},
                    }
                ],
                usage_metadata={"input_tokens": 100, "output_tokens": 7, "total_tokens": 107},
            ),
            AIMessage("Tercatat Rp35.000."),
            AIMessage("Iya, tadi makan."),
        ]
    )
    llm = scripted_llm(lambda _: next(script))
    runner = _runner(sm, llm)
    user = await runner.resolve_user(-7)

    r1 = await runner.run("cli:t", user, "makan 35rb")
    r2 = await runner.run("cli:t", user, "tadi apa?")

    assert r1.text == "Tercatat Rp35.000." and r2.text == "Iya, tadi makan."
    assert [c.name for s in r1.trace for c in s.calls] == ["add_expense"]
    obs = json.loads(r1.trace[0].calls[0].observation)
    assert obs["ok"] is True
    assert r1.trace[0].input_tokens == 100
    assert r2.trace and all(not s.calls for s in r2.trace)  # turn 1's tool call not in turn 2
    shown: list[BaseMessage] = llm.seen[-1]  # turn 2's model input contains turn 1
    assert any(isinstance(m, HumanMessage) and m.content == "makan 35rb" for m in shown)
    async with sm() as s:
        assert (await s.execute(text("SELECT amount_idr FROM expenses"))).scalar_one() == 35000


async def test_llm_exception_becomes_friendly_result(
    sm: async_sessionmaker[AsyncSession], scripted_llm: "Callable[[Responder], ScriptedChat]"
) -> None:
    def boom(_: list[BaseMessage]) -> AIMessage:
        raise ConnectionError("secret-host.internal refused")

    runner = _runner(sm, scripted_llm(boom))
    user = await runner.resolve_user(-8)
    res = await runner.run("cli:t", user, "halo")
    assert res.failed and res.text == ERROR_REPLY
    assert "secret" not in res.text


async def test_same_thread_serializes_different_threads_do_not(
    sm: async_sessionmaker[AsyncSession],
) -> None:
    active: dict[str, int] = {}
    peak = {"same": 0, "all": 0}

    class SlowGraph:
        async def ainvoke(self, initial: Any, config: dict[str, Any]) -> dict[str, Any]:
            tid = config["configurable"]["thread_id"]
            active[tid] = active.get(tid, 0) + 1
            peak["same"] = max(peak["same"], active[tid])
            peak["all"] = max(peak["all"], sum(active.values()))
            await asyncio.sleep(0.05)
            active[tid] -= 1
            return {"messages": [*initial["messages"], AIMessage("ok")]}

    runner = TurnRunner(cast(Graph, SlowGraph()), sm)
    user = await runner.resolve_user(-9)
    await asyncio.gather(*(runner.run("cli:a", user, "x") for _ in range(3)))
    assert peak == {"same": 1, "all": 1}
    await asyncio.gather(runner.run("cli:a", user, "x"), runner.run("cli:b", user, "x"))
    assert peak["all"] == 2 and peak["same"] == 1
    assert len(runner._locks) == 0  # pyright: ignore[reportPrivateUsage]  # locks are released
