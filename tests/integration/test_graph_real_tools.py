"""T-202: graph + real registry tools + scripted model against the test DB."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.agent import graph as graph_mod
from duitku.agent.graph import build_graph, run_turn
from duitku.agent.nodes import repo_slug_loader
from duitku.db.repositories import users
from duitku.tools import clock
from duitku.tools.registry import build_tools

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import Responder, ScriptedChat


async def test_add_then_query_through_graph(
    sm: async_sessionmaker[AsyncSession],
    scripted_llm: "Callable[[Responder], ScriptedChat]",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clock, "now", lambda: datetime(2026, 10, 12, 3, 0, tzinfo=UTC))
    fake = SimpleNamespace(max_agent_steps=8, short_term_max_tokens=6000)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)
    async with sm() as s:
        user, _ = await users.get_or_create_user(s, 910000202)
        await s.commit()
    config: RunnableConfig = {"configurable": {"user_id": str(user.id), "timezone": "Asia/Jakarta"}}
    script = iter(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "a1",
                        "name": "add_expense",
                        "args": {"amount": "35rb", "category": "makan-minum", "merchant": "Warteg"},
                    }
                ],
            ),
            AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "q1",
                        "name": "query_expenses",
                        "args": {
                            "start_date": "2026-10-01",
                            "end_date": "2026-10-31",
                            "group_by": "none",
                        },
                    }
                ],
            ),
            AIMessage("Tercatat Rp35.000."),
        ]
    )
    llm = scripted_llm(lambda _: next(script))
    graph = build_graph(llm=llm, tools=build_tools(sm), category_slugs=repo_slug_loader(sm))
    out = await run_turn(graph, [HumanMessage("makan warteg 35rb")], config)

    results: list[dict[str, Any]] = [
        json.loads(str(m.content)) for m in out["messages"] if isinstance(m, ToolMessage)
    ]
    assert [r["ok"] for r in results] == [True, True]
    assert results[1]["data"]["total_idr"] == 35000
    assert out["messages"][-1].content == "Tercatat Rp35.000."
    system = llm.seen[0][0].content
    assert isinstance(system, str) and "makan-minum" in system  # slugs came from the DB
    async with sm() as s:
        rows = (await s.execute(text("SELECT amount_idr, idempotency_key FROM expenses"))).all()
    assert [tuple(r) for r in rows] == [(35000, "a1")]
