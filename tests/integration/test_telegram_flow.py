"""T-401: Telegram update -> guard -> handler -> real TurnRunner/graph/DB (scripted model)."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.fake_embedder import FakeEmbedder
from tests.telegram_fakes import FakeBot, make_app, text_update

from duitku.agent import graph as graph_mod
from duitku.agent.graph import build_graph
from duitku.agent.nodes import repo_slug_loader
from duitku.interfaces.runner import TurnRunner
from duitku.tools.registry import build_tools

if TYPE_CHECKING:
    from tests.conftest import Responder, ScriptedChat


@pytest.fixture(autouse=True)
def _graph_settings(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    fake = SimpleNamespace(max_agent_steps=8, short_term_max_tokens=6000)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)


async def test_allowed_user_full_path_and_stranger_leaves_no_trace(
    sm: async_sessionmaker[AsyncSession], scripted_llm: "Callable[[Responder], ScriptedChat]"
) -> None:
    graph = build_graph(
        llm=scripted_llm(lambda _: AIMessage("Dicatat <kopi> & selesai")),
        tools=build_tools(sm),
        checkpointer=InMemorySaver(),
        category_slugs=repo_slug_loader(sm),
        sessionmaker=sm,
        embedder=FakeEmbedder({}, fail=True),
    )
    bot = FakeBot()
    app = await make_app(bot, TurnRunner(graph, sm), allowed=[42])

    await app.process_update(text_update(bot, "kopi 18rb", user_id=999))  # stranger
    async with sm() as s:
        assert (await s.execute(text("SELECT count(*) FROM users"))).scalar_one() == 0
    assert bot.calls == []

    await app.process_update(text_update(bot, "kopi 18rb", user_id=42))
    assert [m["text"] for m in bot.sent()] == ["Dicatat &lt;kopi&gt; &amp; selesai"]
    async with sm() as s:
        row = (await s.execute(text("SELECT telegram_user_id FROM users"))).scalar_one()
    assert row == 42
    state = await graph.aget_state({"configurable": {"thread_id": "tg:42"}})
    assert len(state.values["messages"]) == 2  # the thread is keyed by chat id
