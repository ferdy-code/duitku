"""T-403: "hapus yang tadi" through the Telegram handlers with the REAL TurnRunner + DB + graph
(scripted model, in-memory checkpointer): question with buttons -> press -> soft delete / kept."""

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.fake_embedder import FakeEmbedder
from tests.telegram_fakes import FakeBot, callback_update, make_app, text_update

from duitku.agent import graph as graph_mod
from duitku.agent.graph import build_graph
from duitku.agent.nodes import repo_slug_loader
from duitku.db.repositories import users
from duitku.interfaces.runner import NOTHING_PENDING_REPLY, TurnRunner
from duitku.interfaces.telegram import texts
from duitku.tools.registry import build_tools

if TYPE_CHECKING:
    from tests.conftest import Responder, ScriptedChat

TG = 42


@pytest.fixture(autouse=True)
def _graph_settings(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    fake = SimpleNamespace(max_agent_steps=8, short_term_max_tokens=6000, log_content=False)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)


async def _seed_expense(sm: async_sessionmaker[AsyncSession]) -> str:
    async with sm() as s:
        user, _ = await users.get_or_create_user(s, TG)
        await s.commit()
    config: RunnableConfig = {"configurable": {"user_id": str(user.id), "timezone": "Asia/Jakarta"}}
    call = {
        "type": "tool_call",
        "name": "add_expense",
        "args": {"amount": "18rb", "category": "makan-minum", "merchant": "Kopi"},
        "id": "seed",
    }
    tool = {t.name: t for t in build_tools(sm)}["add_expense"]
    res = json.loads((await tool.ainvoke(call, config)).content)
    assert res["ok"], res
    return res["data"]["expense_id"]


async def _live(sm: async_sessionmaker[AsyncSession]) -> int:
    q = text("SELECT count(*) FROM expenses WHERE deleted_at IS NULL")
    async with sm() as s:
        return (await s.execute(q)).scalar_one()


async def _flow(
    sm: async_sessionmaker[AsyncSession],
    scripted_llm: "Callable[[Responder], ScriptedChat]",
    press: str,
    final: str,
) -> tuple[FakeBot, Any, Any]:
    eid = await _seed_expense(sm)
    script = iter(
        [
            AIMessage(
                "", tool_calls=[{"id": "d1", "name": "delete_expense", "args": {"expense_id": eid}}]
            ),
            AIMessage(final),
        ]
    )
    graph = build_graph(
        llm=scripted_llm(lambda _: next(script)),
        tools=build_tools(sm),
        checkpointer=InMemorySaver(),
        category_slugs=repo_slug_loader(sm),
        sessionmaker=sm,
        embedder=FakeEmbedder({}, fail=True),
    )
    bot = FakeBot()
    app = await make_app(bot, TurnRunner(graph, sm), allowed=[TG])

    await app.process_update(text_update(bot, "hapus yang tadi", user_id=TG))
    (question,) = bot.sent()
    assert question["text"].startswith("Hapus pengeluaran: Kopi")
    markup = question["reply_markup"]
    assert [b.callback_data for r in markup.inline_keyboard for b in r] == ["appr:y", "appr:n"]
    assert await _live(sm) == 1  # nothing deleted before the press

    await app.process_update(text_update(bot, "eh sebentar", user_id=TG))  # re-asks, same buttons
    assert bot.sent()[-1]["text"] == question["text"] and bot.sent()[-1].get("reply_markup")
    assert await _live(sm) == 1

    await app.process_update(callback_update(bot, press, user_id=TG, text=question["text"]))
    return bot, app, graph


async def test_approve_button_soft_deletes_and_double_press_is_harmless(
    sm: async_sessionmaker[AsyncSession], scripted_llm: "Callable[[Responder], ScriptedChat]"
) -> None:
    bot, app, _ = await _flow(sm, scripted_llm, "appr:y", "Sudah kuhapus.")
    assert await _live(sm) == 0
    (edit,) = [d for e, d in bot.calls if e == "editMessageText"]
    assert edit["text"].endswith(texts.OUTCOME_APPROVED)
    assert bot.sent()[-1]["text"] == "Sudah kuhapus."

    await app.process_update(callback_update(bot, "appr:y", user_id=TG))  # stale second press
    assert bot.sent()[-1]["text"] == NOTHING_PENDING_REPLY
    assert await _live(sm) == 0


async def test_reject_button_keeps_the_expense(
    sm: async_sessionmaker[AsyncSession], scripted_llm: "Callable[[Responder], ScriptedChat]"
) -> None:
    bot, _, _ = await _flow(sm, scripted_llm, "appr:n", "Oke, tidak jadi.")
    assert await _live(sm) == 1
    (edit,) = [d for e, d in bot.calls if e == "editMessageText"]
    assert edit["text"].endswith(texts.OUTCOME_REJECTED)
    assert bot.sent()[-1]["text"] == "Oke, tidak jadi."
