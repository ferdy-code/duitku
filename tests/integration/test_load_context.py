"""T-305: `load_context` through the real graph + real DB, FakeEmbedder (exact cosines, no model).

Calibrated recall threshold in effect: 0.835 (m95 recalled, m80 not).
"""

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.fake_embedder import FakeEmbedder, unit

from duitku.agent.context import make_load_context_node
from duitku.agent.graph import build_graph, run_turn
from duitku.agent.state import AgentState
from duitku.db.models import Memory
from duitku.db.repositories import users
from duitku.memory import store
from duitku.tools.registry import build_tools

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import Responder, ScriptedChat

SM = async_sessionmaker[AsyncSession]


def _emb() -> FakeEmbedder:
    return FakeEmbedder(
        {
            "q": unit(1.0),
            "hello": unit(0.0, axis=9),
            "m95": unit(0.95, axis=3),  # cos to q: recalled
            "m80": unit(0.80, axis=4),  # below 0.835: not recalled
        }
    )


def _cfg(uid: uuid.UUID, thread: str) -> RunnableConfig:
    return {"configurable": {"user_id": str(uid), "timezone": "Asia/Jakarta", "thread_id": thread}}


async def _user(sm: SM, tg: int) -> uuid.UUID:
    async with sm() as s:
        user, _ = await users.get_or_create_user(s, tg)
        await s.commit()
        return user.id


async def _seed(sm: SM, emb: FakeEmbedder, uid: uuid.UUID, *contents: str) -> None:
    async with sm() as s:
        for c in contents:
            await store.save(s, uid, "fact", c, emb, dedupe_threshold=0.985)
        await s.commit()


def _system_text(llm: "ScriptedChat") -> str:
    first = llm.seen[-1][0]
    assert isinstance(first, SystemMessage)
    return str(first.content)


def _graph(sm: SM, llm: "ScriptedChat", emb: FakeEmbedder, **kw: Any) -> Any:
    return build_graph(
        llm=llm,
        tools=build_tools(sm, emb),
        checkpointer=InMemorySaver(),
        sessionmaker=sm,
        embedder=emb,
        **kw,
    )


async def test_memory_saved_in_thread_a_is_recalled_in_thread_b_same_user_only(
    sm: SM, scripted_llm: "Callable[[Responder], ScriptedChat]"
) -> None:
    emb = _emb()
    me, other = await _user(sm, 910000305), await _user(sm, 910000306)
    script = iter(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "s1",
                        "name": "save_memory",
                        "args": {"content": "m95", "kind": "fact"},
                    }
                ],
            ),
            AIMessage("oke, tersimpan"),
        ]
    )
    llm = scripted_llm(lambda _: next(script, AIMessage("ok")))
    graph = _graph(sm, llm, emb)

    await run_turn(graph, [HumanMessage("hello")], _cfg(me, "thread-a"))
    assert "m95" not in _system_text(llm).split("<memories>")[1]  # nothing recalled yet

    out = await run_turn(graph, [HumanMessage("q")], _cfg(me, "thread-b"))
    assert out["recalled_memories"] == ["m95"]
    assert "- m95" in _system_text(llm)

    out = await run_turn(graph, [HumanMessage("q")], _cfg(other, "thread-c"))
    assert out["recalled_memories"] == []
    assert "m95" not in _system_text(llm)


async def test_embedding_failure_does_not_fail_the_run(
    sm: SM, scripted_llm: "Callable[[Responder], ScriptedChat]", caplog: pytest.LogCaptureFixture
) -> None:
    emb = FakeEmbedder({}, fail=True)
    uid = await _user(sm, 910000307)
    llm = scripted_llm(lambda _: AIMessage("halo"))
    with caplog.at_level("WARNING", logger="duitku.agent.context"):
        out = await run_turn(_graph(sm, llm, emb), [HumanMessage("rahasia123")], _cfg(uid, "t"))
    assert out["messages"][-1].content == "halo"
    assert out["recalled_memories"] == []
    assert "EmbeddingError" in caplog.text and "rahasia123" not in caplog.text


class _HangingEmbedder(FakeEmbedder):
    async def embed_query(self, text: str) -> list[float]:
        await asyncio.sleep(30)
        return await super().embed_query(text)


async def test_hanging_embedder_times_out_and_run_continues(sm: SM) -> None:
    node = make_load_context_node(sm, _HangingEmbedder(_emb().table), timeout_seconds=0.05)
    uid = await _user(sm, 910000308)
    state = AgentState(
        messages=[HumanMessage("q")], recalled_memories=["old"], rejected_call_ids=[]
    )
    out = await asyncio.wait_for(node(state, _cfg(uid, "t")), 5)
    assert out == {"recalled_memories": [], "rejected_call_ids": []}


async def test_rejected_call_ids_reset_on_success_and_failure(sm: SM) -> None:
    uid = await _user(sm, 910000309)
    state = AgentState(messages=[HumanMessage("q")], recalled_memories=[], rejected_call_ids=["x"])
    ok = await make_load_context_node(sm, _emb())(state, _cfg(uid, "t"))
    bad = await make_load_context_node(sm, FakeEmbedder({}, fail=True))(state, _cfg(uid, "t"))
    assert ok["rejected_call_ids"] == [] and bad["rejected_call_ids"] == []


@pytest.mark.parametrize(
    "messages",
    [[], [AIMessage("hai")], [HumanMessage("   ")], [HumanMessage([{"type": "image_url"}])]],  # pyright: ignore[reportArgumentType]
)
async def test_no_usable_human_text_means_no_embed_call(sm: SM, messages: list[AnyMessage]) -> None:
    emb = _emb()
    uid = await _user(sm, 910000310)
    state = AgentState(messages=messages, recalled_memories=[], rejected_call_ids=[])
    out = await make_load_context_node(sm, emb)(state, _cfg(uid, "t"))
    assert out == {"recalled_memories": [], "rejected_call_ids": []}
    assert emb.calls == []


async def test_uses_last_human_text_and_joins_text_parts(sm: SM) -> None:
    emb = _emb()
    uid = await _user(sm, 910000311)
    parts: list[str | dict[str, Any]] = [{"type": "text", "text": "q"}, {"type": "image_url"}]
    state = AgentState(
        messages=[HumanMessage("hello"), AIMessage("x"), HumanMessage(parts)],  # pyright: ignore[reportArgumentType]
        recalled_memories=[],
        rejected_call_ids=[],
    )
    await make_load_context_node(sm, emb)(state, _cfg(uid, "t"))
    assert emb.calls == [("query", "q")]


async def test_recall_bumps_recall_count_only_for_returned_memories(sm: SM) -> None:
    emb = _emb()
    uid = await _user(sm, 910000312)
    await _seed(sm, emb, uid, "m95", "m80")
    state = AgentState(messages=[HumanMessage("q")], recalled_memories=[], rejected_call_ids=[])
    out = await make_load_context_node(sm, emb)(state, _cfg(uid, "t"))
    assert out["recalled_memories"] == ["m95"]  # m80 (cos 0.80 < 0.835) is filtered out
    async with sm() as s:
        rows = {m.content: m for m in (await s.execute(select(Memory))).scalars()}
    assert rows["m95"].recall_count == 1 and rows["m95"].last_recalled_at is not None
    assert rows["m80"].recall_count == 0
