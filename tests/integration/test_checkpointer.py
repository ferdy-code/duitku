"""T-203: Postgres checkpointer. Test DB only; skips (via `fresh_db`) if Postgres is unreachable."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from alembic import command
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import make_url
from tests.fake_embedder import FakeEmbedder

from duitku.agent import graph as graph_mod
from duitku.agent.checkpointer import cli_thread_id, open_checkpointer, tg_thread_id
from duitku.agent.graph import Graph, build_graph, run_turn

if TYPE_CHECKING:
    from collections.abc import Callable

    from alembic.config import Config
    from tests.conftest import Responder, ScriptedChat

CHECKPOINT_TABLES = {
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
}


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(max_agent_steps=8, short_term_max_tokens=6000)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)


@pytest.fixture
def libpq_url(fresh_db: str) -> str:
    return make_url(fresh_db).set(drivername="postgresql").render_as_string(hide_password=False)


def _cfg(thread_id: str) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": "00000000-0000-0000-0000-000000000001",  # no DB access: slugs are stubbed
            "timezone": "Asia/Jakarta",
        }
    }


async def _no_slugs(_: RunnableConfig) -> Sequence[str]:
    return ["lain-lain"]


def _graph(llm: ScriptedChat, saver: AsyncPostgresSaver) -> Graph:
    return build_graph(
        llm=llm,
        tools=[],
        checkpointer=saver,
        category_slugs=_no_slugs,
        embedder=FakeEmbedder({}, fail=True),  # no memory: keep this test about checkpoints
    )


def _echo(msgs: list[BaseMessage]) -> AIMessage:
    humans = [str(m.content) for m in msgs if isinstance(m, HumanMessage)]
    return AIMessage(f"saw {len(humans)}: {humans[-1]}")


async def _stored(graph: Graph, thread_id: str) -> list[BaseMessage]:
    return (await graph.aget_state(_cfg(thread_id))).values.get("messages", [])


async def test_second_message_sees_first(
    libpq_url: str, scripted_llm: Callable[[Responder], ScriptedChat]
) -> None:
    llm = scripted_llm(_echo)
    async with open_checkpointer(libpq_url) as saver:
        graph = _graph(llm, saver)
        await run_turn(graph, [HumanMessage("satu")], _cfg("cli:a"))
        out = await run_turn(graph, [HumanMessage("dua")], _cfg("cli:a"))
        stored = await _stored(graph, "cli:a")

    # Turn 2's model input held turn 1's human + AI messages (after the system prompt).
    turn2 = llm.seen[1]
    assert isinstance(turn2[0], SystemMessage)
    assert [m.content for m in turn2[1:]] == ["satu", "saw 1: satu", "dua"]
    assert out["messages"][-1].content == "saw 2: dua"
    assert [m.content for m in stored] == ["satu", "saw 1: satu", "dua", "saw 2: dua"]


async def test_threads_are_isolated(
    libpq_url: str, scripted_llm: Callable[[Responder], ScriptedChat]
) -> None:
    llm = scripted_llm(_echo)
    async with open_checkpointer(libpq_url) as saver:
        graph = _graph(llm, saver)
        await run_turn(graph, [HumanMessage("rahasia A")], _cfg("cli:a"))
        await run_turn(graph, [HumanMessage("halo B")], _cfg("cli:b"))
        a, b = await _stored(graph, "cli:a"), await _stored(graph, "cli:b")
        assert await _stored(graph, "cli:never-used") == []

    assert [m.content for m in a] == ["rahasia A", "saw 1: rahasia A"]
    assert [m.content for m in b] == ["halo B", "saw 1: halo B"]
    assert all("rahasia" not in str(m.content) for m in llm.seen[1])  # B's call never saw A


async def test_history_survives_restart(
    libpq_url: str, scripted_llm: Callable[[Responder], ScriptedChat]
) -> None:
    async with open_checkpointer(libpq_url) as saver:
        await run_turn(_graph(scripted_llm(_echo), saver), [HumanMessage("satu")], _cfg("tg:1"))

    # Everything above is closed: pool, saver, graph. A brand-new set must see the history.
    llm2 = scripted_llm(_echo)
    async with open_checkpointer(libpq_url) as saver2:
        graph2 = _graph(llm2, saver2)
        await run_turn(graph2, [HumanMessage("dua")], _cfg("tg:1"))
        stored = await _stored(graph2, "tg:1")

    assert [m.content for m in llm2.seen[0][1:]] == ["satu", "saw 1: satu", "dua"]
    assert [m.content for m in stored] == ["satu", "saw 1: satu", "dua", "saw 2: dua"]


async def test_history_readable_from_another_process(
    libpq_url: str, scripted_llm: Callable[[Responder], ScriptedChat]
) -> None:
    async with open_checkpointer(libpq_url) as saver:
        await run_turn(_graph(scripted_llm(_echo), saver), [HumanMessage("satu")], _cfg("tg:7"))

    code = (
        "import asyncio, os\n"
        "from duitku.agent.checkpointer import open_checkpointer\n"
        "async def main():\n"
        "    async with open_checkpointer(os.environ['CP_URL']) as s:\n"
        "        t = await s.aget_tuple({'configurable': {'thread_id': 'tg:7'}})\n"
        "        print([m.content for m in t.checkpoint['channel_values']['messages']])\n"
        "asyncio.run(main())\n"
    )
    env = {**os.environ, "CP_URL": libpq_url}
    proc = await asyncio.to_thread(
        subprocess.run, [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "['satu', 'saw 1: satu']"


async def test_setup_twice_is_harmless(libpq_url: str) -> None:
    async with open_checkpointer(libpq_url) as saver:
        await saver.setup()
        await saver.setup()
    async with open_checkpointer(libpq_url):  # a "restart" runs setup() on existing tables
        pass


async def test_stored_state_has_no_system_message_and_trim_leaves_it_whole(
    libpq_url: str,
    scripted_llm: Callable[[Responder], ScriptedChat],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_mod,
        "get_settings",
        lambda: SimpleNamespace(max_agent_steps=8, short_term_max_tokens=1),
    )  # tiny budget: the model only sees the last turn
    llm = scripted_llm(_echo)
    async with open_checkpointer(libpq_url) as saver:
        graph = _graph(llm, saver)
        for text in ("satu", "dua", "tiga"):
            await run_turn(graph, [HumanMessage(text)], _cfg("cli:t"))
        stored = await _stored(graph, "cli:t")

    assert [m.content for m in llm.seen[2][1:]] == ["tiga"]  # trimmed view
    assert len(stored) == 6  # stored history is whole
    assert not any(isinstance(m, SystemMessage) for m in stored)


def test_alembic_roundtrip_with_checkpoint_tables_present(
    alembic_cfg: Config, fresh_db: str
) -> None:
    url = make_url(fresh_db).set(drivername="postgresql").render_as_string(hide_password=False)

    async def make_tables() -> None:
        async with open_checkpointer(url):
            pass

    def tables() -> set[str]:
        engine = create_engine(fresh_db)
        with engine.connect() as c:
            names = set(inspect(c).get_table_names())
        engine.dispose()
        return names

    command.upgrade(alembic_cfg, "head")
    asyncio.run(make_tables())
    assert tables() >= CHECKPOINT_TABLES
    command.downgrade(alembic_cfg, "base")  # must not drop or choke on them
    assert tables() >= CHECKPOINT_TABLES
    command.upgrade(alembic_cfg, "head")
    command.check(alembic_cfg)  # autogenerate (via migrations/env.py filter) finds no diff


def test_thread_id_format() -> None:
    assert tg_thread_id(42) == "tg:42"
    assert cli_thread_id("keina") == "cli:keina"
