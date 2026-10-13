"""T-406: agent_runs rows written by the runner (real Postgres, scripted model, no network)."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.conftest import Otel
from tests.fake_embedder import FakeEmbedder

from duitku.agent import graph as graph_mod
from duitku.agent.graph import build_graph
from duitku.agent.nodes import repo_slug_loader
from duitku.db.repositories import audit
from duitku.interfaces import run_log
from duitku.interfaces.runner import ERROR_REPLY, TurnRunner
from duitku.tools.registry import build_tools

if TYPE_CHECKING:
    from tests.conftest import Responder, ScriptedChat

SM = async_sessionmaker[AsyncSession]
type Llm = Callable[[Responder], ScriptedChat]


def usage(i: int, o: int) -> dict[str, int]:
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


@pytest.fixture(autouse=True)
def _graph_settings(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    fake = SimpleNamespace(max_agent_steps=8, short_term_max_tokens=6000)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)


def _runner(sm: SM, llm: ScriptedChat) -> TurnRunner:
    graph = build_graph(
        llm=llm,
        tools=build_tools(sm),
        checkpointer=InMemorySaver(),
        category_slugs=repo_slug_loader(sm),
        sessionmaker=sm,
        embedder=FakeEmbedder({}, fail=True),
    )
    return TurnRunner(graph, sm)


def _add_call(i: int) -> dict[str, Any]:
    args = {"amount": "35rb", "category": "makan-minum"}
    return {"id": f"a{i}", "name": "add_expense", "args": args}


async def _runs(sm: SM, user_id: Any) -> list[Any]:
    async with sm() as s:
        return list(reversed(await audit.list_agent_runs(s, user_id)))  # oldest first


async def test_row_has_tokens_duration_steps_and_trace_id(
    sm: SM, scripted_llm: Llm, otel: Otel
) -> None:
    script = iter(
        [
            AIMessage("", tool_calls=[_add_call(1)], usage_metadata=usage(100, 7)),
            AIMessage("Tercatat.", usage_metadata=usage(50, 3)),
        ]
    )

    def respond(_: list[BaseMessage]) -> AIMessage:
        time.sleep(0.02)
        return next(script)

    runner = _runner(sm, scripted_llm(respond))
    user = await runner.resolve_user(-406)
    res = await runner.run("cli:t", user, "makan 35rb")
    assert not res.failed

    (row,) = await _runs(sm, user.user_id)
    assert (row.status, row.steps, row.thread_id) == ("ok", 2, "cli:t")
    assert (row.input_tokens, row.output_tokens) == (150, 10)
    assert row.duration_ms >= 40 and row.model
    (run_span,) = [s for s in otel.spans.get_finished_spans() if s.name == "agent.run"]
    assert run_span.attributes and run_span.attributes["duitku.steps"] == 2


async def test_run_plus_resume_counts_each_model_call_once(
    sm: SM, scripted_llm: Llm, otel: Otel
) -> None:
    # Turn 1 adds an expense (id read back from the DB), turn 2 pauses on delete_expense and the
    # resume finishes it. resume()'s trace repeats the pre-pause steps; the rows must not.
    eid: list[str] = []
    script = iter(
        [
            AIMessage("", tool_calls=[_add_call(1)], usage_metadata=usage(10, 1)),
            AIMessage("ok", usage_metadata=usage(11, 1)),
        ]
    )

    def respond(msgs: list[BaseMessage]) -> AIMessage:
        if not eid:
            return next(script)
        if not any(isinstance(m, AIMessage) and m.tool_calls for m in msgs[-3:]):
            call = {"id": "d1", "name": "delete_expense", "args": {"expense_id": eid[0]}}
            return AIMessage("", tool_calls=[call], usage_metadata=usage(100, 10))
        return AIMessage("Dihapus.", usage_metadata=usage(20, 5))

    llm = scripted_llm(respond)
    runner = _runner(sm, llm)
    user = await runner.resolve_user(-407)
    await runner.run("cli:t", user, "makan 35rb")
    async with sm() as s:
        eid.append(str((await s.execute(text("SELECT id FROM expenses"))).scalar_one()))

    paused = await runner.run("cli:t", user, "hapus tadi")
    assert paused.pending is not None
    done = await runner.resume("cli:t", user, "approve")
    assert done.text == "Dihapus." and len(done.trace) == 2  # whole turn, incl. the pre-pause step

    rows = await _runs(sm, user.user_id)
    assert [r.status for r in rows] == ["ok", "interrupted", "ok"]
    assert [(r.steps, r.input_tokens, r.output_tokens) for r in rows] == [
        (2, 21, 2),
        (1, 100, 10),  # segment up to the pause only
        (1, 20, 5),  # resume segment: its own model call only
    ]
    assert sum(r.steps for r in rows) == len(llm.seen)  # = real model calls, none double counted
    assert otel.metric_total("llm_tokens_total", direction="input") == 21 + 100 + 20
    assert [
        s.attributes["duitku.run.kind"]  # type: ignore[index]
        for s in otel.spans.get_finished_spans()
        if s.name == "agent.run"
    ] == ["run", "run", "resume"]


async def test_failed_turn_gets_an_error_row_and_log_with_trace_id(
    sm: SM, scripted_llm: Llm, otel: Otel, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_: list[BaseMessage]) -> AIMessage:
        raise ConnectionError("secret-host.internal refused")

    runner = _runner(sm, scripted_llm(boom))
    user = await runner.resolve_user(-408)
    with caplog.at_level(logging.ERROR):
        res = await runner.run("cli:t", user, "halo")
    assert res.failed and res.text == ERROR_REPLY
    (row,) = await _runs(sm, user.user_id)
    assert (row.status, row.steps, row.input_tokens) == ("error", 1, None)
    rec = next(r for r in caplog.records if r.message.startswith("turn failed"))
    assert re.fullmatch(r"[0-9a-f]{32}", getattr(rec, "trace_id"))  # noqa: B009
    assert "secret-host" not in caplog.text


async def test_recording_failure_never_fails_the_turn(
    sm: SM,
    scripted_llm: Llm,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def broken(*_a: Any, **_k: Any) -> None:
        raise OperationalError("INSERT secret-sql", {}, Exception("secret-conn-detail"))

    monkeypatch.setattr(run_log.audit, "record_agent_run", broken)
    runner = _runner(sm, scripted_llm(lambda _: AIMessage("hai")))
    user = await runner.resolve_user(-409)
    with caplog.at_level(logging.ERROR):
        res = await runner.run("cli:t", user, "halo")
    assert res.text == "hai" and not res.failed
    assert "OperationalError" in caplog.text and "secret" not in caplog.text
    assert await _runs(sm, user.user_id) == []
