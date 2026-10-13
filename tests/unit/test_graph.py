"""T-202: agent <-> tools loop with a fake model and fake tools (no network, no DB)."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from tests.fake_embedder import FakeEmbedder

from duitku.agent import graph as graph_mod
from duitku.agent.graph import build_graph, run_turn
from duitku.agent.nodes import STEP_LIMIT_FALLBACK, repo_slug_loader
from duitku.agent.prompts import SYSTEM_PROMPT_TEMPLATE
from duitku.domain.categories import DEFAULT_CATEGORIES
from duitku.tools.base import ok

if TYPE_CHECKING:
    from tests.conftest import Responder, ScriptedChat

UID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
CONFIG: RunnableConfig = {"configurable": {"user_id": UID, "timezone": "Asia/Jakarta"}}
type Llm = Callable[[Responder], ScriptedChat]
MAX_STEPS = 3
calls: list[str] = []  # names of tool bodies that actually ran


async def _echo(x: str = "") -> dict[str, Any]:
    calls.append("echo")
    return ok({"x": x})


async def _boom(x: str = "") -> dict[str, Any]:
    calls.append("boom")
    raise RuntimeError("secret internal detail")


async def _whoami(config: RunnableConfig, x: str = "") -> dict[str, Any]:
    from duitku.tools.base import get_context

    ctx = get_context(config)
    return ok({"user_id": str(ctx.user_id), "tz": ctx.timezone.key})


def _tool(fn: Callable[..., Any], name: str) -> StructuredTool:
    return StructuredTool.from_function(coroutine=fn, name=name, description=name)


TOOLS = [_tool(_echo, "echo"), _tool(_boom, "boom"), _tool(_whoami, "whoami")]


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls.clear()
    fake = SimpleNamespace(max_agent_steps=MAX_STEPS, short_term_max_tokens=6000)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)


async def _slugs(config: RunnableConfig) -> list[str]:
    return ["makan-minum", "transport"]


def make_graph(llm: ScriptedChat, **kw: Any) -> Any:
    # Failing embedder: load_context degrades to no memories, never touches a model or the DB.
    kw.setdefault("embedder", FakeEmbedder({}, fail=True))
    return build_graph(llm=llm, tools=TOOLS, category_slugs=_slugs, **kw)


def call(name: str, i: int, **args: Any) -> dict[str, Any]:
    return {"id": f"c{i}", "name": name, "args": args or {"x": "hi"}}


def assert_paired(messages: list[AnyMessage]) -> None:
    """Every tool_call id (valid or not) has exactly one ToolMessage; nothing dangles."""
    for i, m in enumerate(messages):
        if not isinstance(m, AIMessage):
            continue
        ids = [c["id"] or "" for c in (*m.tool_calls, *m.invalid_tool_calls)]
        answers = [t.tool_call_id for t in messages[i + 1 :] if isinstance(t, ToolMessage)]
        for cid in ids:
            assert answers.count(cid) == 1, f"{cid} answered {answers.count(cid)}x"
    all_calls = {
        c["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for c in (*m.tool_calls, *m.invalid_tool_calls)
    }
    for t in messages:
        if isinstance(t, ToolMessage):
            assert t.tool_call_id in all_calls  # no orphan results


async def test_step_cap_stops_loop_with_polite_fallback(scripted_llm: Llm) -> None:
    n = 0

    def always_tool(_: Any) -> AIMessage:
        nonlocal n
        n += 1
        return AIMessage("", tool_calls=[call("echo", n)])

    llm = scripted_llm(always_tool)
    out = await run_turn(make_graph(llm), [HumanMessage("halo")], CONFIG)
    msgs = out["messages"]
    assert len(llm.seen) == MAX_STEPS  # exactly the cap in model calls
    assert calls == ["echo"] * (MAX_STEPS - 1)  # the capped step's call is refused, not run
    assert msgs[-1].content == STEP_LIMIT_FALLBACK
    assert isinstance(msgs[-1], AIMessage) and not msgs[-1].tool_calls
    refused = json.loads(str(msgs[-2].content))
    assert refused["error"]["code"] == "STEP_LIMIT"
    assert_paired(msgs)


async def test_no_tool_calls_ends_immediately(scripted_llm: Llm) -> None:
    llm = scripted_llm(lambda _: AIMessage("Halo!"))
    out = await run_turn(make_graph(llm), [HumanMessage("halo")], CONFIG)
    assert [type(m) for m in out["messages"]] == [HumanMessage, AIMessage]


def _scenario(rng: random.Random) -> list[list[dict[str, Any]]]:
    """1-2 steps of 1-4 calls (name 'nope' = unknown tool, 'bad' = invalid JSON args)."""
    n = 0
    steps: list[list[dict[str, Any]]] = []
    for _ in range(rng.randint(1, 2)):
        step: list[dict[str, Any]] = []
        for _ in range(rng.randint(1, 4)):
            n += 1
            step.append(call(rng.choice(["echo", "boom", "nope", "whoami", "bad"]), n))
        steps.append(step)
    return steps


@pytest.mark.parametrize("seed", range(40))
async def test_every_tool_call_id_gets_exactly_one_tool_message(
    scripted_llm: Llm, seed: int
) -> None:
    steps = _scenario(random.Random(seed))
    step_ids = iter(range(len(steps)))

    def respond(_: Any) -> AIMessage:
        i = next(step_ids, None)
        if i is None:
            return AIMessage("selesai")
        valid = [c for c in steps[i] if c["name"] != "bad"]
        invalid = [
            {"id": c["id"], "name": "echo", "args": "{oops", "error": "bad json", "type": None}
            for c in steps[i]
            if c["name"] == "bad"
        ]
        return AIMessage("", tool_calls=valid, invalid_tool_calls=invalid)  # type: ignore[arg-type]

    out = await run_turn(make_graph(scripted_llm(respond)), [HumanMessage("x")], CONFIG)
    assert_paired(out["messages"])
    assert out["messages"][-1].content == "selesai"


async def test_tool_errors_are_envelopes_without_exception_text(scripted_llm: Llm) -> None:
    script = iter(
        [
            AIMessage("", tool_calls=[call("boom", 1), call("nope", 2)]),
            AIMessage("ok"),
        ]
    )
    out = await run_turn(
        make_graph(scripted_llm(lambda _: next(script))), [HumanMessage("x")], CONFIG
    )
    boom, nope = (json.loads(str(m.content)) for m in out["messages"] if isinstance(m, ToolMessage))
    assert boom["error"]["code"] == "INTERNAL_ERROR"
    assert "secret" not in json.dumps(boom)
    assert nope["error"]["code"] == "UNKNOWN_TOOL"


async def test_model_view_is_trimmed_but_stored_state_is_whole(
    scripted_llm: Llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = SimpleNamespace(max_agent_steps=MAX_STEPS, short_term_max_tokens=50)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)
    history: list[AnyMessage] = []
    for t in range(6):
        history += [
            HumanMessage(f"old question {t} " + "x" * 200),
            AIMessage("old answer " + "y" * 200),
        ]
    history.append(HumanMessage("pertanyaan baru"))
    llm = scripted_llm(lambda _: AIMessage("jawab"))
    out = await run_turn(make_graph(llm), history, CONFIG)
    shown = llm.seen[0]
    assert isinstance(shown[0], SystemMessage)
    assert len(shown) < len(history) + 1  # older turns dropped from the model view
    assert shown[1].type == "human" and shown[-1].content == "pertanyaan baru"
    assert len(out["messages"]) == len(history) + 1  # stored state keeps everything + reply
    assert not any(isinstance(m, SystemMessage) for m in out["messages"])


async def test_system_prompt_has_slugs_and_user_timezone(scripted_llm: Llm) -> None:
    llm = scripted_llm(lambda _: AIMessage("hi"))
    await run_turn(make_graph(llm), [HumanMessage("halo")], CONFIG)
    system = llm.seen[0][0].content
    assert isinstance(system, str)
    assert "makan-minum, transport" in system
    assert "timezone Asia/Jakarta" in system
    assert "never guess an id" in SYSTEM_PROMPT_TEMPLATE


async def test_user_id_and_timezone_come_only_from_config(scripted_llm: Llm) -> None:
    evil = call("whoami", 1, x="a", user_id="evil-user", timezone="Mars/Base")
    script = iter([AIMessage("", tool_calls=[evil]), AIMessage("ok")])
    out = await run_turn(
        make_graph(scripted_llm(lambda _: next(script))), [HumanMessage("x")], CONFIG
    )
    (tm,) = (m for m in out["messages"] if isinstance(m, ToolMessage))
    assert json.loads(str(tm.content))["data"] == {"user_id": UID, "tz": "Asia/Jakarta"}


async def test_recursion_limit_backstop_returns_fallback(scripted_llm: Llm) -> None:
    n = 0

    def always_tool(_: Any) -> AIMessage:
        nonlocal n
        n += 1
        return AIMessage("", tool_calls=[call("echo", n)])

    cfg: RunnableConfig = {**CONFIG, "recursion_limit": 2}  # lower than the routing cap
    out = await run_turn(make_graph(scripted_llm(always_tool)), [HumanMessage("x")], cfg)
    assert out["messages"][-1].content == STEP_LIMIT_FALLBACK


async def test_slug_loader_falls_back_to_defaults_on_failure() -> None:
    class Broken:
        def __call__(self) -> Any:
            raise ConnectionError("db down")

    load = repo_slug_loader(Broken())  # type: ignore[arg-type]
    assert list(await load(CONFIG)) == [s for s, _ in DEFAULT_CATEGORIES]
    assert list(await load({})) == [s for s, _ in DEFAULT_CATEGORIES]  # no user_id either


def test_import_boundaries() -> None:
    from pathlib import Path

    src = Path(graph_mod.__file__).resolve().parents[1]
    for f in (src / "agent").glob("*.py"):
        assert "telegram" not in f.read_text().lower().replace("docs/spec", ""), f
    for f in (src / "tools").glob("*.py"):
        assert "duitku.agent" not in f.read_text(), f
