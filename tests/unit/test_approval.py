"""T-402: approval node, routing and resume semantics with stub tools (no DB, no network).

`add_expense(source="receipt")` is the risky call here because its approval summary needs no DB
lookup; the delete_expense flows live in tests/integration/test_approval_delete.py.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from tests.fake_embedder import FakeEmbedder
from tests.unit.test_graph import assert_paired

from duitku.agent import graph as graph_mod
from duitku.agent.approval import is_approved
from duitku.agent.approval_policy import needs_approval
from duitku.agent.graph import Graph, build_graph, resume_turn, run_turn
from duitku.agent.nodes import STEP_LIMIT_FALLBACK
from duitku.agent.routing import route_after_agent
from duitku.agent.state import AgentState
from duitku.tools.base import ok

if TYPE_CHECKING:
    from tests.conftest import Responder, ScriptedChat

type Llm = Callable[[Responder], ScriptedChat]
UID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
MAX_STEPS = 3
ran: list[str] = []  # tool bodies that actually executed (by tool_call amount/x)


async def _add_expense(amount: str, category: str = "makan-minum", source: str = "text") -> Any:
    ran.append(f"add:{amount}")
    return ok({"amount": amount, "source": source})


async def _echo(x: str = "") -> Any:
    ran.append(f"echo:{x}")
    return ok({"x": x})


TOOLS = [
    StructuredTool.from_function(coroutine=_add_expense, name="add_expense", description="add"),
    StructuredTool.from_function(coroutine=_echo, name="echo", description="echo"),
]


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    ran.clear()
    fake = SimpleNamespace(max_agent_steps=MAX_STEPS, short_term_max_tokens=6000)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: fake)


async def _slugs(config: RunnableConfig) -> list[str]:
    return ["makan-minum"]


def _graph(llm: ScriptedChat) -> Graph:
    return build_graph(
        llm=llm,
        tools=TOOLS,
        checkpointer=InMemorySaver(),
        category_slugs=_slugs,
        embedder=FakeEmbedder({}, fail=True),
    )


def _config(thread: str = "t") -> RunnableConfig:
    return {"configurable": {"user_id": UID, "timezone": "Asia/Jakarta", "thread_id": thread}}


def receipt(i: int, amount: str = "10rb") -> dict[str, Any]:
    return {
        "id": f"r{i}",
        "name": "add_expense",
        "args": {"amount": amount, "category": "makan-minum", "source": "receipt"},
    }


def echo(i: int) -> dict[str, Any]:
    return {"id": f"e{i}", "name": "echo", "args": {"x": str(i)}}


def scripted(*msgs: AIMessage) -> Callable[[Any], AIMessage]:
    it = iter(msgs)
    return lambda _: next(it)


def answer(decision: str) -> dict[str, Any]:
    return {"decision": decision, "reason": None}


def codes(messages: list[AnyMessage]) -> dict[str, str | None]:
    """tool_call_id -> error code (None when ok)."""
    out: dict[str, str | None] = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            body = json.loads(str(m.content))
            out[m.tool_call_id] = None if body["ok"] else body["error"]["code"]
    return out


def pending(out: dict[str, Any]) -> list[dict[str, Any]]:
    return [i.value for i in out.get("__interrupt__", [])]


# --- policy & routing ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("delete_expense", {"expense_id": "x"}, True),
        ("add_expense", {"amount": "1rb", "source": "receipt"}, True),
        ("add_expense", {"amount": "1rb", "source": "text"}, False),
        ("add_expense", {"amount": "1rb"}, False),
        ("update_expense", {"expense_id": "x"}, False),
        ("query_expenses", {}, False),
    ],
)
def test_needs_approval(name: str, args: dict[str, Any], expected: bool) -> None:
    assert needs_approval({"id": "1", "name": name, "args": args, "type": "tool_call"}) is expected


def _state(*msgs: AnyMessage) -> AgentState:
    return AgentState(messages=list(msgs), recalled_memories=[], rejected_call_ids=[])


def test_routing_picks_approval_and_cap_wins() -> None:
    risky = _state(HumanMessage("x"), AIMessage("", tool_calls=[echo(1), receipt(2)]))  # type: ignore[list-item]
    plain = _state(HumanMessage("x"), AIMessage("", tool_calls=[echo(1)]))  # type: ignore[list-item]
    assert route_after_agent(risky, max_steps=3) == "approval"
    assert route_after_agent(plain, max_steps=3) == "tools"
    assert route_after_agent(risky, max_steps=1) == "limit"  # cap check stays first
    assert route_after_agent(_state(HumanMessage("x")), max_steps=3) == "__end__"


@pytest.mark.parametrize(
    ("value", "approved"),
    [
        ({"decision": "approve", "reason": None}, True),
        ({"decision": "approve"}, True),
        ({"decision": "reject"}, False),
        ({"decision": "APPROVE"}, False),
        ({"decision": ["approve"]}, False),
        ({"approve": True}, False),
        ({}, False),
        ("approve", False),
        (True, False),
        (None, False),
    ],
)
def test_is_approved_is_strict(value: object, approved: bool) -> None:
    assert is_approved(value) is approved


# --- graph flows ---------------------------------------------------------------------------


async def test_pause_then_approve_runs_the_tool(scripted_llm: Llm) -> None:
    graph = _graph(scripted_llm(scripted(AIMessage("", tool_calls=[receipt(1)]), AIMessage("ok"))))
    cfg = _config()
    out = await run_turn(graph, [HumanMessage("struk")], cfg)

    (p,) = pending(out)
    assert p["tool"] == "add_expense" and p["tool_call_id"] == "r1"
    assert p["human_summary"].startswith("Catat pengeluaran dari struk: Rp10.000")
    assert ran == []  # nothing ran before the user answered
    assert (await graph.aget_state(cfg)).next == ("approval",)

    out = await resume_turn(graph, answer("approve"), cfg)
    assert "__interrupt__" not in out and ran == ["add:10rb"]
    assert codes(out["messages"]) == {"r1": None}
    assert out["messages"][-1].content == "ok"
    assert_paired(out["messages"])


async def test_reject_never_executes_and_agent_still_answers(scripted_llm: Llm) -> None:
    llm = scripted_llm(scripted(AIMessage("", tool_calls=[receipt(1)]), AIMessage("Dibatalkan.")))
    graph = _graph(llm)
    cfg = _config()
    await run_turn(graph, [HumanMessage("struk")], cfg)
    out = await resume_turn(graph, answer("reject"), cfg)

    assert ran == []
    assert codes(out["messages"]) == {"r1": "REJECTED_BY_USER"}
    assert out["messages"][-1].content == "Dibatalkan."
    seen_tool = [m for m in llm.seen[-1] if isinstance(m, ToolMessage)]
    assert "REJECTED_BY_USER" in str(seen_tool[0].content)  # the model saw the refusal
    assert_paired(out["messages"])


# None and {} would crash / be ignored by langgraph 1.2.14 itself; resume_turn normalises them.
@pytest.mark.parametrize("bad", [None, "approve", {"decision": "yes"}, {}, 42, {"decision": None}])
async def test_malformed_resume_value_is_a_reject(scripted_llm: Llm, bad: Any) -> None:
    graph = _graph(scripted_llm(scripted(AIMessage("", tool_calls=[receipt(1)]), AIMessage("ok"))))
    cfg = _config()
    await run_turn(graph, [HumanMessage("struk")], cfg)
    out = await resume_turn(graph, bad, cfg)
    assert ran == [] and codes(out["messages"]) == {"r1": "REJECTED_BY_USER"}


async def test_mixed_message_runs_safe_call_and_asks_only_for_the_risky_one(
    scripted_llm: Llm,
) -> None:
    graph = _graph(
        scripted_llm(
            scripted(AIMessage("", tool_calls=[echo(1), receipt(2), echo(3)]), AIMessage("ok"))
        )
    )
    cfg = _config()
    out = await run_turn(graph, [HumanMessage("x")], cfg)
    assert [p["tool_call_id"] for p in pending(out)] == ["r2"]
    assert ran == []  # even the safe calls wait: the whole AIMessage executes after approval
    out = await resume_turn(graph, answer("reject"), cfg)
    assert ran == ["echo:1", "echo:3"]
    assert codes(out["messages"]) == {"e1": None, "r2": "REJECTED_BY_USER", "e3": None}
    assert_paired(out["messages"])


async def test_two_risky_calls_are_asked_in_order_and_answered_independently(
    scripted_llm: Llm,
) -> None:
    graph = _graph(
        scripted_llm(
            scripted(
                AIMessage("", tool_calls=[receipt(1, "10rb"), receipt(2, "20rb")]),
                AIMessage("ok"),
            )
        )
    )
    cfg = _config()
    out = await run_turn(graph, [HumanMessage("x")], cfg)
    assert [p["tool_call_id"] for p in pending(out)] == ["r1"]

    out = await resume_turn(graph, answer("approve"), cfg)  # answers r1 only
    assert [p["tool_call_id"] for p in pending(out)] == ["r2"]
    assert ran == []

    out = await resume_turn(graph, answer("reject"), cfg)  # answers r2
    assert "__interrupt__" not in out
    assert ran == ["add:10rb"]
    assert codes(out["messages"]) == {"r1": None, "r2": "REJECTED_BY_USER"}
    assert_paired(out["messages"])


async def test_step_cap_still_ends_the_loop_with_approval_in_it(scripted_llm: Llm) -> None:
    n = 0

    def always_risky(_: Any) -> AIMessage:
        nonlocal n
        n += 1
        return AIMessage("", tool_calls=[receipt(n)])

    llm = scripted_llm(always_risky)
    graph = _graph(llm)
    cfg = _config()
    out = await run_turn(graph, [HumanMessage("x")], cfg)
    asked = 0
    while pending(out):
        asked += 1
        assert asked <= MAX_STEPS
        out = await resume_turn(graph, answer("approve"), cfg)

    assert len(llm.seen) == MAX_STEPS
    assert asked == MAX_STEPS - 1  # the capped step is refused without asking
    assert out["messages"][-1].content == STEP_LIMIT_FALLBACK
    assert codes(out["messages"])[f"r{MAX_STEPS}"] == "STEP_LIMIT"
    assert_paired(out["messages"])


async def test_rejected_ids_are_cleared_after_the_tools_node(scripted_llm: Llm) -> None:
    graph = _graph(scripted_llm(scripted(AIMessage("", tool_calls=[receipt(1)]), AIMessage("ok"))))
    cfg = _config()
    await run_turn(graph, [HumanMessage("x")], cfg)
    await resume_turn(graph, answer("reject"), cfg)
    assert (await graph.aget_state(cfg)).values["rejected_call_ids"] == []
