"""T-204: history trimming invariants (pure, no network)."""

import copy
import random

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage

from duitku.agent.trimming import trim_history

BUDGETS = [1, 20, 60, 150, 400, 10_000]


def _history(rng: random.Random, turns: int) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    n = 0
    for t in range(turns):
        msgs.append(HumanMessage(f"user turn {t} " + "x" * rng.randint(0, 200)))
        for _ in range(rng.randint(0, 2)):  # tool loop steps
            ids = [f"c{n + k}" for k in range(rng.randint(1, 3))]
            n += len(ids)
            msgs.append(
                AIMessage(
                    "", tool_calls=[{"id": i, "name": "t", "args": {"q": "y" * 30}} for i in ids]
                )
            )
            msgs += [
                ToolMessage('{"ok":true}' + "z" * rng.randint(0, 150), tool_call_id=i) for i in ids
            ]
        msgs.append(AIMessage("answer " + "a" * rng.randint(0, 100)))
    return msgs


def _check_wellformed(view: list[BaseMessage]) -> None:
    assert not isinstance(view[0], ToolMessage)
    for i, m in enumerate(view):
        if isinstance(m, AIMessage) and m.tool_calls:
            want = {c["id"] for c in m.tool_calls}
            got = [
                x.tool_call_id
                for x in view[i + 1 : i + 1 + len(want)]
                if isinstance(x, ToolMessage)
            ]
            assert set(got) == want and len(got) == len(want)
        if isinstance(m, ToolMessage):  # preceded (through siblings) by its AIMessage
            j = i
            while isinstance(view[j], ToolMessage):
                j -= 1
            prev = view[j]
            assert isinstance(prev, AIMessage)
            assert m.tool_call_id in {c["id"] for c in prev.tool_calls}


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("budget", BUDGETS)
def test_invariants_property(seed: int, budget: int) -> None:
    msgs = _history(random.Random(seed), random.Random(seed).randint(1, 8))
    snapshot = copy.deepcopy(msgs)
    original = list(msgs)

    view = trim_history(msgs, budget)

    assert msgs == snapshot and msgs == original  # input list and messages unchanged
    assert view is not msgs
    _check_wellformed(view)
    last_human = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
    assert view[-len(msgs[last_human:]) :] == msgs[last_human:]  # latest human + tail kept
    assert view == msgs[len(msgs) - len(view) :]  # contiguous suffix
    if len(view) > len(msgs[last_human:]):  # beyond the mandatory part, budget is respected
        from langchain_core.messages.utils import count_tokens_approximately

        assert count_tokens_approximately(view) <= budget


def test_short_and_empty_pass_through() -> None:
    assert trim_history([], 100) == []
    msgs: list[BaseMessage] = [HumanMessage("hi"), AIMessage("halo")]
    assert trim_history(msgs, 6000) == msgs


def test_multi_tool_calls_kept_together() -> None:
    calls: list[ToolCall] = [ToolCall(id=f"c{i}", name="t", args={}) for i in range(3)]
    msgs: list[BaseMessage] = [
        HumanMessage("old " * 200),
        AIMessage("old answer " * 50),
        HumanMessage("now"),
        AIMessage("", tool_calls=calls),
        *[ToolMessage("{}", tool_call_id=c["id"]) for c in calls],
        AIMessage("done"),
    ]
    view = trim_history(msgs, 60)
    assert view == msgs[2:]


def test_latest_human_kept_over_budget() -> None:
    msgs: list[BaseMessage] = [HumanMessage("a"), AIMessage("b"), HumanMessage("c" * 4000)]
    assert trim_history(msgs, 5) == msgs[2:]


def test_drops_whole_old_group_not_half() -> None:
    call: list[ToolCall] = [ToolCall(id="c1", name="t", args={})]
    msgs: list[BaseMessage] = [
        HumanMessage("q1"),
        AIMessage("", tool_calls=call),
        ToolMessage("r" * 400, tool_call_id="c1"),
        AIMessage("a1"),
        HumanMessage("q2"),
        AIMessage("a2"),
    ]
    view = trim_history(msgs, 30)
    assert view == msgs[4:]
