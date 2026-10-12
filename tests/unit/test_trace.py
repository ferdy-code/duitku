"""T-205: trace extraction looks only at the current turn."""

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from duitku.interfaces.trace import ModelStep, extract_trace, final_text, format_trace


def _history() -> list[AnyMessage]:
    usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    call = {"id": "c1", "name": "add_expense", "args": {"amount": "40rb"}}
    return [
        HumanMessage("turn 1"),
        AIMessage("", tool_calls=[{"id": "old", "name": "old_tool", "args": {}}]),
        ToolMessage('{"ok":true}', tool_call_id="old"),
        AIMessage("jawaban lama"),
        HumanMessage("turn 2"),
        AIMessage("", tool_calls=[call], usage_metadata=usage),
        ToolMessage('{"ok":true,"data":{"id":1}}', tool_call_id="c1"),
        AIMessage(
            "Tercatat.", usage_metadata={"input_tokens": 30, "output_tokens": 5, "total_tokens": 35}
        ),
    ]


def test_trace_has_only_this_turn_with_paired_observations() -> None:
    steps = extract_trace(_history())
    assert len(steps) == 2
    assert [c.name for s in steps for c in s.calls] == ["add_expense"]
    assert steps[0].calls[0].args == {"amount": "40rb"}
    assert steps[0].calls[0].observation == '{"ok":true,"data":{"id":1}}'
    assert (steps[0].input_tokens, steps[0].output_tokens) == (10, 2)
    assert (steps[1].input_tokens, steps[1].output_tokens) == (30, 5)


def test_final_text_is_last_plain_ai_message_of_this_turn() -> None:
    assert final_text(_history()) == "Tercatat."
    assert final_text([HumanMessage("x"), AIMessage("")]) is None


def test_format_trace_is_readable_and_tolerates_missing_usage() -> None:
    out = format_trace(extract_trace(_history()))
    assert "step 1  [tokens in=10 out=2]" in out
    assert '-> add_expense({"amount": "40rb"})' in out
    assert '<- {"ok": true, "data": {"id": 1}}' in out
    assert "old_tool" not in out
    assert "tokens" not in format_trace([ModelStep()])
