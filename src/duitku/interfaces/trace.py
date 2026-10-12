"""What happened during one turn: model steps, tool calls and their observations.

Built from the NEW messages of the turn only (everything after the user's message), so a
long checkpointed thread never leaks older turns into the trace. Pure functions, no I/O.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage


@dataclass(frozen=True)
class ToolStep:
    name: str
    args: dict[str, object]
    observation: str  # the ToolMessage content (our JSON envelope); "" if none was found


@dataclass(frozen=True)
class ModelStep:
    """One LLM call: optional tool calls it requested, token usage if the provider sent it."""

    calls: list[ToolStep] = field(default_factory=lambda: list[ToolStep]())
    input_tokens: int | None = None
    output_tokens: int | None = None


def new_messages(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Messages after the last HumanMessage, i.e. the part produced by this turn."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return list(messages[i + 1 :])
    return list(messages)


def extract_trace(messages: Sequence[AnyMessage]) -> list[ModelStep]:
    """One ModelStep per AIMessage of this turn; tool calls paired with their ToolMessage."""
    turn = new_messages(messages)
    observations = {m.tool_call_id: str(m.content) for m in turn if isinstance(m, ToolMessage)}
    steps: list[ModelStep] = []
    for m in turn:
        if not isinstance(m, AIMessage):
            continue
        usage = m.usage_metadata
        steps.append(
            ModelStep(
                calls=[
                    ToolStep(c["name"], dict(c["args"]), observations.get(c["id"] or "", ""))
                    for c in m.tool_calls
                ],
                input_tokens=usage["input_tokens"] if usage else None,
                output_tokens=usage["output_tokens"] if usage else None,
            )
        )
    return steps


def final_text(messages: Sequence[AnyMessage]) -> str | None:
    """Text of the last AIMessage of this turn that is not a tool request."""
    for m in reversed(new_messages(messages)):
        if isinstance(m, AIMessage) and not m.tool_calls and m.text.strip():
            return m.text
    return None


def _compact(s: str) -> str:
    try:
        return json.dumps(json.loads(s), ensure_ascii=False)
    except ValueError:
        return s


def format_trace(steps: Sequence[ModelStep]) -> str:
    """Readable multi-line text for `--verbose` (a deliberate developer feature of the CLI)."""
    lines: list[str] = []
    for n, step in enumerate(steps, 1):
        usage = (
            f"  [tokens in={step.input_tokens} out={step.output_tokens}]"
            if step.input_tokens is not None
            else ""
        )
        lines.append(f"  step {n}{usage}")
        for call in step.calls:
            args = json.dumps(call.args, ensure_ascii=False)
            lines.append(f"    -> {call.name}({args})")
            lines.append(f"    <- {_compact(call.observation)}")
    return "\n".join(lines)
