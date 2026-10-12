"""T-105: pure helpers of experiments/01_manual_loop.py (fake LLM, no network, no DB)."""

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from openai.types.chat import ChatCompletion

from duitku.tools.registry import build_tools

_path = Path(__file__).resolve().parents[2] / "experiments" / "01_manual_loop.py"
_spec = importlib.util.spec_from_file_location("manual_loop", _path)
assert _spec and _spec.loader
loop: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loop)

CONFIG = RunnableConfig(configurable={"user_id": "x", "timezone": "Asia/Jakarta"})


async def _echo(text: str) -> str:
    return json.dumps({"ok": True, "data": text})


async def _boom() -> str:
    raise RuntimeError("secret internals")


TOOLS = {
    "echo": StructuredTool.from_function(coroutine=_echo, name="echo", description="echo"),
    "boom": StructuredTool.from_function(coroutine=_boom, name="boom", description="boom"),
}


def _completion(
    finish: str, content: str | None = None, calls: Sequence[tuple[str, str, str]] = ()
) -> ChatCompletion:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [
            {"id": i, "type": "function", "function": {"name": n, "arguments": a}}
            for i, n, a in calls
        ]
    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
        }
    )


def test_openai_tool_hides_injected_params() -> None:
    for t in build_tools():
        f = loop.to_openai_tool(t)
        assert f["type"] == "function"
        assert f["function"]["name"] == t.name
        props = f["function"]["parameters"]["properties"]
        assert not {"config", "tool_call_id"} & set(props)
        assert "description" not in f["function"]["parameters"]


async def test_every_tool_call_id_gets_exactly_one_tool_message() -> None:
    first = _completion(
        "tool_calls",
        calls=[
            ("a", "echo", '{"text": "hi"}'),
            ("b", "nope", "{}"),  # unknown tool
            ("c", "echo", "{not json"),  # bad JSON
            ("d", "boom", "{}"),  # crashes
            ("e", "echo", "[1]"),  # not an object
        ],
    )
    replies = iter([first, _completion("stop", "selesai")])

    async def complete(_: list[Any]) -> ChatCompletion:
        return next(replies)

    messages: list[Any] = [{"role": "user", "content": "x"}]
    out = await loop.run_turn(messages, complete, TOOLS, CONFIG, max_steps=5)

    assert out == "selesai"
    assistant = messages[1]
    ids = [tc["id"] for tc in assistant["tool_calls"]]
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ids == ["a", "b", "c", "d", "e"]
    assert json.loads(tool_msgs[0]["content"])["ok"] is True
    codes = [json.loads(m["content"])["error"]["code"] for m in tool_msgs[1:]]
    assert codes == ["UNKNOWN_TOOL", "INVALID_ARGUMENT", "INTERNAL_ERROR", "INVALID_ARGUMENT"]
    assert "secret" not in tool_msgs[3]["content"]


async def test_max_steps_guard_keeps_pairing() -> None:
    async def complete(_: list[Any]) -> ChatCompletion:
        return _completion("tool_calls", calls=[("x", "echo", '{"text": "a"}')])

    messages: list[Any] = []
    out = await loop.run_turn(messages, complete, TOOLS, CONFIG, max_steps=3)

    assert "3" in out
    assert [m["role"] for m in messages] == ["assistant", "tool"] * 3
