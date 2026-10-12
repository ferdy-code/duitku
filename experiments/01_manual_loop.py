"""T-105: tool calling from first principles. No LangGraph, no LangChain agent: just the OpenAI SDK.

The whole "agent" is the loop in `run_turn`:

    send messages -> read message.tool_calls -> append the assistant message (WITH its tool_calls)
    -> run each tool -> append ONE {"role": "tool", "tool_call_id": ...} message per call -> repeat
    until the model answers without tool_calls (finish_reason != "tool_calls").

Run (dev DB, real LLM):  uv run python experiments/01_manual_loop.py --verbose [--reset]
Only the T-104 tools are reused (their schemas and execution); the loop is ours.
"""

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from sqlalchemy import delete

from duitku.config import get_settings
from duitku.db.engine import get_sessionmaker
from duitku.db.models import User
from duitku.db.repositories.users import get_or_create_user
from duitku.tools import clock
from duitku.tools.base import error
from duitku.tools.registry import build_tools

# Fixed experiment user (dev DB only). --reset deletes it (ON DELETE CASCADE) and recreates it.
EXPERIMENT_TELEGRAM_ID = 900_000_105
SCENARIO = ["makan siang 35rb di warteg", "total hari ini berapa?"]

Messages = list[Any]
Complete = Callable[[Messages], Awaitable[ChatCompletion]]
Log = Callable[[str], None]


def system_prompt(tz: str) -> str:
    now = clock.now().astimezone(ZoneInfo(tz))
    return (
        "Kamu Duitku, asisten pencatat pengeluaran. Balas dalam bahasa Indonesia, singkat. "
        f"Hari ini {now:%Y-%m-%d} (zona waktu {tz}). "
        "Jangan pernah menghitung angka atau tanggal sendiri: pakai tool."
    )


def to_openai_tool(tool: StructuredTool) -> dict[str, Any]:
    """StructuredTool -> Chat Completions `tools` entry.

    `tool_call_schema` is the LLM-facing schema: injected params (config, tool_call_id) are
    already stripped, so the model can never see or fake them.
    """
    schema: Any = tool.tool_call_schema  # a pydantic model for our tools (typed loosely: pyright)
    params: dict[str, Any] = schema.model_json_schema()
    params.pop("description", None)  # lives on the function, not in the parameters schema
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": params},
    }


def _envelope(code: str, message: str, hint: str) -> str:
    return json.dumps(error(code, message, hint), ensure_ascii=False)


async def execute_tool_call(
    tools: dict[str, StructuredTool], call_id: str, name: str, raw_args: str, config: RunnableConfig
) -> str:
    """Run one tool call and ALWAYS return a string: the content of its `role:"tool"` message.

    Unknown tool / bad JSON / crash become error envelopes (observations), never exceptions,
    because every tool_call_id must get exactly one answer or the next API call is rejected.
    """
    tool = tools.get(name)
    if tool is None:
        return _envelope("UNKNOWN_TOOL", f"No tool named {name!r}.", f"Use one of: {sorted(tools)}")
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        return _envelope("INVALID_ARGUMENT", "Arguments are not a JSON object.", "Send valid JSON.")
    try:
        # A ToolCall dict (not bare args) is what makes InjectedToolCallId work.
        result = await tool.ainvoke(
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}, config=config
        )
    except Exception as e:  # boundary: any crash must still produce an answer
        return _envelope("INTERNAL_ERROR", f"Tool crashed ({type(e).__name__}).", "Do not retry.")
    return str(getattr(result, "content", result))  # ToolMessage.content is the JSON envelope


async def run_turn(
    messages: Messages,
    complete: Complete,
    tools: dict[str, StructuredTool],
    config: RunnableConfig,
    max_steps: int,
    log: Log = lambda _: None,
) -> str:
    """One user turn. Mutates `messages` (the conversation state) and returns the final reply."""
    for step in range(1, max_steps + 1):
        resp = await complete(messages)
        choice = resp.choices[0]
        msg = choice.message
        usage = resp.usage
        tokens = usage.total_tokens if usage else None
        log(f"[step {step}] finish_reason={choice.finish_reason} tokens={tokens}")
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
                if tc.type == "function"
            ]
        messages.append(assistant)  # must carry tool_calls, or the tool messages below are orphans
        calls = assistant.get("tool_calls", [])
        if not calls:  # stop condition: model answered instead of asking for tools
            return msg.content or ""
        for tc in calls:
            fn = tc["function"]
            log(f"  assistant -> tool_call {tc['id']}: {fn['name']}({fn['arguments']})")
            obs = await execute_tool_call(tools, tc["id"], fn["name"], fn["arguments"], config)
            log(f"  tool <- {tc['id']}: {obs}")
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})
    return f"(berhenti: melewati batas {max_steps} langkah)"


async def ensure_user() -> tuple[Any, str]:
    sm = get_sessionmaker()
    async with sm.begin() as session:
        user, _ = await get_or_create_user(session, EXPERIMENT_TELEGRAM_ID, "experiment-105")
    return user.id, user.timezone


async def main(verbose: bool, reset: bool) -> None:
    s = get_settings()
    if reset:
        async with get_sessionmaker().begin() as session:
            await session.execute(
                delete(User).where(User.telegram_user_id == EXPERIMENT_TELEGRAM_ID)
            )
    user_id, tz = await ensure_user()
    tool_list = build_tools()
    tools = {t.name: t for t in tool_list}
    client = AsyncOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key.get_secret_value(),
        timeout=s.llm_timeout_seconds,
    )
    openai_tools: list[Any] = [to_openai_tool(t) for t in tool_list]

    async def complete(msgs: Messages) -> ChatCompletion:
        return await client.chat.completions.create(
            model=s.model_agent, messages=msgs, tools=openai_tools
        )

    config = RunnableConfig(configurable={"user_id": str(user_id), "timezone": tz})
    log: Log = print if verbose else (lambda _: None)
    messages: Messages = [{"role": "system", "content": system_prompt(tz)}]  # kept across turns
    for text in SCENARIO:
        print(f"\nuser: {text}")
        messages.append({"role": "user", "content": text})
        print(
            f"duitku: {await run_turn(messages, complete, tools, config, s.max_agent_steps, log)}"
        )
    if verbose:
        print(f"\n--- final messages ({len(messages)}) ---")
        for m in messages:
            print(json.dumps(m, ensure_ascii=False, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verbose", action="store_true", help="show tool calls, observations, messages"
    )
    ap.add_argument("--reset", action="store_true", help="delete the experiment user's data first")
    a = ap.parse_args()
    asyncio.run(main(a.verbose, a.reset))
