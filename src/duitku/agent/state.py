"""LangGraph state (docs/spec/agent.md). user_id/timezone live in RunnableConfig, not here."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage

# langgraph/ has no __init__.py (namespace package) so pyright strict reports missing stubs.
from langgraph.graph import add_messages  # pyright: ignore[reportMissingTypeStubs]


class AgentState(TypedDict):
    # add_messages appends/merges by message id instead of overwriting the list.
    messages: Annotated[list[AnyMessage], add_messages]
    # Set by load_context each run (plain overwrite); rendered into the prompt as DATA.
    recalled_memories: list[str]
    # Set by approval, read by tools. Plain overwrite: the approval node must return the full list.
    rejected_call_ids: list[str]
