"""Conditional edges. Pure functions of the state, so they are trivially testable.

The step cap lives here (not only in langgraph's recursion_limit): when the cap is reached and
the model still asks for tools, we route to `limit` instead of `tools`/END. That node answers
the pending tool_calls itself, so the stored history never ends with an unanswered tool_call.
"""

from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from duitku.agent.approval_policy import needs_approval
from duitku.agent.state import AgentState

Route = Literal["tools", "approval", "limit", "__end__"]


def _wants_tools(msg: AnyMessage) -> bool:
    # invalid_tool_calls (unparseable JSON args) are sent back to the API too, so they count.
    return isinstance(msg, AIMessage) and bool(msg.tool_calls or msg.invalid_tool_calls)


def agent_steps_this_turn(messages: list[AnyMessage]) -> int:
    """Model calls since the last HumanMessage, derived from history (no extra state field)."""
    steps = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        steps += isinstance(m, AIMessage)
    return steps


def route_after_agent(state: AgentState, *, max_steps: int) -> Route:
    messages = state["messages"]
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or not _wants_tools(last):
        return "__end__"
    if agent_steps_this_turn(messages) >= max_steps:  # cap first: nothing runs, nothing is asked
        return "limit"
    return "approval" if any(needs_approval(c) for c in last.tool_calls) else "tools"
