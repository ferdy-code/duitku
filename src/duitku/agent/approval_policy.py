"""Which tool calls need the user's explicit OK before they run (docs/spec/agent.md)."""

from langchain_core.messages import ToolCall


def needs_approval(call: ToolCall) -> bool:
    """`delete_expense` always; `add_expense` only when it records a receipt (OCR can be wrong).
    Everything else is fixable afterwards via update_expense."""
    if call["name"] == "delete_expense":
        return True
    return call["name"] == "add_expense" and call["args"].get("source") == "receipt"
