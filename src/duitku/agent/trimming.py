"""Trimmed view of the message history for the model call (T-204).

Stored state is never touched: we return a NEW list (sharing message objects, which are not
mutated). Starting the window on a HumanMessage is what keeps tool_calls <-> ToolMessage groups
whole: an AIMessage(tool_calls) and its ToolMessages always sit after the human message that
triggered them, so a window beginning at a human message can never begin with an orphan
ToolMessage or cut a group in half.
"""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, trim_messages  # pyright: ignore[reportUnknownVariableType]


def trim_history(messages: Sequence[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Newest-first window of `messages` within `max_tokens` (approximate count).

    The latest HumanMessage and everything after it is always kept, even over budget:
    dropping it would erase the question (or split a running tool loop). The system prompt is
    not handled here; the caller prepends it separately.
    """
    # shortcut: chars/4 approximate counter, swap for a real tokenizer if budgets get tight.
    trimmed = trim_messages(
        list(messages),
        max_tokens=max_tokens,
        token_counter="approximate",
        strategy="last",
        start_on="human",
    )
    last_human = next(
        (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
        None,
    )
    if last_human is None:
        return list(messages)  # no human turn to anchor on; nothing safe to cut
    mandatory = list(messages[last_human:])
    return trimmed if len(trimmed) >= len(mandatory) else mandatory
