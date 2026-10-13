"""Input hygiene for long-term memories. T-304 extends `validate_content` (sensitive-data regex)."""

from typing import Final, Literal, get_args

MemoryKind = Literal["preference", "category_rule", "goal", "fact"]
MEMORY_KINDS: Final[tuple[str, ...]] = get_args(MemoryKind)

# Same cap the system prompt applies when rendering a memory (T-201); longer text would be cut
# there anyway, so reject instead of silently truncating a fact.
MAX_CONTENT_CHARS: Final = 300


class InvalidMemoryError(ValueError):
    """Bad kind or content. A ValueError, so the tool layer maps it to INVALID_ARGUMENT."""


def validate_kind(kind: str) -> MemoryKind:
    if kind not in MEMORY_KINDS:
        raise InvalidMemoryError(f"kind must be one of {', '.join(MEMORY_KINDS)}")
    return kind  # type: ignore[return-value]  # narrowed by the membership check above


def validate_content(content: str) -> str:
    """Strip and length-check; returns the cleaned text. T-304 adds the sensitive-data check."""
    text = content.strip()
    if not text:
        raise InvalidMemoryError("content must not be empty")
    if len(text) > MAX_CONTENT_CHARS:
        raise InvalidMemoryError(f"content too long ({len(text)} > {MAX_CONTENT_CHARS} chars)")
    return text
