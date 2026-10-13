"""Input hygiene for long-term memories: kind, length, and the sensitive-data guard."""

from typing import Final, Literal, get_args

from duitku.memory.sensitive import find_sensitive

MemoryKind = Literal["preference", "category_rule", "goal", "fact"]
MEMORY_KINDS: Final[tuple[str, ...]] = get_args(MemoryKind)

# Same cap the system prompt applies when rendering a memory (T-201); longer text would be cut
# there anyway, so reject instead of silently truncating a fact.
MAX_CONTENT_CHARS: Final = 300


class InvalidMemoryError(ValueError):
    """Bad kind or content. A ValueError, so the tool layer maps it to INVALID_ARGUMENT."""


class SensitiveDataError(InvalidMemoryError):
    """Content looks like a card/ID/account number, PIN/OTP, password or key. `label` names the
    kind of match; the matched text is never kept."""

    def __init__(self, label: str) -> None:
        super().__init__(f"content looks like sensitive data ({label}); not stored")
        self.label = label


def validate_kind(kind: str) -> MemoryKind:
    if kind not in MEMORY_KINDS:
        raise InvalidMemoryError(f"kind must be one of {', '.join(MEMORY_KINDS)}")
    return kind  # type: ignore[return-value]  # narrowed by the membership check above


def validate_content(content: str) -> str:
    """Strip, length-check and sensitive-data-check; returns the cleaned text."""
    text = content.strip()
    if not text:
        raise InvalidMemoryError("content must not be empty")
    if len(text) > MAX_CONTENT_CHARS:
        raise InvalidMemoryError(f"content too long ({len(text)} > {MAX_CONTENT_CHARS} chars)")
    if label := find_sensitive(text):
        raise SensitiveDataError(label)
    return text
