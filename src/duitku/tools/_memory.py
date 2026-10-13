"""Shared bits of the memory tools: lazy embedder and the memory-specific error envelopes."""

from duitku.memory.embeddings import Embedder, get_embedder
from duitku.memory.validation import InvalidMemoryError, SensitiveDataError
from duitku.tools.base import Envelope, error

MEMORY_UNAVAILABLE = error(
    "MEMORY_UNAVAILABLE",
    "Long-term memory is temporarily unavailable.",
    "Continue without memory. Briefly tell the user it could not be saved/recalled right now.",
)


def resolve_embedder(injected: Embedder | None) -> Embedder:
    """Injected fake in tests; otherwise the cached real one, resolved at CALL time so that
    importing or building tools never loads torch."""
    return injected or get_embedder()


def memory_error(e: InvalidMemoryError) -> Envelope:
    """Map validation failures to observations. Messages carry no user content."""
    if isinstance(e, SensitiveDataError):
        return error(
            "SENSITIVE_DATA",
            f"Not saved: the text looks like sensitive data ({e.label}).",
            "Do NOT store it and do not retry a disguised version. Tell the user (in Indonesian) "
            "that you never store card/ID/account numbers, PIN, OTP, passwords or keys, and why.",
        )
    return error("INVALID_ARGUMENT", str(e), "Fix the arguments and call the tool again.")
