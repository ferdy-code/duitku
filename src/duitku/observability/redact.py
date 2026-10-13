"""Privacy helpers for logs, spans and audit rows: ONE place decides what content may leave.

`LOG_CONTENT=false` (default) means message text and tool args never appear in spans/logs.
User ids appear only as a short hash.
"""

import hashlib
import json
import uuid
from typing import Any

from duitku.config import ConfigError, get_settings

REDACTED = "[redacted]"
MAX_ATTR_CHARS = 500


def redact_args(args: dict[str, Any], log_content: bool) -> dict[str, Any]:
    """JSON-safe copy of args; values replaced by a marker unless LOG_CONTENT=true."""
    if not log_content:
        return {k: REDACTED for k in args}
    return json.loads(json.dumps(args, default=str))


def content_attr(value: Any, log_content: bool) -> str | None:
    """Truncated JSON text for a span attribute, or None (= attribute NOT set) when redacting."""
    if not log_content:
        return None
    s = json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= MAX_ATTR_CHARS else s[: MAX_ATTR_CHARS - 1] + "…"


def user_hash(user_id: uuid.UUID | str) -> str:
    """Stable 16-hex pseudonym for span attributes. Plain sha256 (no salt): the input is the
    internal random UUID (122 bits), not a guessable Telegram id, so it cannot be brute-forced."""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:16]


def content_logging_enabled() -> bool:
    """LOG_CONTENT, defaulting to the safe False if settings cannot be loaded."""
    try:
        return get_settings().log_content
    except ConfigError:
        return False
