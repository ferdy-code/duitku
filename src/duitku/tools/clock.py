"""Injectable clock: tool logic calls `clock.now()` (module attribute), never datetime.now().

Tests override it with `monkeypatch.setattr(clock, "now", lambda: ...)`.
"""

from datetime import UTC, datetime


def now() -> datetime:
    """Timezone-aware current time (UTC); convert to the user's tz at the point of use."""
    return datetime.now(UTC)
