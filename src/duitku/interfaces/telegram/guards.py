"""Gatekeeping that runs before every other handler: allowlist (R-14) and per-user rate limit.

Both live in ONE group -1 handler because PTB runs only the first matching handler per group; the
`ApplicationHandlerStop` it raises keeps later groups (the real handlers) from ever seeing the
update. Default-deny: a user not in the allowlist, an update without a user, or a non-private
chat (threads are keyed by chat id, so a group would mix users) is dropped silently.
"""

import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable, Collection, Coroutine
from typing import Any

from telegram import Update
from telegram.constants import ChatType
from telegram.error import TelegramError
from telegram.ext import ApplicationHandlerStop, ContextTypes

from duitku.interfaces.telegram.texts import RATE_LIMITED

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding window per user, in memory (one process). Only accepted messages count, so a
    spammer is unblocked `window` seconds after the last accepted one, not after the last try."""

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._limit = limit
        self._window = window
        self._clock = clock
        self._hits: defaultdict[int, deque[float]] = defaultdict(deque)

    def allow(self, user_id: int) -> bool:
        now = self._clock()
        hits = self._hits[user_id]
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return False
        hits.append(now)
        return True


def make_guard(
    allowed_user_ids: Collection[int], limiter: RateLimiter
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, None]]:
    allowed = frozenset(allowed_user_ids)

    async def guard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user, chat = update.effective_user, update.effective_chat
        if user is None or chat is None or chat.type != ChatType.PRIVATE or user.id not in allowed:
            logger.warning("dropped update from a non-allowlisted source")  # no id, no content
            raise ApplicationHandlerStop
        if update.message is not None and not limiter.allow(user.id):
            try:
                await context.bot.send_message(chat.id, RATE_LIMITED)
            except TelegramError as e:
                logger.warning("rate-limit notice failed (%s)", type(e).__name__)
            raise ApplicationHandlerStop

    return guard
