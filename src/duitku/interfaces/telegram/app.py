"""Handler registration only, no runner/DB, so tests can build the app offline. The entrypoint
(`bot.py`) injects `bot_data["runner"]` in `post_init`."""

from collections.abc import Callable
from typing import Any

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    TypeHandler,
    filters,
)

from duitku.interfaces.telegram.approval import CALLBACK_PATTERN, on_approval, on_batal
from duitku.interfaces.telegram.guards import RateLimiter, make_guard
from duitku.interfaces.telegram.handlers import on_error, on_help, on_start, on_text
from duitku.interfaces.telegram.receipts import on_receipt

type App = Application[Any, Any, Any, Any, Any, Any]

ALLOWED_UPDATES = ["message", "callback_query"]
GUARD_GROUP = -1  # lower group runs first; the guard's ApplicationHandlerStop blocks group 0


def register(
    app: App,
    allowed_user_ids: list[int],
    rate_limit_per_minute: int,
    clock: Callable[[], float] | None = None,
) -> App:
    limiter = (
        RateLimiter(rate_limit_per_minute, clock=clock)
        if clock
        else RateLimiter(rate_limit_per_minute)
    )
    app.add_handler(TypeHandler(Update, make_guard(allowed_user_ids, limiter)), GUARD_GROUP)
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("bantuan", on_help))
    app.add_handler(CommandHandler("batal", on_batal))
    app.add_handler(CallbackQueryHandler(on_approval, pattern=CALLBACK_PATTERN))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    # Photos and documents (any document: a non-image one gets a polite refusal in the handler).
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, on_receipt))
    app.add_error_handler(on_error)
    return app
