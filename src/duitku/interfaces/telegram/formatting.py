"""Telegram output helpers: HTML escaping, 4096-char splitting, typing keep-alive."""

import asyncio
import contextlib
import html
import logging
from collections.abc import AsyncGenerator

from telegram import Bot, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, TelegramError

logger = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4096
TYPING_INTERVAL = 4.0  # Telegram shows "typing" for ~5 s per call


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split at the last newline (else space) within `limit`; hard-cut only if there is none."""
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    if text.strip():
        chunks.append(text)
    return chunks


async def send_reply(
    bot: Bot, chat_id: int, text: str, reply_markup: InlineKeyboardMarkup | None = None
) -> None:
    """Send as HTML (model text is escaped, so `<b>` in a reply is shown, never interpreted).
    Chunks are split BEFORE escaping so an entity is never cut in half. If Telegram rejects the
    HTML (BadRequest), that chunk is resent once as plain text. `reply_markup` (buttons) goes on
    the last chunk only."""
    chunks = split_message(text)
    for i, chunk in enumerate(chunks):
        markup = reply_markup if i == len(chunks) - 1 else None
        try:
            await bot.send_message(
                chat_id,
                html.escape(chunk, quote=False),
                parse_mode=ParseMode.HTML,
                reply_markup=markup,
            )
        except BadRequest as e:
            logger.warning("HTML send rejected (%s); resending as plain text", type(e).__name__)
            await bot.send_message(chat_id, chunk, reply_markup=markup)


@contextlib.asynccontextmanager
async def typing_indicator(
    bot: Bot, chat_id: int, interval: float = TYPING_INTERVAL
) -> AsyncGenerator[None]:
    """Keep "typing..." on while the body runs; a failed ping never breaks the turn."""

    async def loop() -> None:
        while True:
            try:
                await bot.send_chat_action(chat_id, ChatAction.TYPING)
            except TelegramError as e:
                logger.debug("typing action failed (%s)", type(e).__name__)
            await asyncio.sleep(interval)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
