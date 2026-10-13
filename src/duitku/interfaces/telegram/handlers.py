"""Command, text and error handlers.

Only `/start` and `/bantuan` exist now; `/batal` and `/lupakan` arrive with later tasks.
T-403 (approval): add a `CallbackQueryHandler(on_approval, pattern=r"^appr:(y|n)$")` in
`app.build_application`, call `runner.resume_turn(...)` there, and add "callback_query" to
`app.ALLOWED_UPDATES` and a callback branch to the guard (it only rate-limits `update.message`).
"""

import logging
from typing import cast

from telegram import Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from duitku.agent.checkpointer import tg_thread_id
from duitku.interfaces.runner import TurnRunner
from duitku.interfaces.telegram import texts
from duitku.interfaces.telegram.formatting import send_reply, typing_indicator

logger = logging.getLogger(__name__)

Ctx = ContextTypes.DEFAULT_TYPE


async def on_start(update: Update, context: Ctx) -> None:
    if update.effective_chat:
        await send_reply(context.bot, update.effective_chat.id, texts.START)


async def on_help(update: Update, context: Ctx) -> None:
    if update.effective_chat:
        await send_reply(context.bot, update.effective_chat.id, texts.HELP)


async def on_text(update: Update, context: Ctx) -> None:
    user, chat, msg = update.effective_user, update.effective_chat, update.effective_message
    text = (msg.text or "").strip() if msg else ""
    if not (user and chat and text):  # non-text / empty: ignore
        return
    if len(text) > texts.MAX_TEXT_CHARS:
        await send_reply(context.bot, chat.id, texts.TOO_LONG)
        return
    runner = cast(TurnRunner, context.bot_data["runner"])
    async with typing_indicator(context.bot, chat.id):
        ctx = await runner.resolve_user(user.id, user.full_name)
        result = await runner.run(tg_thread_id(chat.id), ctx, text)
    await send_reply(
        context.bot, chat.id, result.text
    )  # a failed turn already holds a friendly text


async def on_error(update: object, context: Ctx) -> None:
    """Never raises: log the exception class only (messages can carry content or the token URL)."""
    logger.error("update handler failed (%s)", type(context.error).__name__)
    chat = update.effective_chat if isinstance(update, Update) else None
    if chat is None:
        return
    try:
        await context.bot.send_message(chat.id, texts.GENERIC_APOLOGY)
    except TelegramError as e:
        logger.error("could not send the apology (%s)", type(e).__name__)
