"""Approval UI: inline buttons, the button callback, and `/batal`.

The thread is always derived from the chat (never from callback data, which is only `appr:y` /
`appr:n`). The runner owns the pause/resume guards, so a stale or double press comes back as the
friendly NOTHING_PENDING_REPLY and has no effect. Updates run one after another (PTB default) and
`TurnRunner` locks per thread, so a button press and a text for the same chat cannot interleave.
"""

import html
import logging
from typing import Literal, cast

from telegram import Bot, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from duitku.agent.checkpointer import tg_thread_id
from duitku.interfaces.runner import NOTHING_PENDING_REPLY, TurnResult, TurnRunner
from duitku.interfaces.telegram import texts
from duitku.interfaces.telegram.formatting import send_reply, typing_indicator

logger = logging.getLogger(__name__)

Ctx = ContextTypes.DEFAULT_TYPE

APPROVE_DATA = "appr:y"
REJECT_DATA = "appr:n"
CALLBACK_PATTERN = r"^appr:(y|n)$"


def approval_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(texts.BTN_APPROVE, callback_data=APPROVE_DATA),
                InlineKeyboardButton(texts.BTN_REJECT, callback_data=REJECT_DATA),
            ]
        ]
    )


async def send_result(bot: Bot, chat_id: int, result: TurnResult) -> None:
    """Send a turn's text; when the graph is paused, the question comes with the buttons."""
    await send_reply(bot, chat_id, result.text, approval_keyboard() if result.pending else None)


async def _answer(query: CallbackQuery) -> None:
    try:
        await query.answer()  # always: stops the button spinner
    except TelegramError as e:  # e.g. the query is too old; the press is still processed
        logger.warning("callback answer failed (%s)", type(e).__name__)


async def _close_buttons(query: CallbackQuery, outcome: str | None) -> None:
    """Edit the question: drop the keyboard (editMessageText without reply_markup removes it) and
    append the outcome. A message too old to edit (InaccessibleMessage) or a failed edit is
    ignored: the final reply goes out as a new message anyway."""
    msg = query.message
    if not isinstance(msg, Message) or msg.text is None:
        return
    body = html.escape(msg.text, quote=False) + (f"\n\n{outcome}" if outcome else "")
    try:
        await query.edit_message_text(body, parse_mode=ParseMode.HTML)
    except TelegramError as e:
        logger.warning("approval message edit failed (%s)", type(e).__name__)


async def _resume(
    update: Update, context: Ctx, decision: Literal["approve", "reject"]
) -> TurnResult | None:
    user, chat = update.effective_user, update.effective_chat
    if user is None or chat is None:
        return None
    runner = cast(TurnRunner, context.bot_data["runner"])
    async with typing_indicator(context.bot, chat.id):
        ctx = await runner.resolve_user(user.id, user.full_name)
        return await runner.resume(tg_thread_id(chat.id), ctx, decision)


async def on_approval(update: Update, context: Ctx) -> None:
    query, chat = update.callback_query, update.effective_chat
    if query is None:
        return
    await _answer(query)
    if query.data not in (APPROVE_DATA, REJECT_DATA):
        return  # PTB's pattern lets an empty/missing payload through; never default to "reject"
    approve = query.data == APPROVE_DATA
    result = await _resume(update, context, "approve" if approve else "reject")
    if result is None or chat is None:
        return
    stale = result.pending is None and result.text == NOTHING_PENDING_REPLY
    if not result.failed:  # on failure the buttons stay so the user can press again
        outcome = texts.OUTCOME_APPROVED if approve else texts.OUTCOME_REJECTED
        await _close_buttons(query, None if stale else outcome)
    await send_result(context.bot, chat.id, result)


async def on_batal(update: Update, context: Ctx) -> None:
    """`/batal` rejects the pending approval; with nothing pending the runner says so."""
    result = await _resume(update, context, "reject")
    if result is not None and update.effective_chat is not None:
        await send_result(context.bot, update.effective_chat.id, result)
