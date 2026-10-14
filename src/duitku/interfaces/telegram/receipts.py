"""Receipt photos: download, store (file + `receipts` row), then hand the agent a marker message.

The agent never sees image bytes: it gets `[User sent a receipt photo. receipt_id=<uuid>]` (+ the
caption) and calls `parse_receipt`. Order of checks (cheapest first): file type and the size
Telegram reports (before any download), pending approval (one thing at a time), download, size and
magic bytes again on the real bytes, only then the file and the DB row. Only photos and JPEG/PNG
documents are accepted; anything else gets a polite refusal.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import cast

from telegram import Message, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from duitku.agent.checkpointer import tg_thread_id
from duitku.config import get_settings
from duitku.interfaces.runner import TurnResult, TurnRunner, UserCtx
from duitku.interfaces.telegram import texts
from duitku.interfaces.telegram.approval import send_result
from duitku.interfaces.telegram.formatting import send_reply, typing_indicator
from duitku.receipts.storage import ReceiptRejected

logger = logging.getLogger(__name__)

Ctx = ContextTypes.DEFAULT_TYPE

_IMAGE_MIMES = frozenset({"image/jpeg", "image/png"})


@dataclass(frozen=True)
class Upload:
    file_id: str
    size: int | None  # as reported by Telegram (may be missing)


def pick_upload(msg: Message) -> Upload | None:
    """The photo's largest size, or an image document (JPEG/PNG); None = not a usable receipt.
    The real type is still checked on the downloaded bytes (magic bytes)."""
    if msg.photo:
        best = max(msg.photo, key=lambda p: p.width * p.height)
        return Upload(best.file_id, best.file_size)
    doc = msg.document
    if doc is not None and doc.mime_type in _IMAGE_MIMES:
        return Upload(doc.file_id, doc.file_size)
    return None


def agent_message(receipt_id: uuid.UUID, caption: str | None) -> str:
    marker = f"[User sent a receipt photo. receipt_id={receipt_id}]"
    cap = (caption or "").strip()[: texts.MAX_TEXT_CHARS]
    return f"{marker}\n{cap}" if cap else marker


def _too_big(max_bytes: int) -> str:
    return texts.RECEIPT_TOO_BIG.format(mb=f"{max_bytes / 1_048_576:g}")


async def _download(context: Ctx, upload: Upload, max_bytes: int) -> bytes | str:
    """The image bytes, or a user-facing refusal text."""
    try:
        tg_file = await context.bot.get_file(upload.file_id)
        if tg_file.file_size is not None and tg_file.file_size > max_bytes:
            return _too_big(max_bytes)
        return bytes(await tg_file.download_as_bytearray())
    except TelegramError as e:  # incl. "file is too big" (>20 MB) and network errors
        logger.warning("receipt download failed (%s)", type(e).__name__)
        return texts.RECEIPT_DOWNLOAD_FAILED


async def _handle(
    context: Ctx, runner: TurnRunner, ctx: UserCtx, chat_id: int, msg: Message, upload: Upload
) -> TurnResult | str:
    thread = tg_thread_id(chat_id)
    if (pending := await runner.pending_approval(thread, ctx)) is not None:
        # Do not even download: the agent is waiting for a button. Re-ask with the buttons.
        await send_reply(context.bot, chat_id, texts.RECEIPT_WHILE_PENDING)
        return TurnResult(str(pending["human_summary"]), pending=pending)
    max_bytes = get_settings().receipt_max_bytes
    data = await _download(context, upload, max_bytes)
    if isinstance(data, str):
        return data
    try:
        receipt_id = await runner.save_receipt(ctx, data)
    except ReceiptRejected as e:
        return _too_big(max_bytes) if e.reason == "size" else texts.RECEIPT_UNSUPPORTED
    finally:
        del data  # the bytes live only for this handler call
    return await runner.run(thread, ctx, agent_message(receipt_id, msg.caption))


async def on_receipt(update: Update, context: Ctx) -> None:
    user, chat, msg = update.effective_user, update.effective_chat, update.effective_message
    if not (user and chat and msg):
        return
    upload = pick_upload(msg)
    if upload is None:  # a document that is not JPEG/PNG
        await send_reply(context.bot, chat.id, texts.RECEIPT_UNSUPPORTED)
        return
    max_bytes = get_settings().receipt_max_bytes
    if upload.size is not None and upload.size > max_bytes:  # before any download
        await send_reply(context.bot, chat.id, _too_big(max_bytes))
        return
    runner = cast(TurnRunner, context.bot_data["runner"])
    async with typing_indicator(context.bot, chat.id):
        ctx = await runner.resolve_user(user.id, user.full_name)
        outcome = await _handle(context, runner, ctx, chat.id, msg, upload)
    if isinstance(outcome, str):
        await send_reply(context.bot, chat.id, outcome)
    else:
        await send_result(context.bot, chat.id, outcome)
