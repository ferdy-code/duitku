"""T-404: receipt photos through the REAL PTB handler chain (fake HTTP layer incl. file download,
fake runner): no network, no DB, no LLM."""

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from telegram.error import NetworkError
from tests.telegram_fakes import (
    PENDING,
    FakeBot,
    FakeRunner,
    document_update,
    make_app,
    photo_update,
    text_update,
)

from duitku.interfaces.telegram import texts
from duitku.receipts.storage import ReceiptRejected

ALLOWED = [1]
MAX = 20000
JPEG = b"\xff\xd8\xff\xe0" + b"j" * 100
PNG = b"\x89PNG\r\n\x1a\n" + b"p" * 100


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.setattr(
        "duitku.interfaces.telegram.receipts.get_settings",
        lambda: SimpleNamespace(receipt_max_bytes=MAX),
    )


async def setup(runner: FakeRunner | None = None) -> tuple[FakeBot, FakeRunner, Any]:
    bot, runner = FakeBot(), runner or FakeRunner("Struknya sudah kubaca.")
    bot.files.update({"ph_s": b"small", "ph_l": JPEG, "doc1": PNG})
    return bot, runner, await make_app(bot, runner, ALLOWED)


def texts_sent(bot: FakeBot) -> list[str]:
    return [m["text"] for m in bot.sent()]


# --- accepted ---------------------------------------------------------------------------------


async def test_photo_downloads_largest_size_stores_it_and_sends_marker_with_caption() -> None:
    async def slow() -> str:  # long enough for the typing indicator to fire once
        await asyncio.sleep(0.05)
        return "Struknya sudah kubaca."

    bot, runner, app = await setup(FakeRunner(slow))
    await app.process_update(photo_update(bot, caption="makan siang tim"))
    assert bot.downloads == ["ph_l"]  # the largest size, never the thumbnail
    assert runner.saved == [JPEG]
    (thread, message) = runner.runs[0]
    rid = uuid.UUID(int=1)
    assert thread == "tg:1"
    assert message == f"[User sent a receipt photo. receipt_id={rid}]\nmakan siang tim"
    assert texts_sent(bot) == ["Struknya sudah kubaca."]
    assert "sendChatAction" in bot.endpoints()


async def test_photo_without_caption_sends_only_the_marker() -> None:
    bot, runner, app = await setup()
    await app.process_update(photo_update(bot))
    assert runner.runs[0][1] == f"[User sent a receipt photo. receipt_id={uuid.UUID(int=1)}]"


async def test_agent_message_never_carries_image_data() -> None:
    bot, runner, app = await setup()
    await app.process_update(photo_update(bot, caption="c" * 5000))
    message = runner.runs[0][1]
    assert len(message) <= texts.MAX_TEXT_CHARS + 100  # caption capped
    assert JPEG.decode("latin-1") not in message and "base64" not in message


async def test_png_document_is_accepted() -> None:
    bot, runner, app = await setup()
    await app.process_update(document_update(bot, mime="image/png"))
    assert runner.saved == [PNG] and len(runner.runs) == 1


async def test_pending_approval_result_is_sent_with_buttons() -> None:
    bot, _, app = await setup(FakeRunner(pending=PENDING))
    await app.process_update(photo_update(bot))
    (msg,) = bot.sent()
    assert [b.callback_data for r in msg["reply_markup"].inline_keyboard for b in r] == [
        "appr:y",
        "appr:n",
    ]


# --- refused ----------------------------------------------------------------------------------


@pytest.mark.parametrize("mime", ["application/pdf", "image/webp", "image/gif", None])
async def test_non_jpeg_png_document_is_refused_without_download(mime: str | None) -> None:
    bot, runner, app = await setup()
    await app.process_update(document_update(bot, mime=mime))
    assert texts_sent(bot) == [texts.RECEIPT_UNSUPPORTED]
    assert "getFile" not in bot.endpoints() and not runner.saved and not runner.runs


async def test_size_reported_by_telegram_is_checked_before_download() -> None:
    bot, runner, app = await setup()
    await app.process_update(photo_update(bot, sizes=[("big", 4000, 3000, MAX + 1)]))
    assert texts_sent(bot) == [texts.RECEIPT_TOO_BIG.format(mb=f"{MAX / 1_048_576:g}")]
    assert "getFile" not in bot.endpoints() and not runner.saved and not runner.runs


async def test_size_reported_by_getfile_is_checked_before_download() -> None:
    bot, runner, app = await setup()
    bot.reported_size["ph_l"] = MAX + 1  # the update said nothing, getFile knows better
    await app.process_update(photo_update(bot, sizes=[("ph_l", 1280, 960, None)]))
    assert texts_sent(bot)[0].startswith("Fotonya terlalu besar")
    assert bot.downloads == [] and not runner.saved and not runner.runs


async def test_oversize_found_after_download_is_refused() -> None:
    runner = FakeRunner()
    runner.save_error = ReceiptRejected("size")
    bot, _, app = await setup(runner)
    await app.process_update(photo_update(bot))
    assert texts_sent(bot)[0].startswith("Fotonya terlalu besar")
    assert not runner.runs


async def test_wrong_magic_bytes_are_refused_even_for_a_photo() -> None:
    runner = FakeRunner()
    runner.save_error = ReceiptRejected("type")
    bot, _, app = await setup(runner)
    await app.process_update(photo_update(bot))
    assert texts_sent(bot) == [texts.RECEIPT_UNSUPPORTED] and not runner.runs


async def test_download_failure_is_reported_and_does_not_run_the_agent() -> None:
    bot, runner, app = await setup()
    bot.fail["getFile"] = [NetworkError("down")]
    await app.process_update(photo_update(bot))
    assert texts_sent(bot) == [texts.RECEIPT_DOWNLOAD_FAILED]
    assert not runner.saved and not runner.runs


async def test_photo_while_approval_pending_reminds_and_reasks_without_downloading() -> None:
    bot, runner, app = await setup(FakeRunner(paused=PENDING))
    await app.process_update(photo_update(bot))
    first, second = bot.sent()
    assert first["text"] == texts.RECEIPT_WHILE_PENDING
    assert second["text"].startswith("Hapus pengeluaran")
    assert second.get("reply_markup") is not None
    assert "getFile" not in bot.endpoints() and not runner.saved and not runner.runs


# --- guards and resilience ---------------------------------------------------------------------


async def test_photo_from_non_allowlisted_user_is_dropped() -> None:
    bot, runner, app = await setup()
    await app.process_update(photo_update(bot, user_id=99))
    assert bot.sent() == [] and "getFile" not in bot.endpoints() and not runner.saved


async def test_photo_in_a_group_is_dropped() -> None:
    bot, runner, app = await setup()
    await app.process_update(photo_update(bot, chat_type="supergroup"))
    assert bot.sent() == [] and not runner.saved


async def test_photos_share_the_rate_limit_with_text() -> None:
    bot, runner = FakeBot(), FakeRunner()
    bot.files["ph_l"] = JPEG
    app = await make_app(bot, runner, ALLOWED, limit=1)
    await app.process_update(text_update(bot, "hai"))
    await app.process_update(photo_update(bot))
    assert texts_sent(bot)[-1] == texts.RATE_LIMITED and not runner.saved


async def test_runner_failure_gets_the_generic_apology_not_a_crash() -> None:
    bot, _, app = await setup(FakeRunner(boom=RuntimeError("secret detail")))
    await app.process_update(photo_update(bot))
    assert texts_sent(bot) == [texts.GENERIC_APOLOGY]


def test_help_mentions_receipts() -> None:
    assert "struk" in texts.HELP
