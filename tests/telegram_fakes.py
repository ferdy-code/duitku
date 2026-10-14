"""Offline Telegram doubles: a real PTB `ExtBot` whose HTTP layer is replaced, so the real
Application / handler chain / Message objects run without any network."""

import itertools
import uuid
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from telegram import Update
from telegram.ext import Application, ExtBot
from telegram.request import BaseRequest

from duitku.interfaces.telegram.app import App, register

TOKEN = "123456:TEST-TOKEN-NOT-REAL"


class FakeRequest(BaseRequest):
    """Serves file downloads from memory: `files` maps a Telegram file_id to its bytes."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.downloads: list[str] = []

    @property
    def read_timeout(self) -> float | None:
        return None

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def do_request(self, *args: Any, **kwargs: Any) -> tuple[int, bytes]:
        raise AssertionError("only file downloads go through the request object in tests")

    async def retrieve(self, url: str, *args: Any, **kwargs: Any) -> bytes:  # pyright: ignore[reportIncompatibleMethodOverride]
        file_id = url.rsplit("/", 1)[-1]
        self.downloads.append(file_id)
        return self.files[file_id]


class FakeBot(ExtBot[None]):
    """Records every API call in `calls` as (endpoint, data). `fail` maps an endpoint to a list
    of exceptions raised one per call (then calls succeed). Files for `getFile` come from
    `files` (file_id -> bytes); `reported_size` overrides the size getFile claims."""

    def __init__(self) -> None:
        request = FakeRequest()
        super().__init__(TOKEN, request=request)
        self._fake_request = request
        self._calls: list[tuple[str, dict[str, Any]]] = []
        self._fail: dict[str, list[Exception]] = {}
        self._reported_size: dict[str, int] = {}

    async def _do_post(self, endpoint: str, data: dict[str, Any], **_: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        self._calls.append((endpoint, dict(data)))
        if self._fail.get(endpoint):
            raise self._fail[endpoint].pop(0)
        if endpoint == "getMe":
            return {"id": 1, "is_bot": True, "first_name": "B", "username": "b_bot"}
        if endpoint == "getFile":
            fid = data["file_id"]
            size = self._reported_size.get(fid, len(self._fake_request.files.get(fid, b"")))
            return {
                "file_id": fid,
                "file_unique_id": f"u-{fid}",
                "file_size": size,
                "file_path": f"files/{fid}",
            }
        if endpoint == "sendMessage":
            return {"message_id": 1, "date": 0, "chat": {"id": data["chat_id"], "type": "private"}}
        return True

    @property
    def calls(self) -> list[tuple[str, dict[str, Any]]]:
        return self._calls

    @property
    def files(self) -> dict[str, bytes]:
        return self._fake_request.files

    @property
    def downloads(self) -> list[str]:
        return self._fake_request.downloads

    @property
    def reported_size(self) -> dict[str, int]:
        return self._reported_size

    @property
    def fail(self) -> dict[str, list[Exception]]:
        return self._fail

    def sent(self) -> list[dict[str, Any]]:
        return [d for e, d in self._calls if e == "sendMessage"]

    def endpoints(self) -> list[str]:
        return [e for e, _ in self._calls]


def fake_result(text: str, pending: dict[str, Any] | None = None, failed: bool = False) -> Any:
    """Shaped like `TurnResult` (only the fields the handlers read)."""
    return SimpleNamespace(text=text, failed=failed, pending=pending)


PENDING = {
    "tool": "delete_expense",
    "args_summary": "Kopi 18rb",
    "human_summary": "Hapus pengeluaran: Kopi <Rp18.000> (12 Okt)?",
    "tool_call_id": "c1",
}


class FakeRunner:
    """Stands in for TurnRunner; `reply` may be a str, or a coroutine function for slow turns.
    `pending` makes `run` answer with a paused-for-approval result; `resumes` is a queue of
    results for `resume` (default: an approval-agnostic "Selesai")."""

    def __init__(
        self,
        reply: str | Callable[[], Awaitable[str]] = "ok",
        boom: Exception | None = None,
        pending: dict[str, Any] | None = None,
        resumes: list[Any] | None = None,
        paused: dict[str, Any] | None = None,
    ) -> None:
        self.reply, self.boom, self.pending = reply, boom, pending
        self.paused = paused  # what `pending_approval` reports (a thread already waiting)
        self.saved: list[bytes] = []
        self.save_error: Exception | None = None
        self.resumes = resumes or []
        self.runs: list[tuple[str, str]] = []
        self.resume_calls: list[tuple[str, str]] = []
        self.resolved: list[int] = []

    async def resolve_user(self, telegram_user_id: int, display_name: str | None = None) -> Any:
        self.resolved.append(telegram_user_id)
        return object()

    async def pending_approval(self, thread_id: str, user: Any) -> dict[str, Any] | None:
        return self.paused

    async def save_receipt(self, user: Any, data: bytes) -> uuid.UUID:
        if self.save_error:
            raise self.save_error
        self.saved.append(data)
        return uuid.UUID(int=len(self.saved))

    async def run(self, thread_id: str, user: Any, message: str) -> Any:
        self.runs.append((thread_id, message))
        if self.boom:
            raise self.boom
        text = self.reply if isinstance(self.reply, str) else await self.reply()
        if self.pending:
            return fake_result(self.pending["human_summary"], self.pending)
        return fake_result(text)

    async def resume(
        self, thread_id: str, user: Any, decision: str, reason: str | None = None
    ) -> Any:
        self.resume_calls.append((thread_id, decision))
        if self.boom:
            raise self.boom
        return self.resumes.pop(0) if self.resumes else fake_result("Selesai")


_ids = itertools.count(1)


async def make_app(
    bot: FakeBot,
    runner: Any,
    allowed: list[int],
    limit: int = 20,
    clock: Any = None,
) -> App:
    app = Application.builder().bot(bot).build()
    app.bot_data["runner"] = runner
    register(app, allowed, limit, clock)
    await app.initialize()  # process_update refuses to run before this (calls getMe)
    bot.calls.clear()
    return app


def text_update(
    bot: FakeBot,
    text: str,
    user_id: int = 1,
    chat_id: int | None = None,
    chat_type: str = "private",
    with_user: bool = True,
) -> Update:
    msg: dict[str, Any] = {
        "message_id": next(_ids),
        "date": 0,
        "chat": {"id": user_id if chat_id is None else chat_id, "type": chat_type},
        "text": text,
    }
    if with_user:
        msg["from"] = {"id": user_id, "is_bot": False, "first_name": "Tester"}
    if text.startswith("/"):
        msg["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
    return Update.de_json({"update_id": next(_ids), "message": msg}, bot)


def callback_update(
    bot: FakeBot,
    data: str | None = "appr:y",
    user_id: int = 1,
    chat_id: int | None = None,
    chat_type: str = "private",
    text: str = "Hapus pengeluaran: Kopi Rp18.000 (12 Okt)?",
    accessible: bool = True,
) -> Update:
    """A button press on a bot message. `accessible=False` is a message too old to edit (Bot API
    sends date 0 and no text)."""
    chat = {"id": user_id if chat_id is None else chat_id, "type": chat_type}
    message: dict[str, Any] = {
        "message_id": next(_ids),
        "date": 1 if accessible else 0,
        "chat": chat,
    }
    if accessible:
        message["text"] = text
    query: dict[str, Any] = {
        "id": str(next(_ids)),
        "from": {"id": user_id, "is_bot": False, "first_name": "Tester"},
        "chat_instance": "ci",
        "message": message,
    }
    if data is not None:
        query["data"] = data
    return Update.de_json({"update_id": next(_ids), "callback_query": query}, bot)


def photo_update(
    bot: FakeBot,
    *,
    caption: str | None = None,
    user_id: int = 1,
    chat_type: str = "private",
    sizes: list[tuple[str, int, int, int | None]] | None = None,
) -> Update:
    """A photo message; `sizes` = (file_id, width, height, file_size), default small + large."""
    sizes = sizes or [("ph_s", 90, 90, 900), ("ph_l", 1280, 960, 9000)]
    msg: dict[str, Any] = {
        "message_id": next(_ids),
        "date": 0,
        "chat": {"id": user_id, "type": chat_type},
        "from": {"id": user_id, "is_bot": False, "first_name": "Tester"},
        "photo": [
            {
                "file_id": fid,
                "file_unique_id": f"u-{fid}",
                "width": w,
                "height": h,
                **({"file_size": sz} if sz is not None else {}),
            }
            for fid, w, h, sz in sizes
        ],
    }
    if caption is not None:
        msg["caption"] = caption
    return Update.de_json({"update_id": next(_ids), "message": msg}, bot)


def document_update(
    bot: FakeBot,
    *,
    mime: str | None,
    file_id: str = "doc1",
    file_size: int | None = 1000,
    user_id: int = 1,
    caption: str | None = None,
) -> Update:
    doc: dict[str, Any] = {"file_id": file_id, "file_unique_id": f"u-{file_id}"}
    if mime is not None:
        doc["mime_type"] = mime
    if file_size is not None:
        doc["file_size"] = file_size
    msg: dict[str, Any] = {
        "message_id": next(_ids),
        "date": 0,
        "chat": {"id": user_id, "type": "private"},
        "from": {"id": user_id, "is_bot": False, "first_name": "Tester"},
        "document": doc,
    }
    if caption is not None:
        msg["caption"] = caption
    return Update.de_json({"update_id": next(_ids), "message": msg}, bot)
