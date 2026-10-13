"""Offline Telegram doubles: a real PTB `ExtBot` whose HTTP layer is replaced, so the real
Application / handler chain / Message objects run without any network."""

import itertools
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from telegram import Update
from telegram.ext import Application, ExtBot

from duitku.interfaces.telegram.app import App, register

TOKEN = "123456:TEST-TOKEN-NOT-REAL"


class FakeBot(ExtBot[None]):
    """Records every API call in `calls` as (endpoint, data). `fail` maps an endpoint to a list
    of exceptions raised one per call (then calls succeed)."""

    def __init__(self) -> None:
        super().__init__(TOKEN)
        self._calls: list[tuple[str, dict[str, Any]]] = []
        self._fail: dict[str, list[Exception]] = {}

    async def _do_post(self, endpoint: str, data: dict[str, Any], **_: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        self._calls.append((endpoint, dict(data)))
        if self._fail.get(endpoint):
            raise self._fail[endpoint].pop(0)
        if endpoint == "getMe":
            return {"id": 1, "is_bot": True, "first_name": "B", "username": "b_bot"}
        if endpoint == "sendMessage":
            return {"message_id": 1, "date": 0, "chat": {"id": data["chat_id"], "type": "private"}}
        return True

    @property
    def calls(self) -> list[tuple[str, dict[str, Any]]]:
        return self._calls

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
    ) -> None:
        self.reply, self.boom, self.pending = reply, boom, pending
        self.resumes = resumes or []
        self.runs: list[tuple[str, str]] = []
        self.resume_calls: list[tuple[str, str]] = []
        self.resolved: list[int] = []

    async def resolve_user(self, telegram_user_id: int, display_name: str | None = None) -> Any:
        self.resolved.append(telegram_user_id)
        return object()

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
