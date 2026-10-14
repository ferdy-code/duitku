"""T-404: a receipt photo through the Telegram handlers with the REAL TurnRunner, DB, graph,
parse_receipt and add_expense (scripted agent model, fake vision model, in-memory checkpointer):

photo -> file + receipts row -> parse_receipt -> add_expense(source="receipt") -> approval buttons
-> Setuju inserts ONE expense linked to the receipt / Batal inserts none.
Also proves images never reach the model's context or the checkpoint.
"""

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.fake_embedder import FakeEmbedder
from tests.fake_vision import FakeVision, extraction
from tests.telegram_fakes import FakeBot, callback_update, make_app, photo_update

from duitku.agent import graph as graph_mod
from duitku.agent.graph import build_graph
from duitku.agent.nodes import repo_slug_loader
from duitku.db.models import Expense, Receipt
from duitku.interfaces.runner import TurnRunner
from duitku.interfaces.telegram import receipts as tg_receipts
from duitku.interfaces.telegram import texts
from duitku.receipts import ingest as ingest_mod
from duitku.tools import parse_receipt as tool_mod
from duitku.tools.registry import build_tools

if TYPE_CHECKING:
    from tests.conftest import Responder, ScriptedChat

TG = 42
JKT = ZoneInfo("Asia/Jakarta")
# A "photo": PNG magic + 30 KB of non-trivial bytes, so a leak is easy to spot in any blob.
IMAGE = b"\x89PNG\r\n\x1a\n" + bytes((i * 7 + 3) % 256 for i in range(30_000))
IMAGE_B64_HEAD = "iVBORw0KGgo"  # base64 of the PNG magic: every base64 form of IMAGE starts so


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]
    app = SimpleNamespace(max_agent_steps=8, short_term_max_tokens=6000, log_content=False)
    monkeypatch.setattr(graph_mod, "get_settings", lambda: app)
    rec = SimpleNamespace(
        receipt_dir=tmp_path / "receipts", receipt_max_bytes=100_000, log_content=False
    )
    for mod in (tg_receipts, ingest_mod, tool_mod):
        monkeypatch.setattr(mod, "get_settings", lambda: rec)


def _script(final: str) -> Callable[[list[BaseMessage]], AIMessage]:
    """parse_receipt -> add_expense(source=receipt) -> final text; the receipt id is read from
    the user's marker message exactly as the real model would."""
    step = iter(range(10))

    def respond(messages: list[BaseMessage]) -> AIMessage:
        human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        rid = re.search(r"receipt_id=([0-9a-f-]{36})", str(human.content))
        assert rid, "the marker message must carry the receipt id"
        n = next(step)
        if n == 0:
            return AIMessage(
                "",
                tool_calls=[{"id": "p1", "name": "parse_receipt", "args": {"receipt_id": rid[1]}}],
            )
        if n == 1:
            args = {
                "amount": "35000",
                "category": "makan-minum",
                "merchant": "Indomaret",
                "occurred_on": "2026-10-13",
                "source": "receipt",
                "receipt_id": rid[1],
            }
            return AIMessage("", tool_calls=[{"id": "a1", "name": "add_expense", "args": args}])
        return AIMessage(final)

    return respond


def _all_bytes(obj: Any) -> Iterator[bytes]:
    """Every bytes/str leaf inside a nested checkpoint-saver structure."""
    if isinstance(obj, bytes):
        yield obj
    elif isinstance(obj, str):
        yield obj.encode()
    elif isinstance(obj, dict):
        pairs = cast(dict[Any, Any], obj)  # nested saver structure: untyped by design
        for k, v in pairs.items():
            yield from _all_bytes(k)
            yield from _all_bytes(v)
    elif isinstance(obj, (list, tuple, set)):
        leaves = cast(list[Any], obj)
        for v in leaves:
            yield from _all_bytes(v)


def _read(p: Path) -> bytes:
    return p.read_bytes()


async def _expenses(sm: async_sessionmaker[AsyncSession]) -> list[Expense]:
    async with sm() as s:
        return list(
            (await s.execute(select(Expense).where(Expense.deleted_at.is_(None)))).scalars()
        )


async def _start(
    sm: async_sessionmaker[AsyncSession],
    scripted_llm: "Callable[[Responder], ScriptedChat]",
    tmp_path: Path,
    final: str,
) -> tuple[FakeBot, Any, "ScriptedChat", InMemorySaver, FakeVision]:
    saver, vision = InMemorySaver(), FakeVision(extraction())
    llm = scripted_llm(_script(final))
    graph = build_graph(
        llm=llm,
        tools=build_tools(sm, vision=vision, receipt_dir=tmp_path / "receipts"),
        checkpointer=saver,
        category_slugs=repo_slug_loader(sm),
        sessionmaker=sm,
        embedder=FakeEmbedder({}, fail=True),
    )
    bot = FakeBot()
    bot.files["ph_l"] = IMAGE
    app = await make_app(bot, TurnRunner(graph, sm), allowed=[TG])
    await app.process_update(photo_update(bot, user_id=TG, caption="makan siang"))
    return bot, app, llm, saver, vision


async def test_photo_pauses_for_approval_with_nothing_saved_yet(
    sm: async_sessionmaker[AsyncSession],
    scripted_llm: "Callable[[Responder], ScriptedChat]",
    tmp_path: Path,
) -> None:
    bot, _, _, _, vision = await _start(sm, scripted_llm, tmp_path, "Sudah dicatat.")
    (question,) = bot.sent()
    assert question["text"] == "Catat pengeluaran dari struk: Rp35.000 di Indomaret (makan-minum)?"
    assert [b.callback_data for r in question["reply_markup"].inline_keyboard for b in r] == [
        "appr:y",
        "appr:n",
    ]
    assert await _expenses(sm) == []  # R-08: nothing before the button
    assert vision.calls and vision.calls[0][0] == IMAGE  # the vision model saw the real image
    async with sm() as s:
        (receipt,) = (await s.execute(select(Receipt))).scalars()
    assert receipt.extraction is not None and receipt.extraction["total_idr"] == 35_000
    assert _read(Path(receipt.file_path)) == IMAGE and "receipts" in receipt.file_path


async def test_approve_inserts_exactly_one_expense_linked_to_the_receipt(
    sm: async_sessionmaker[AsyncSession],
    scripted_llm: "Callable[[Responder], ScriptedChat]",
    tmp_path: Path,
) -> None:
    bot, app, _, _, _ = await _start(sm, scripted_llm, tmp_path, "Sudah kucatat: Rp35.000.")
    await app.process_update(callback_update(bot, "appr:y", user_id=TG))
    async with sm() as s:
        (receipt,) = (await s.execute(select(Receipt))).scalars()
    (e,) = await _expenses(sm)
    assert (e.amount_idr, e.source, e.merchant) == (35_000, "receipt", "Indomaret")
    assert (
        e.receipt_id == receipt.id
        and e.occurred_at.astimezone(JKT).date().isoformat() == "2026-10-13"
    )
    assert bot.sent()[-1]["text"] == "Sudah kucatat: Rp35.000."
    await app.process_update(callback_update(bot, "appr:y", user_id=TG))  # stale second press
    assert len(await _expenses(sm)) == 1


async def test_reject_inserts_nothing_but_keeps_the_receipt(
    sm: async_sessionmaker[AsyncSession],
    scripted_llm: "Callable[[Responder], ScriptedChat]",
    tmp_path: Path,
) -> None:
    bot, app, _, _, _ = await _start(sm, scripted_llm, tmp_path, "Oke, tidak jadi dicatat.")
    await app.process_update(callback_update(bot, "appr:n", user_id=TG))
    assert await _expenses(sm) == []
    assert bot.sent()[-1]["text"] == "Oke, tidak jadi dicatat."
    async with sm() as s:
        assert (await s.execute(text("SELECT count(*) FROM receipts"))).scalar_one() == 1


async def test_images_never_enter_the_model_context_or_the_checkpoint(
    sm: async_sessionmaker[AsyncSession],
    scripted_llm: "Callable[[Responder], ScriptedChat]",
    tmp_path: Path,
) -> None:
    bot, app, llm, saver, _ = await _start(sm, scripted_llm, tmp_path, "Sudah kucatat.")

    def assert_clean() -> None:
        blobs = list(_all_bytes([saver.storage, saver.writes, saver.blobs]))
        assert blobs, "the checkpoint must hold something, or this test proves nothing"
        for b in blobs:
            assert IMAGE[:1000] not in b and IMAGE_B64_HEAD.encode() not in b
            assert b"data:image" not in b and b"base64" not in b
        assert sum(len(b) for b in blobs) < len(IMAGE)  # the whole checkpoint is smaller than it
        for view in llm.seen:  # everything the agent model was ever shown
            blob = json.dumps([str(m.content) for m in view])
            assert IMAGE_B64_HEAD not in blob and "data:image" not in blob

    assert_clean()  # while paused at the approval
    await app.process_update(callback_update(bot, "appr:y", user_id=TG))
    assert_clean()  # after the turn finished
    assert any("receipt_id=" in str(m.content) for v in llm.seen for m in v)  # marker was seen


async def test_second_photo_while_approval_is_pending_is_not_stored(
    sm: async_sessionmaker[AsyncSession],
    scripted_llm: "Callable[[Responder], ScriptedChat]",
    tmp_path: Path,
) -> None:
    bot, app, _, _, vision = await _start(sm, scripted_llm, tmp_path, "ok")
    await app.process_update(photo_update(bot, user_id=TG))
    reminder, question = bot.sent()[-2:]
    assert reminder["text"] == texts.RECEIPT_WHILE_PENDING
    assert question["text"].startswith("Catat pengeluaran dari struk") and question["reply_markup"]
    async with sm() as s:
        assert (await s.execute(text("SELECT count(*) FROM receipts"))).scalar_one() == 1
    assert len(vision.calls) == 1
