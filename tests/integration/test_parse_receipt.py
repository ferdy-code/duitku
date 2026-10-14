"""T-404: receipt ingest + the parse_receipt tool against real Postgres, fake vision model.

Calls go through `ainvoke` with a tool-call dict (like the graph does). The vision model is a
fake, so nothing here touches the network.
"""

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.fake_vision import FakeVision, extraction

from duitku.db.models import Receipt, ToolAuditLog
from duitku.db.repositories import receipts, users
from duitku.receipts import ingest as ingest_mod
from duitku.receipts.ingest import ingest_receipt
from duitku.receipts.storage import ReceiptRejected
from duitku.tools import clock
from duitku.tools import parse_receipt as tool_mod
from duitku.tools.parse_receipt import build_parse_receipt_tool

SM = async_sessionmaker[AsyncSession]
JPEG = b"\xff\xd8\xff\xe0" + b"j" * 200
NOW = datetime(2026, 10, 14, 3, 0, tzinfo=UTC)  # 10:00 in Jakarta


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]
    fake = SimpleNamespace(
        receipt_dir=tmp_path / "receipts", receipt_max_bytes=10_000, log_content=False
    )
    monkeypatch.setattr(tool_mod, "get_settings", lambda: fake)
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: fake)
    monkeypatch.setattr(clock, "now", lambda: NOW)


def _resolved(p: Path) -> Path:
    return p.resolve()


def _read(p: Path) -> bytes:
    return p.read_bytes()


def _unlink(p: Path) -> None:
    p.unlink()


async def make_user(sm: SM, tg: int) -> uuid.UUID:
    async with sm() as s:
        user, _ = await users.get_or_create_user(s, tg)
        await s.commit()
        return user.id


async def call(sm: SM, vision: FakeVision, uid: uuid.UUID, receipt_id: str) -> dict[str, Any]:
    tool = build_parse_receipt_tool(sm, vision, None)
    config: RunnableConfig = {"configurable": {"user_id": str(uid), "timezone": "Asia/Jakarta"}}
    msg = await tool.ainvoke(
        {
            "type": "tool_call",
            "name": "parse_receipt",
            "args": {"receipt_id": receipt_id},
            "id": "c",
        },
        config,
    )
    return json.loads(msg.content)


async def stored_extraction(sm: SM, rid: uuid.UUID) -> dict[str, Any] | None:
    async with sm() as s:
        return (await s.execute(select(Receipt.extraction).where(Receipt.id == rid))).scalar_one()


# --- ingest ----------------------------------------------------------------------------------


async def test_ingest_stores_file_inside_receipt_dir_and_row_for_the_user(
    sm: SM, tmp_path: Path
) -> None:
    uid = await make_user(sm, 1)
    rid = await ingest_receipt(uid, JPEG, sessionmaker=sm)
    async with sm() as s:
        row = await receipts.get_receipt(s, uid, rid)
    assert row is not None and row.extraction is None
    p = Path(row.file_path)
    assert p.is_absolute() and p.parent == _resolved(tmp_path / "receipts")  # retention-safe
    assert p.suffix == ".jpg" and _read(p) == JPEG


async def test_ingest_rejects_bad_uploads_without_row_or_file(sm: SM, tmp_path: Path) -> None:
    uid = await make_user(sm, 1)
    with pytest.raises(ReceiptRejected):
        await ingest_receipt(uid, b"%PDF-1.4 nope", sessionmaker=sm)
    with pytest.raises(ReceiptRejected):
        await ingest_receipt(uid, JPEG + b"x" * 10_000, sessionmaker=sm)
    async with sm() as s:
        assert (await s.execute(select(func.count()).select_from(Receipt))).scalar_one() == 0
    assert not (tmp_path / "receipts").exists() or os.listdir(tmp_path / "receipts") == []


async def test_ingest_removes_the_file_when_the_row_cannot_be_written(
    sm: SM, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uid = await make_user(sm, 1)

    async def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(receipts, "create_receipt", boom)
    with pytest.raises(RuntimeError):
        await ingest_receipt(uid, JPEG, sessionmaker=sm)
    assert os.listdir(tmp_path / "receipts") == []  # no orphan image


# --- parse_receipt -----------------------------------------------------------------------------


async def test_parse_returns_extraction_stores_it_and_never_returns_image_data(sm: SM) -> None:
    uid = await make_user(sm, 1)
    rid = await ingest_receipt(uid, JPEG, sessionmaker=sm)
    vision = FakeVision(extraction(warnings=["slightly blurry"]))
    res = await call(sm, vision, uid, str(rid))
    assert res["ok"], res
    d = res["data"]
    assert d["receipt_id"] == str(rid)
    assert (d["merchant"], d["date"], d["total_idr"]) == ("Indomaret", "2026-10-13", 35_000)
    assert d["items"][0] == {"name": "Teh Botol", "qty": 2, "price_idr": 5_000}
    assert d["warnings"] == ["slightly blurry"]  # 2*5000 + 25000 == 35000: no sum warning
    assert vision.calls == [(JPEG, "image/jpeg")]  # the model got the real bytes ...
    raw = json.dumps(res)
    assert "base64" not in raw and "data:image" not in raw and len(raw) < 1500  # ... we return none
    saved = await stored_extraction(sm, rid)
    assert (
        saved is not None
        and saved["total_idr"] == 35_000
        and saved["warnings"] == ["slightly blurry"]
    )


async def test_sum_mismatch_warning_is_added_and_stored(sm: SM) -> None:
    uid = await make_user(sm, 1)
    rid = await ingest_receipt(uid, JPEG, sessionmaker=sm)
    vision = FakeVision(extraction(total_idr=40_000))
    res = await call(sm, vision, uid, str(rid))
    (w,) = res["data"]["warnings"]
    assert "35000" in w and "40000" in w
    saved = await stored_extraction(sm, rid)
    assert saved is not None and saved["warnings"] == [w]


async def test_future_receipt_date_is_flagged_using_the_users_timezone(sm: SM) -> None:
    uid = await make_user(sm, 1)
    rid = await ingest_receipt(uid, JPEG, sessionmaker=sm)
    res = await call(sm, FakeVision(extraction(date="2026-10-15")), uid, str(rid))
    assert any("future" in w for w in res["data"]["warnings"])


async def test_model_failure_is_an_error_observation_and_stores_nothing(sm: SM) -> None:
    uid = await make_user(sm, 1)
    rid = await ingest_receipt(uid, JPEG, sessionmaker=sm)
    res = await call(sm, FakeVision(error="ValidationError"), uid, str(rid))
    assert res["ok"] is False and res["error"]["code"] == "PARSE_FAILED"
    assert "ValidationError" not in json.dumps(res)  # no internals for the LLM
    assert await stored_extraction(sm, rid) is None


async def test_other_users_receipt_is_not_found_and_the_model_is_never_called(sm: SM) -> None:
    owner, intruder = await make_user(sm, 1), await make_user(sm, 2)
    rid = await ingest_receipt(owner, JPEG, sessionmaker=sm)
    vision = FakeVision()
    res = await call(sm, vision, intruder, str(rid))
    assert res["ok"] is False and res["error"]["code"] == "NOT_FOUND"
    assert vision.calls == [] and await stored_extraction(sm, rid) is None


@pytest.mark.parametrize("bad", ["not-a-uuid", str(uuid.uuid4()), ""])
async def test_unknown_or_malformed_id_is_not_found(sm: SM, bad: str) -> None:
    uid = await make_user(sm, 1)
    vision = FakeVision()
    res = await call(sm, vision, uid, bad)
    assert res["error"]["code"] == "NOT_FOUND" and vision.calls == []


async def test_missing_file_is_reported_as_unavailable(sm: SM) -> None:
    uid = await make_user(sm, 1)
    rid = await ingest_receipt(uid, JPEG, sessionmaker=sm)
    async with sm() as s:
        row = await receipts.require_receipt(s, uid, rid)
    _unlink(Path(row.file_path))  # e.g. purged by the retention job
    vision = FakeVision()
    res = await call(sm, vision, uid, str(rid))
    assert res["error"]["code"] == "FILE_UNAVAILABLE" and vision.calls == []


async def test_a_row_pointing_outside_the_receipt_dir_is_refused(sm: SM, tmp_path: Path) -> None:
    uid = await make_user(sm, 1)
    secret = tmp_path / "elsewhere.jpg"
    secret.write_bytes(JPEG)
    async with sm() as s:
        rid = (await receipts.create_receipt(s, uid, str(secret))).id
        await s.commit()
    vision = FakeVision()
    res = await call(sm, vision, uid, str(rid))
    assert res["error"]["code"] == "FILE_UNAVAILABLE" and vision.calls == []


async def test_call_is_audited_without_content(sm: SM) -> None:
    uid = await make_user(sm, 1)
    rid = await ingest_receipt(uid, JPEG, sessionmaker=sm)
    await call(sm, FakeVision(), uid, str(rid))
    async with sm() as s:
        rows = (await s.execute(select(ToolAuditLog).where(ToolAuditLog.user_id == uid))).scalars()
        (row,) = [r for r in rows if r.tool_name == "parse_receipt"]
    assert row.status == "ok" and str(rid) not in json.dumps(row.args)  # redacted (LOG_CONTENT off)
    async with sm() as s:
        n = (await s.execute(text("SELECT count(*) FROM expenses"))).scalar_one()
    assert n == 0  # parsing never records an expense
