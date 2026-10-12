"""T-103: an audited call writes a tool_audit_log row; args redacted when LOG_CONTENT=false."""

import uuid
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.config import get_settings
from duitku.db.repositories import audit, users
from duitku.tools.base import audited, error, ok


async def _make_user(sm: async_sessionmaker[AsyncSession]) -> uuid.UUID:
    async with sm() as s:
        user, _ = await users.get_or_create_user(s, 12345)
        await s.commit()
        return user.id


@pytest.mark.parametrize("log_content", [False, True])
async def test_audited_call_writes_row(
    sm: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch, log_content: bool
) -> None:
    uid = await _make_user(sm)
    monkeypatch.setattr(get_settings(), "log_content", log_content)
    cfg: RunnableConfig = {"configurable": {"user_id": str(uid), "timezone": "Asia/Jakarta"}}

    @audited("demo", sessionmaker=sm)
    async def demo(note: str, config: RunnableConfig) -> dict[str, Any]:
        return error("NOT_FOUND", "nope", "hint") if note == "x" else ok({"n": 1})

    assert (await demo(note="makan di warteg", config=cfg))["ok"] is True
    assert (await demo(note="x", config=cfg))["ok"] is False

    async with sm() as s:
        rows = {r.error_code: r for r in await audit.list_tool_calls(s, uid)}
    assert set(rows) == {None, "NOT_FOUND"}
    assert rows[None].tool_name == "demo" and rows[None].status == "ok"
    assert rows["NOT_FOUND"].status == "error" and rows[None].latency_ms >= 0
    expected = {"note": "makan di warteg"} if log_content else {"note": "[redacted]"}
    assert rows[None].args == expected
