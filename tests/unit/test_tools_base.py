"""T-103: envelope, context extraction, exception mapping, redaction (no DB)."""

import json
import logging
import uuid
from types import SimpleNamespace
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, StructuredTool
from pydantic import BaseModel, ValidationError

from duitku.tools import base
from duitku.tools.base import (
    ContextError,
    audited,
    error,
    get_context,
    ok,
    redact_args,
    validation_error_envelope,
)

UID = uuid.uuid4()
CFG: RunnableConfig = {"configurable": {"user_id": str(UID), "timezone": "Asia/Jakarta"}}


@pytest.fixture
def audit_rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    async def fake(_sm: object, user_id: uuid.UUID, **fields: Any) -> None:
        rows.append({"user_id": user_id, **fields})

    monkeypatch.setattr(base, "_write_audit", fake)
    monkeypatch.setattr(base, "get_settings", lambda: SimpleNamespace(log_content=False))
    return rows


def test_envelopes() -> None:
    assert ok({"a": 1}) == {"ok": True, "data": {"a": 1}}
    assert error("X", "m", "h") == {
        "ok": False,
        "error": {"code": "X", "message": "m", "hint": "h"},
    }


def test_context_ok() -> None:
    ctx = get_context(CFG)
    assert ctx.user_id == UID and ctx.timezone.key == "Asia/Jakarta"


@pytest.mark.parametrize(
    "cfg",
    [
        {},
        {"configurable": {}},
        {"configurable": {"timezone": "Asia/Jakarta"}},
        {"configurable": {"user_id": "nope", "timezone": "Asia/Jakarta"}},
        {"configurable": {"user_id": str(UID), "timezone": "Mars/Base"}},
    ],
)
def test_context_missing_or_invalid(cfg: Any) -> None:
    with pytest.raises(ContextError):
        get_context(cfg)


def test_redaction() -> None:
    args = {"note": "makan di warteg", "amount": 35000}
    assert redact_args(args, False) == {"note": "[redacted]", "amount": "[redacted]"}
    assert redact_args({"d": uuid.UUID(int=1)}, True) == {"d": str(uuid.UUID(int=1))}


async def test_ok_passthrough_and_audit(audit_rows: list[dict[str, Any]]) -> None:
    @audited("t")
    async def t(x: int, config: RunnableConfig) -> dict[str, Any]:
        return ok({"x": x})

    assert await t(x=3, config=CFG) == ok({"x": 3})
    (row,) = audit_rows
    assert row["tool_name"] == "t" and row["status"] == "ok" and row["error_code"] is None
    assert row["user_id"] == UID and row["args"] == {"x": "[redacted]"}


async def test_business_error_is_observation(audit_rows: list[dict[str, Any]]) -> None:
    @audited("t")
    async def t(config: RunnableConfig) -> dict[str, Any]:
        return error("NOT_FOUND", "no", "try X")

    assert (await t(config=CFG))["error"]["code"] == "NOT_FOUND"
    assert audit_rows[0]["status"] == "error" and audit_rows[0]["error_code"] == "NOT_FOUND"


async def test_unexpected_exception_is_generic(
    audit_rows: list[dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    @audited("t")
    async def t(config: RunnableConfig) -> dict[str, Any]:
        raise RuntimeError("secret-db-password boom")

    with caplog.at_level(logging.ERROR):
        res = await t(config=CFG)
    assert res["error"]["code"] == "INTERNAL_ERROR"
    assert "secret-db-password" not in json.dumps(res)
    assert "RuntimeError" in caplog.text and "secret-db-password" not in caplog.text
    assert audit_rows[0]["error_code"] == "INTERNAL_ERROR"


async def test_value_error_is_invalid_argument(audit_rows: list[dict[str, Any]]) -> None:
    @audited("t")
    async def t(config: RunnableConfig) -> dict[str, Any]:
        raise ValueError("amount must be positive")

    err = (await t(config=CFG))["error"]
    assert err["code"] == "INVALID_ARGUMENT" and err["hint"] and "positive" in err["message"]


async def test_pydantic_validation_error_has_hint(audit_rows: list[dict[str, Any]]) -> None:
    class M(BaseModel):
        n: int

    @audited("t")
    async def t(config: RunnableConfig) -> dict[str, Any]:
        M.model_validate({"n": "SECRETVALUE"})
        return ok()

    err = (await t(config=CFG))["error"]
    assert err["code"] == "INVALID_ARGUMENT" and err["hint"]
    assert "n:" in err["message"] and "SECRETVALUE" not in err["message"]


async def test_missing_context_is_internal_error(audit_rows: list[dict[str, Any]]) -> None:
    @audited("t")
    async def t(config: RunnableConfig) -> dict[str, Any]:
        return ok()

    assert (await t(config={}))["error"]["code"] == "INTERNAL_ERROR"
    assert audit_rows == []


async def test_audit_failure_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def boom(*_a: object, **_k: object) -> None:
        raise ConnectionError("db down")

    monkeypatch.setattr(base, "_write_audit", boom)
    monkeypatch.setattr(base, "get_settings", lambda: SimpleNamespace(log_content=False))

    @audited("t")
    async def t(config: RunnableConfig) -> dict[str, Any]:
        return ok(1)

    with caplog.at_level(logging.ERROR):
        assert await t(config=CFG) == ok(1)
    assert "audit write failed" in caplog.text


async def test_langchain_integration(audit_rows: list[dict[str, Any]]) -> None:
    """config/tool_call_id are injected, hidden from the LLM schema; arg errors become envelopes."""

    @audited("add")
    async def _add(
        amount: int, config: RunnableConfig, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> dict[str, Any]:
        """Add. Example: amount=3"""
        return ok({"amount": amount, "id": tool_call_id})

    add = StructuredTool.from_function(
        coroutine=_add, name="add", handle_validation_error=validation_error_envelope
    )
    assert set(add.args) == {"amount"}
    good = await add.ainvoke(
        {"type": "tool_call", "name": "add", "args": {"amount": 3}, "id": "c1"}, CFG
    )
    assert json.loads(good.content) == ok({"amount": 3, "id": "c1"})
    bad = await add.ainvoke(
        {"type": "tool_call", "name": "add", "args": {"amount": "x"}, "id": "c2"}, CFG
    )
    body = json.loads(bad.content)
    assert body["error"]["code"] == "INVALID_ARGUMENT" and body["error"]["hint"]
    assert bad.tool_call_id == "c2"


def test_validation_error_envelope_direct() -> None:
    class M(BaseModel):
        n: int

    with pytest.raises(ValidationError) as ei:
        M.model_validate({"n": "x"})
    assert json.loads(validation_error_envelope(ei.value))["error"]["code"] == "INVALID_ARGUMENT"
