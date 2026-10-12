"""T-206: get_exchange_rate against httpx.MockTransport (no network, no DB)."""

import json
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from duitku.tools import base, clock, get_exchange_rate
from duitku.tools.get_exchange_rate import build_get_exchange_rate_tool

CFG: RunnableConfig = {"configurable": {"user_id": str(uuid.uuid4()), "timezone": "Asia/Jakarta"}}
T0 = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
OK_BODY = '{"amount":1.0,"base":"USD","date":"2026-10-09","rates":{"IDR":16250.5}}'


class Fake:
    """Scripted HTTP responses (or exceptions); counts requests."""

    def __init__(self, *script: httpx.Response | Exception) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


def resp(status: int = 200, text: str = OK_BODY) -> httpx.Response:
    return httpx.Response(status, text=text)


@pytest.fixture
def now(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    t = [T0]
    monkeypatch.setattr(clock, "now", lambda: t[0])
    monkeypatch.setattr(base, "_write_audit", _noop)
    monkeypatch.setattr(base, "get_settings", lambda: SimpleNamespace(log_content=False))
    monkeypatch.setattr(
        get_exchange_rate, "get_settings", lambda: SimpleNamespace(tool_timeout_seconds=5)
    )
    return t


async def _noop(*_a: Any, **_k: Any) -> None:
    return None


def make(fake: Fake) -> StructuredTool:
    return build_get_exchange_rate_tool(
        base_url="http://fx.test", transport=httpx.MockTransport(fake)
    )


async def call(tool: StructuredTool, **args: Any) -> dict[str, Any]:
    msg = await tool.ainvoke({"type": "tool_call", "name": tool.name, "args": args, "id": "c"}, CFG)
    return json.loads(msg.content)


async def test_success_converts_with_half_up_rounding(now: list[datetime]) -> None:
    fake = Fake(resp())
    res = await call(make(fake), from_currency="usd", amount="3")  # 3 * 16250.5 = 48751.5
    assert res["ok"] is True
    d = res["data"]
    assert (d["from"], d["to"], d["rate"], d["rate_date"]) == (
        "USD",
        "IDR",
        "16250.5",
        "2026-10-09",
    )
    assert d["converted_idr"] == 48752 and isinstance(d["converted_idr"], int)
    assert d["cached"] is False
    req = fake.requests[0]
    assert req.url.params["base"] == "USD" and req.url.params["symbols"] == "IDR"


async def test_non_idr_target_two_decimals(now: list[datetime]) -> None:
    body = '{"date":"2026-10-09","rates":{"EUR":0.9234}}'
    res = await call(
        make(Fake(resp(text=body))), from_currency="USD", to_currency="EUR", amount="10"
    )
    assert res["data"]["converted"] == "9.23" and "converted_idr" not in res["data"]


async def test_timeout_then_retry_succeeds(now: list[datetime]) -> None:
    fake = Fake(httpx.ReadTimeout("slow"), resp())
    res = await call(make(fake), from_currency="USD")
    assert res["ok"] is True and len(fake.requests) == 2


async def test_5xx_then_retry_succeeds(now: list[datetime]) -> None:
    fake = Fake(resp(503, "down"), resp())
    assert (await call(make(fake), from_currency="USD"))["ok"] is True
    assert len(fake.requests) == 2


async def test_timeout_twice_is_error_envelope(now: list[datetime]) -> None:
    fake = Fake(httpx.ConnectTimeout("slow"))
    res = await call(make(fake), from_currency="USD")
    assert res["ok"] is False and res["error"]["code"] == "UPSTREAM_ERROR"
    assert len(fake.requests) == 2  # exactly one retry


async def test_persistent_5xx_is_error_envelope(now: list[datetime]) -> None:
    fake = Fake(resp(500, "boom"))
    res = await call(make(fake), from_currency="USD")
    assert res["error"]["code"] == "UPSTREAM_ERROR" and len(fake.requests) == 2


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "[]",
        '{"date":"2026-10-09"}',
        '{"date":"2026-10-09","rates":{}}',
        '{"date":"2026-10-09","rates":{"IDR":"abc"}}',
        '{"date":"2026-10-09","rates":{"IDR":0}}',
        '{"date":"2026-10-09","rates":{"IDR":-5}}',
        '{"date":"2026-10-09","rates":{"IDR":true}}',
    ],
)
async def test_malformed_response(now: list[datetime], body: str) -> None:
    res = await call(make(Fake(resp(text=body))), from_currency="USD")
    assert res["ok"] is False and res["error"]["code"] == "UPSTREAM_ERROR"


async def test_unknown_currency_from_api(now: list[datetime]) -> None:
    fake = Fake(resp(404, '{"message":"not found"}'))
    res = await call(make(fake), from_currency="XXX")
    assert res["error"]["code"] == "UNKNOWN_CURRENCY" and len(fake.requests) == 1  # no retry


@pytest.mark.parametrize("bad", ["US", "DOLLAR", "U$D", "12A", ""])
async def test_invalid_code_rejected_before_http(now: list[datetime], bad: str) -> None:
    fake = Fake(resp())
    res = await call(make(fake), from_currency=bad)
    assert res["error"]["code"] == "INVALID_ARGUMENT" and fake.requests == []


async def test_invalid_amount_rejected(now: list[datetime]) -> None:
    fake = Fake(resp())
    for amt in ("-5", "0", "abc"):
        res = await call(make(fake), from_currency="USD", amount=amt)
        assert res["error"]["code"] == "INVALID_ARGUMENT"
    assert fake.requests == []


async def test_cache_hit_then_expiry(now: list[datetime]) -> None:
    fake = Fake(resp())
    tool = make(fake)
    first = await call(tool, from_currency="USD")
    now[0] = T0 + timedelta(minutes=59)
    second = await call(tool, from_currency="usd", amount="2")
    assert len(fake.requests) == 1 and second["data"]["cached"] is True
    assert (
        second["data"]["rate"] == first["data"]["rate"] and second["data"]["converted_idr"] == 32501
    )
    now[0] = T0 + timedelta(hours=1, seconds=1)
    third = await call(tool, from_currency="USD")
    assert len(fake.requests) == 2 and third["data"]["cached"] is False


async def test_failures_are_not_cached(now: list[datetime]) -> None:
    fake = Fake(resp(500), resp(500), resp())
    tool = make(fake)
    assert (await call(tool, from_currency="USD"))["ok"] is False
    assert (await call(tool, from_currency="USD"))["ok"] is True


async def test_same_currency_needs_no_http(now: list[datetime]) -> None:
    fake = Fake(resp())
    res = await call(make(fake), from_currency="IDR", amount="1000")
    assert res["data"]["rate"] == "1" and res["data"]["converted_idr"] == 1000
    assert fake.requests == []
