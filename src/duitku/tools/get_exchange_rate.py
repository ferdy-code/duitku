"""Tool `get_exchange_rate`: Frankfurter (https://frankfurter.dev, no key) with cache + retry.

Money rule: the rate is a Decimal (JSON parsed with parse_float=Decimal, never float). A
conversion to IDR is rounded ONCE, half-up, to whole rupiah. Other targets: 2 decimals, half-up.
"""

import json
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Annotated, Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BeforeValidator, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.config import get_settings
from duitku.tools import clock
from duitku.tools.base import Envelope, audited, error, ok, validation_error_envelope

BASE_URL = "https://api.frankfurter.dev/v1"
CACHE_TTL = timedelta(hours=1)

Currency = Annotated[
    str,
    BeforeValidator(lambda v: v.strip().upper() if isinstance(v, str) else v),
    Field(pattern=r"^[A-Z]{3}$", description="ISO 4217 code, e.g. USD."),
]

DESCRIPTION = (
    "Get the current exchange rate (and optionally convert an amount) between two currencies. "
    "Never convert yourself: use converted_idr / converted from the result. "
    "Example: from_currency='USD', to_currency='IDR', amount=25."
)


class Upstream(Exception):
    """FX API unavailable or answered garbage (message is safe to show the model)."""


class UnknownCurrency(Exception):
    pass


async def _fetch(
    client: httpx.AsyncClient, frm: str, to: str, base_url: str
) -> tuple[Decimal, str]:
    """One GET with ONE retry on timeout / network error / 5xx. Returns (rate, rate_date)."""
    resp: httpx.Response | None = None
    for _ in range(2):
        try:
            resp = await client.get(f"{base_url}/latest", params={"base": frm, "symbols": to})
        except httpx.TransportError:  # includes timeouts
            resp = None
        if resp is not None and resp.status_code < 500:
            break
        # shortcut: immediate retry, no backoff; add jitter if this ever hits a rate limit
    if resp is None or resp.status_code >= 500:
        raise Upstream("The exchange-rate service is not reachable right now.")
    if resp.status_code in (404, 422):
        raise UnknownCurrency
    if resp.status_code != 200:
        raise Upstream("The exchange-rate service rejected the request.")
    try:
        body = json.loads(resp.text, parse_float=Decimal)
        rate = Decimal(str(body["rates"][to]))
        rate_date = str(body["date"])
        if isinstance(body["rates"][to], bool) or not rate.is_finite() or rate <= 0:
            raise ValueError
    except (ValueError, KeyError, TypeError, InvalidOperation):
        raise Upstream("The exchange-rate service returned an unexpected response.") from None
    return rate, rate_date


def build_get_exchange_rate_tool(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    *,
    base_url: str = BASE_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> StructuredTool:
    cache: dict[tuple[str, str], tuple[datetime, Decimal, str]] = {}  # per tool instance

    async def get_exchange_rate(
        from_currency: Currency,
        config: RunnableConfig,
        to_currency: Currency = "IDR",
        amount: Annotated[
            Decimal | None,
            Field(gt=0, max_digits=15, decimal_places=2, description="Amount in from_currency."),
        ] = None,
    ) -> Envelope:
        now = clock.now()
        key = (from_currency, to_currency)
        hit = cache.get(key)
        if from_currency == to_currency:
            rate, rate_date, cached = Decimal(1), now.date().isoformat(), False
        elif hit and hit[0] > now:
            _, rate, rate_date = hit
            cached = True
        else:
            try:
                async with httpx.AsyncClient(
                    timeout=get_settings().tool_timeout_seconds, transport=transport
                ) as client:
                    rate, rate_date = await _fetch(client, from_currency, to_currency, base_url)
            except UnknownCurrency:
                return error(
                    "UNKNOWN_CURRENCY",
                    "Unsupported currency code.",
                    "Use a major ISO 4217 code such as USD, EUR, SGD, JPY, MYR.",
                )
            except Upstream as e:
                return error("UPSTREAM_ERROR", str(e), "Tell the user the rate is unavailable.")
            cache[key] = (now + CACHE_TTL, rate, rate_date)
            cached = False

        data: dict[str, Any] = {
            "from": from_currency,
            "to": to_currency,
            "rate": format(rate, "f"),  # string: never a float
            "rate_date": rate_date,
            "cached": cached,
        }
        if amount is not None:
            data["amount"] = format(amount, "f")
            if to_currency == "IDR":
                data["converted_idr"] = int((amount * rate).quantize(Decimal(1), ROUND_HALF_UP))
            else:
                data["converted"] = format(
                    (amount * rate).quantize(Decimal("0.01"), ROUND_HALF_UP), "f"
                )
        return ok(data)

    return StructuredTool.from_function(
        coroutine=audited("get_exchange_rate", sessionmaker=sessionmaker)(get_exchange_rate),
        name="get_exchange_rate",
        description=DESCRIPTION,
        handle_validation_error=validation_error_envelope,
    )
