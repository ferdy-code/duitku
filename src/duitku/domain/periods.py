"""Deterministic period resolution (R-04). Pure: `now` is injected, no clock or I/O here."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

PERIODS = (
    "today",
    "yesterday",
    "this_week",
    "last_week",
    "this_month",
    "last_month",
    "last_7_days",
    "last_30_days",
    "this_year",
)

_LABELS = {
    "today": "Hari ini",
    "yesterday": "Kemarin",
    "this_week": "Minggu ini",
    "last_week": "Minggu lalu",
    "this_month": "Bulan ini",
    "last_month": "Bulan lalu",
    "last_7_days": "7 hari terakhir",
    "last_30_days": "30 hari terakhir",
    "this_year": "Tahun ini",
}


class InvalidPeriodError(ValueError):
    """Unknown period name; the tool layer maps this to INVALID_ARGUMENT with `hint`."""

    def __init__(self, period: str) -> None:
        self.hint = f"period must be one of: {', '.join(PERIODS)}"
        super().__init__(f"Unknown period {period!r}. {self.hint}")


@dataclass(frozen=True, slots=True)
class Period:
    """Inclusive date range in the user's timezone (same convention as `query_expenses`)."""

    start_date: date
    end_date: date
    label: str


def resolve_period(period: str, now: datetime, tz: str | ZoneInfo) -> Period:
    """Resolve `period` to inclusive local dates. `now` must be timezone-aware.

    `this_*` periods run from the period start up to and including today (not future days).
    Weeks start on Monday. `last_N_days` includes today.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    zone = tz if isinstance(tz, ZoneInfo) else ZoneInfo(tz)
    today = now.astimezone(zone).date()
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)

    match period:
        case "today":
            start, end = today, today
        case "yesterday":
            start = end = today - timedelta(days=1)
        case "this_week":
            start, end = week_start, today
        case "last_week":
            start, end = week_start - timedelta(days=7), week_start - timedelta(days=1)
        case "this_month":
            start, end = month_start, today
        case "last_month":
            end = month_start - timedelta(days=1)
            start = end.replace(day=1)
        case "last_7_days":
            start, end = today - timedelta(days=6), today
        case "last_30_days":
            start, end = today - timedelta(days=29), today
        case "this_year":
            start, end = today.replace(month=1, day=1), today
        case _:
            raise InvalidPeriodError(period)
    return Period(start, end, _LABELS[period])
