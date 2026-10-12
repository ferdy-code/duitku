from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from duitku.domain import periods
from duitku.domain.periods import PERIODS, InvalidPeriodError, Period, resolve_period

JKT = "Asia/Jakarta"


def at(y: int, m: int, d: int, hh: int = 12, mm: int = 0, tz: str = JKT) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(tz))


def r(period: str, now: datetime, tz: str = JKT) -> tuple[date, date]:
    p = resolve_period(period, now, tz)
    return p.start_date, p.end_date


# 2025-03-12 is a Wednesday.
@pytest.mark.parametrize(
    ("period", "start", "end"),
    [
        ("today", date(2025, 3, 12), date(2025, 3, 12)),
        ("yesterday", date(2025, 3, 11), date(2025, 3, 11)),
        ("this_week", date(2025, 3, 10), date(2025, 3, 12)),
        ("last_week", date(2025, 3, 3), date(2025, 3, 9)),
        ("this_month", date(2025, 3, 1), date(2025, 3, 12)),
        ("last_month", date(2025, 2, 1), date(2025, 2, 28)),
        ("last_7_days", date(2025, 3, 6), date(2025, 3, 12)),
        ("last_30_days", date(2025, 2, 11), date(2025, 3, 12)),
        ("this_year", date(2025, 1, 1), date(2025, 3, 12)),
    ],
)
def test_every_period(period: str, start: date, end: date) -> None:
    assert r(period, at(2025, 3, 12)) == (start, end)


def test_labels_cover_every_period() -> None:
    assert all(resolve_period(p, at(2025, 3, 12), JKT).label for p in PERIODS)
    assert len(PERIODS) == 9


def test_return_shape() -> None:
    p = resolve_period("today", at(2025, 3, 12), JKT)
    assert p == Period(date(2025, 3, 12), date(2025, 3, 12), "Hari ini")


def test_year_rollover_january() -> None:
    now = at(2025, 1, 15)
    assert r("last_month", now) == (date(2024, 12, 1), date(2024, 12, 31))
    assert r("this_year", now) == (date(2025, 1, 1), date(2025, 1, 15))
    assert r("last_30_days", now) == (date(2024, 12, 17), date(2025, 1, 15))


def test_week_spanning_new_year() -> None:
    # Wed 2025-01-01: this week started Mon 2024-12-30.
    now = at(2025, 1, 1)
    assert r("this_week", now) == (date(2024, 12, 30), date(2025, 1, 1))
    assert r("last_week", now) == (date(2024, 12, 23), date(2024, 12, 29))


def test_last_week_of_year() -> None:
    # Tue 2024-12-31
    now = at(2024, 12, 31)
    assert r("this_week", now) == (date(2024, 12, 30), date(2024, 12, 31))
    assert r("this_month", now) == (date(2024, 12, 1), date(2024, 12, 31))


def test_leap_february() -> None:
    assert r("last_month", at(2024, 3, 10)) == (date(2024, 2, 1), date(2024, 2, 29))
    assert r("last_month", at(2025, 3, 10)) == (date(2025, 2, 1), date(2025, 2, 28))
    assert r("yesterday", at(2024, 3, 1)) == (date(2024, 2, 29), date(2024, 2, 29))
    assert r("yesterday", at(2025, 3, 1)) == (date(2025, 2, 28), date(2025, 2, 28))


def test_last_month_from_31st_does_not_overflow() -> None:
    assert r("last_month", at(2025, 3, 31)) == (date(2025, 2, 1), date(2025, 2, 28))


def test_monday_start_weeks() -> None:
    sunday, monday = at(2025, 3, 16), at(2025, 3, 17)
    assert r("this_week", sunday) == (date(2025, 3, 10), date(2025, 3, 16))
    assert r("last_week", sunday) == (date(2025, 3, 3), date(2025, 3, 9))
    assert r("this_week", monday) == (date(2025, 3, 17), date(2025, 3, 17))
    assert r("last_week", monday) == (date(2025, 3, 10), date(2025, 3, 16))


def test_timezone_edge_utc_evening_is_next_day_in_jakarta() -> None:
    now = datetime(2025, 3, 11, 23, 30, tzinfo=UTC)  # 06:30 on 03-12 in Jakarta (UTC+7)
    assert r("today", now) == (date(2025, 3, 12), date(2025, 3, 12))
    assert r("today", now, "UTC") == (date(2025, 3, 11), date(2025, 3, 11))


def test_timezone_edge_month_boundary() -> None:
    now = datetime(2025, 2, 28, 18, 0, tzinfo=UTC)  # 01:00 on 03-01 in Jakarta
    assert r("this_month", now) == (date(2025, 3, 1), date(2025, 3, 1))
    assert r("last_month", now) == (date(2025, 2, 1), date(2025, 2, 28))


def test_accepts_zoneinfo_object() -> None:
    now = at(2025, 3, 12)
    assert resolve_period("today", now, ZoneInfo(JKT)).start_date == date(2025, 3, 12)


def test_invalid_period() -> None:
    with pytest.raises(InvalidPeriodError) as exc:
        resolve_period("next_week", at(2025, 3, 12), JKT)
    assert "this_week" in exc.value.hint
    assert isinstance(exc.value, ValueError)


def test_naive_now_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_period("today", datetime(2025, 3, 12, 12), JKT)


def test_no_direct_clock_in_source() -> None:
    src = Path(periods.__file__).read_text()
    for banned in ("datetime.now", "date.today", "utcnow", "time.time"):
        assert banned not in src
