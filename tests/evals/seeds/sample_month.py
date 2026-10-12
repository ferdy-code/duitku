"""Ground-truth data for query evals (docs/EVALS.md).

Makan & Minum: Oct = Rp1.200.000, Sep = Rp900.000. Transport: Oct = Rp450.000, Sep = Rp520.000.
Today (2026-10-12) holds Rp85.000 + Rp40.000 = Rp125.000.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SeedExpense:
    day: date
    category: str
    amount_idr: int
    merchant: str | None = None


EXPENSES: list[SeedExpense] = [
    SeedExpense(date(2026, 10, 1), "makan-minum", 300_000),
    SeedExpense(date(2026, 10, 3), "makan-minum", 250_000),
    SeedExpense(date(2026, 10, 6), "makan-minum", 320_000),
    SeedExpense(date(2026, 10, 9), "makan-minum", 245_000),
    SeedExpense(date(2026, 10, 12), "makan-minum", 85_000, "Warung Sederhana"),
    SeedExpense(date(2026, 10, 2), "transport", 150_000),
    SeedExpense(date(2026, 10, 7), "transport", 260_000),
    SeedExpense(date(2026, 10, 12), "transport", 40_000, "Grab"),
    SeedExpense(date(2026, 9, 5), "makan-minum", 400_000),
    SeedExpense(date(2026, 9, 15), "makan-minum", 300_000),
    SeedExpense(date(2026, 9, 25), "makan-minum", 200_000),
    SeedExpense(date(2026, 9, 8), "transport", 220_000),
    SeedExpense(date(2026, 9, 20), "transport", 300_000),
]
