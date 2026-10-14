"""`ReceiptExtraction`: what the vision model must return (docs/spec/tools.md, tool 6).

Everything in here was READ FROM A PHOTO, so it is untrusted data: strings are length-capped and
single-line, numbers are bounds-checked, and nothing here is ever treated as an instruction.
Money is integer rupiah (no floats); `price_idr` is the UNIT price.
"""

import datetime as dt
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

MAX_TOTAL_IDR = 10_000_000_000  # same ceiling as parse_idr_amount
MAX_ITEMS = 100  # enforced by `checks.finalize` (truncation), not by the schema: a long receipt
MAX_WARNINGS = 20  # must not make the whole extraction fail


def _one_line(max_len: int) -> BeforeValidator:
    """Collapse whitespace/control characters (a receipt must not smuggle newlines into prompts)
    and cap the length."""

    def clean(v: object) -> object:
        if isinstance(v, str):
            return " ".join(v.split())[:max_len]
        return v

    return BeforeValidator(clean)


Line = Annotated[str, _one_line(200)]


class ReceiptItem(BaseModel):
    name: Line = Field(description="Item name as printed.")
    qty: int = Field(
        default=1,
        ge=1,
        le=10_000,
        description="Whole quantity. Weighed or fractional items: qty 1 and price = line total.",
    )
    price_idr: int = Field(
        ge=0, le=MAX_TOTAL_IDR, description="UNIT price in whole rupiah (not the line total)."
    )


class ReceiptExtraction(BaseModel):
    """Structured reading of one receipt photo. Use null / 0 for anything not legible."""

    model_config = ConfigDict(extra="ignore")

    merchant: Line | None = Field(default=None, description="Store or restaurant name as printed.")
    date: dt.date | None = Field(
        default=None, description="Purchase date, ISO YYYY-MM-DD; null if missing or ambiguous."
    )
    total_idr: int = Field(
        ge=0,
        le=MAX_TOTAL_IDR,
        description="Amount actually paid in whole rupiah (grand total). 0 if unreadable.",
    )
    items: list[ReceiptItem] = Field(default_factory=lambda: list[ReceiptItem]())
    confidence: float = Field(
        ge=0, le=1, description="0..1: how sure you are that merchant and total are right."
    )
    warnings: list[Line] = Field(
        default_factory=lambda: list[Line](),
        description="Short English notes on anything unclear: blur, crop, handwriting, ...",
    )
