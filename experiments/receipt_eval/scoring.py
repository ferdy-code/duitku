"""Scoring (pure): is the extracted total exactly right, does the merchant match, and would the
user have been warned when it was wrong.

`correct` = total exact AND merchant matches: that is the G3 definition (PRD: >= 85% correct,
total + merchant, on 20 real receipts).

Merchant match is deliberately forgiving about decoration, strict about identity: both names are
reduced to lowercase letters/digits; they match when equal, when one contains the other (>= 4
characters: "Indomaret" vs "INDOMARET PLAZA BARU"), or when they are >= 0.8 similar (typos in
OCR: "Alfarnart" vs "Alfamart").
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from duitku.receipts.checks import LOW_CONFIDENCE
from duitku.receipts.schema import ReceiptExtraction
from receipt_eval.truth import Truth

TARGET = 0.85
MIN_RECEIPTS = 20
SIMILARITY = 0.8
_NOISE = re.compile(r"\b(pt|cv|tbk|ud)\b")


def compact(name: str) -> str:
    """Lowercase letters and digits only, legal-form words dropped."""
    lowered = _NOISE.sub(" ", name.casefold())
    return re.sub(r"[^a-z0-9]+", "", lowered)


def merchant_match(predicted: str | None, truth: str) -> bool:
    if not predicted:
        return False
    a, b = compact(predicted), compact(truth)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 4 and short in long_:
        return True
    return SequenceMatcher(None, a, b).ratio() >= SIMILARITY


@dataclass(frozen=True)
class Row:
    filename: str
    truth_total: int
    truth_merchant: str
    predicted_total: int | None = None
    predicted_merchant: str | None = None
    confidence: float | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None  # model/file failure: counts as wrong in every metric

    @property
    def total_ok(self) -> bool:
        return self.error is None and self.predicted_total == self.truth_total

    @property
    def merchant_ok(self) -> bool:
        return self.error is None and merchant_match(self.predicted_merchant, self.truth_merchant)

    @property
    def correct(self) -> bool:
        return self.total_ok and self.merchant_ok

    @property
    def flagged(self) -> bool:
        """The user would have been warned (warnings, low confidence or a failure)."""
        low = self.confidence is not None and self.confidence < LOW_CONFIDENCE
        return self.error is not None or bool(self.warnings) or low


def score(truth: Truth, ex: ReceiptExtraction | None, error: str | None = None) -> Row:
    if ex is None:
        return Row(truth.filename, truth.total_idr, truth.merchant, error=error or "no result")
    return Row(
        truth.filename,
        truth.total_idr,
        truth.merchant,
        predicted_total=ex.total_idr,
        predicted_merchant=ex.merchant,
        confidence=ex.confidence,
        warnings=tuple(ex.warnings),
    )


@dataclass(frozen=True)
class Summary:
    n: int
    total_ok: int
    merchant_ok: int
    correct: int
    errors: int
    wrong: int
    wrong_flagged: int  # wrong rows the user would have been warned about
    notes: list[str] = field(default_factory=lambda: list[str]())

    def rate(self, count: int) -> float:
        return count / self.n if self.n else 0.0

    @property
    def meets_target(self) -> bool:
        return self.n >= MIN_RECEIPTS and self.rate(self.correct) >= TARGET


def summarize(rows: list[Row]) -> Summary:
    wrong = [r for r in rows if not r.correct]
    notes: list[str] = []
    if len(rows) < MIN_RECEIPTS:
        notes.append(f"only {len(rows)} receipts: the AC needs >= {MIN_RECEIPTS} real ones")
    return Summary(
        n=len(rows),
        total_ok=sum(r.total_ok for r in rows),
        merchant_ok=sum(r.merchant_ok for r in rows),
        correct=sum(r.correct for r in rows),
        errors=sum(r.error is not None for r in rows),
        wrong=len(wrong),
        wrong_flagged=sum(r.flagged for r in wrong),
        notes=notes,
    )
