"""Indonesian rupiah amount parsing. Pure; Decimal (never float) for intermediate math."""

import re
from decimal import Decimal

MAX_IDR = 10_000_000_000
_MAX_INPUT_LEN = 64

_MULTIPLIERS: dict[str, int] = {
    "rb": 1_000,
    "ribu": 1_000,
    "k": 1_000,
    "jt": 1_000_000,
    "juta": 1_000_000,
    "miliar": 1_000_000_000,
    "milyar": 1_000_000_000,
}

_AMOUNT_RE = re.compile(
    r"^(?:rp\.?|idr)?\s*(?P<neg>-)?\s*(?P<num>\d[\d.,]*)\s*(?P<unit>[a-z]+)?$", re.ASCII
)
_PLAIN_RE = re.compile(r"^\d+$", re.ASCII)
_DECIMAL_RE = re.compile(r"^\d+(?:[.,]\d+)?$", re.ASCII)
_THOUSANDS_RE = re.compile(r"^\d{1,3}([.,])\d{3}(?:\1\d{3})*$", re.ASCII)


class AmountError(ValueError):
    """Unparseable/unsafe amount. Tools map this to an error envelope (code, message, hint)."""

    def __init__(self, code: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


def _ambiguous(text: str) -> AmountError:
    return AmountError(
        "AMBIGUOUS_AMOUNT",
        f"Cannot read amount {text!r} unambiguously.",
        "Ask the user to restate the amount, e.g. '35rb', '1,5 juta' or '35.000'.",
    )


def _invalid(text: str, why: str) -> AmountError:
    return AmountError(
        "INVALID_AMOUNT",
        f"Invalid amount {text!r}: {why}.",
        f"Amount must be between 1 and {MAX_IDR:,} rupiah; ask the user to confirm.",
    )


def parse_idr_amount(text: str) -> int:
    """Parse text like '35rb', '1,5 juta', 'Rp 1.250.000' into integer rupiah.

    Raises AmountError: AMBIGUOUS_AMOUNT (never guesses) or INVALID_AMOUNT (garbage, <=0, huge).
    """
    if len(text) > _MAX_INPUT_LEN:
        raise _invalid(text[:20] + "...", "too long")
    m = _AMOUNT_RE.match(text.strip().lower())
    if m is None:
        raise _invalid(text, "not a number")
    num, unit = m["num"], m["unit"]

    if unit is None:
        if _PLAIN_RE.match(num):
            value = Decimal(num)
        elif _THOUSANDS_RE.match(num):
            value = Decimal(num.replace(".", "").replace(",", ""))
        else:  # '1.5', '1.2500', mixed separators: decimal or thousands? don't guess
            raise _ambiguous(text)
    elif unit in _MULTIPLIERS:
        if not _DECIMAL_RE.match(num):
            raise _ambiguous(text)
        value = Decimal(num.replace(",", ".")) * _MULTIPLIERS[unit]
        if value != value.to_integral_value():  # e.g. '1,2345rb' = 1234.5 rupiah
            raise _ambiguous(text)
    else:  # includes bare 'm' (million or billion?) and unknown words
        raise _ambiguous(text)

    if m["neg"] or value <= 0:
        raise _invalid(text, "must be positive")
    if value > MAX_IDR:
        raise _invalid(text, "too large")
    return int(value)
