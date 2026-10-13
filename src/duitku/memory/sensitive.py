"""Regex/heuristic guard against storing secrets in long-term memory (second line of defence).

The first line is the system prompt. Regex cannot catch everything (spelled-out digits, a secret
with no keyword, ciphertext, images), so the goal is: catch the common, obvious shapes while
leaving normal finance text ("budget makan 1.500.000", "gaji tanggal 25", "Netflix 54000") alone.

Deliberately NOT blocked: email addresses and phone numbers (a preference may mention a contact),
bank names, bare 6-digit numbers without a PIN/OTP keyword.
Returns a short LABEL (never the matched text) so nothing sensitive reaches errors or logs.
"""

import re
import unicodedata
from typing import Final

_I = re.IGNORECASE
_DIGITS_NEAR = r"[^\d\n]{0,25}?"  # keyword ... a few filler chars ... digits

# digits joined by single spaces/dashes: "4111 1111 1111 1111", "4111-1111-..."; dots/commas
# break a run on purpose so amounts like 1.500.000 never join up.
_SEP_RUN = re.compile(r"(?<![\d.,])\d+(?:[ -]\d+)*(?![\d])")

# 8+ digits, optional single space/dash groups; no dots, so "15.000.000" is not one.
_ACCOUNT = re.compile(
    r"\b(?:rekening|rek|norek|no\.?\s*rek(?:ening)?|acc(?:ount|t)?(?:\s*(?:no|number|num))?)\b"
    r"(?P<gap>[^\d\n]{0,25}?)(?P<num>\d(?:[ -]?\d){7,})",
    re.IGNORECASE,
)

_KEYWORD_DIGITS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "ID number (NIK/KTP/NPWP/passport)",
        re.compile(
            r"\b(?:nik|ktp|npwp|paspor|passport|no\.?\s*kk|kartu\s+keluarga)\b"
            r"[^\d\nxX*]{0,25}?(?:[\dxX*][ .-]?){8,}",
            _I,
        ),
    ),
    (
        "CVV/CVC",
        re.compile(
            rf"\b(?:cv[vc]2?|cvn|csc|kode\s+keamanan|security\s+code)\b{_DIGITS_NEAR}\d{{3,4}}\b",
            _I,
        ),
    ),
    ("PIN", re.compile(rf"\bpin\b{_DIGITS_NEAR}\d{{4,8}}\b", _I)),
    (
        "OTP / verification code",
        re.compile(
            rf"\b(?:otp|kode\s+(?:otp|verifikasi|rahasia)|verification\s+code"
            rf"|one[- ]time\s+(?:code|password))\b{_DIGITS_NEAR}\d{{4,8}}\b",
            _I,
        ),
    ),
)

_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "password",
        re.compile(r"\b(?:password|passwd|pwd|passcode|passphrase|kata\s*sandi|katasandi)\b", _I),
    ),
    # "Sandi" is also a name ("utang ke Sandi"): only "sandi:" / "sandi m-banking" count.
    (
        "password",
        re.compile(
            r"\bsandi\b(?:\s*[:=]|\s+(?:m-?banking|atm|bank|akun|email|wi-?fi|login|aplikasi|app)\b)",
            _I,
        ),
    ),
    ("API key / token", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    (
        "API key / token",
        re.compile(r"\b(?:ghp|gho|ghs|glpat|xox[abp]|AKIA|AIza|ya29)[-_.A-Za-z0-9]{12,}"),
    ),
    ("API key / token", re.compile(r"\bbearer\s+[A-Za-z0-9._~+/-]{16,}", _I)),
    ("API key / token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.")),
    ("API key / token", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY")),
    # "token listrik 100rb" has a word between; only "token: <value>" / "api key = <value>" match.
    (
        "API key / token",
        re.compile(
            r"\b(?:api[-_ ]?key|secret(?:[-_ ]?key)?|(?:access|auth|refresh)[-_ ]?token|token"
            r"|private[-_ ]?key)"
            r"\s*[:=]\s*\S{8,}",
            _I,
        ),
    ),
    ("API key / token", re.compile(r"\b[0-9a-fA-F]{32,}\b")),
)

_LONG_WORD = re.compile(r"[A-Za-z0-9+/_=-]{32,}")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card_or_nik(text: str) -> bool:
    # 1) contiguous digits: exactly 16 is a card or NIK; 13-19 also if it passes Luhn.
    for m in re.finditer(r"(?<![\d])\d{13,19}(?![\d])", text):
        run = m.group()
        if len(run) == 16 or _luhn(run):
            return True
    # 2) grouped by space/dash: 4-4-4-4 always; otherwise card-sized groups (4-6 digits) + Luhn,
    # so dates ("2026-10-13 2026-10-14") and amount lists ("1500000 2000000") do not trip it.
    for m in _SEP_RUN.finditer(text):
        groups = re.split(r"[ -]", m.group())
        if len(groups) < 2 or any(not 4 <= len(g) <= 6 for g in groups[:-1]):
            continue
        if not 3 <= len(groups[-1]) <= 6:
            continue
        digits = "".join(groups)
        if 13 <= len(digits) <= 19 and ([len(g) for g in groups] == [4, 4, 4, 4] or _luhn(digits)):
            return True
    return False


def _account(text: str) -> bool:
    # "rekening listrik Rp 15000000" is an amount, not an account number.
    return any(not re.search(r"rp\.?\s*$", m["gap"], _I) for m in _ACCOUNT.finditer(text))


def find_sensitive(content: str) -> str | None:
    """Label of the first sensitive pattern found (e.g. "PIN"), else None. Never returns text."""
    # NFKC folds full-width digits/letters; zero-width chars could otherwise split a number.
    text = unicodedata.normalize("NFKC", content)
    text = re.sub(r"[​-‍⁠﻿]", "", text)
    if _card_or_nik(text):
        return "card or ID number"
    for label, pat in _KEYWORD_DIGITS:
        if pat.search(text):
            return label
    if _account(text):
        return "bank account number"
    for label, pat in _PATTERNS:
        if pat.search(text):
            return label
    # Long opaque letter+digit blob = probably a key; long words/URLs without digits pass.
    for m in _LONG_WORD.finditer(text):
        w = m.group()
        if any(c.isdigit() for c in w) and any(c.isalpha() for c in w):
            return "API key / token"
    return None
