"""Ground truth for the receipt accuracy run: one row per image, typed by hand from the paper
receipt (NOT from the model output).

CSV (header required):                      JSON (list of objects, same keys):
    filename,merchant,total_idr                 [{"filename": "r01.jpg", "merchant": "Indomaret",
    r01.jpg,Indomaret,35000                       "total_idr": 35000}]
`total_idr` is a plain integer of rupiah (no "Rp", no separators): it is what the receipt says
was PAID. Anything unparseable is an error, never silently skipped.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TruthError(ValueError):
    """The ground-truth file is malformed (message names the row, never leaks the file)."""


@dataclass(frozen=True)
class Truth:
    filename: str
    merchant: str
    total_idr: int


def _row(raw: dict[str, Any], where: str) -> Truth:
    try:
        filename = str(raw["filename"]).strip()
        merchant = str(raw["merchant"]).strip()
        total = raw["total_idr"]
    except KeyError as e:
        raise TruthError(f"{where}: missing column {e}") from None
    if not filename or not merchant:
        raise TruthError(f"{where}: filename and merchant must not be empty")
    try:
        total_idr = int(str(total).strip())
    except ValueError:
        raise TruthError(f"{where}: total_idr must be a plain integer, got {total!r}") from None
    if total_idr <= 0:
        raise TruthError(f"{where}: total_idr must be > 0")
    return Truth(filename, merchant, total_idr)


def load_truth(path: Path) -> list[Truth]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data: Any = json.loads(text)
        if not isinstance(data, list):
            raise TruthError("JSON truth must be a list of objects")
        items: list[dict[str, Any]] = [d for d in data if isinstance(d, dict)]  # pyright: ignore[reportUnknownVariableType]
        if len(items) != len(data):  # pyright: ignore[reportUnknownArgumentType]
            raise TruthError("JSON truth must be a list of objects")
    else:
        items = list(csv.DictReader(text.splitlines()))
    rows = [_row(d, f"row {i}") for i, d in enumerate(items, start=1)]
    names = [r.filename for r in rows]
    if len(set(names)) != len(names):
        raise TruthError("duplicate filenames in the truth file")
    if not rows:
        raise TruthError("the truth file has no rows")
    return rows
