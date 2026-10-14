"""Deterministic checks on an extraction (pure; the LLM never does this arithmetic).

`finalize` adds the warnings the model cannot be trusted to raise itself: an items sum that does
not match the total, an unreadable total, a date in the future. Warnings are plain English notes
that the agent relays to the user before saving anything.
"""

import datetime as dt

from duitku.receipts.schema import MAX_ITEMS, MAX_WARNINGS, ReceiptExtraction

LOW_CONFIDENCE = 0.6


def items_sum(ex: ReceiptExtraction) -> int:
    return sum(i.qty * i.price_idr for i in ex.items)


def finalize(ex: ReceiptExtraction, today: dt.date) -> ReceiptExtraction:
    """Return a copy with our warnings appended and size caps applied."""
    warnings = list(ex.warnings)
    if ex.total_idr == 0:
        warnings.append("Total could not be read.")
    if ex.items and ex.total_idr > 0 and items_sum(ex) != ex.total_idr:
        warnings.append(
            f"Items sum to {items_sum(ex)} but the total is {ex.total_idr} "
            "(tax, service charge or discounts may explain the difference)."
        )
    if ex.date is not None and ex.date > today:
        warnings.append(f"Receipt date {ex.date.isoformat()} is in the future.")
    if ex.confidence < LOW_CONFIDENCE:
        warnings.append("Low confidence in this reading.")
    return ex.model_copy(
        update={"items": ex.items[:MAX_ITEMS], "warnings": warnings[:MAX_WARNINGS]}
    )
