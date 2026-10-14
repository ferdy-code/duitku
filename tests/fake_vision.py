"""Fake vision model: returns a canned extraction (or raises) and records what it was shown."""

from duitku.receipts.schema import ReceiptExtraction
from duitku.receipts.vision import VisionError


def extraction(**kw: object) -> ReceiptExtraction:
    base: dict[str, object] = {
        "merchant": "Indomaret",
        "date": "2026-10-13",
        "total_idr": 35_000,
        "items": [
            {"name": "Teh Botol", "qty": 2, "price_idr": 5_000},
            {"name": "Roti", "qty": 1, "price_idr": 25_000},
        ],
        "confidence": 0.93,
        "warnings": [],
    }
    return ReceiptExtraction.model_validate({**base, **kw})


class FakeVision:
    def __init__(self, result: ReceiptExtraction | None = None, error: str | None = None) -> None:
        self.result = result or extraction()
        self.error = error  # raise VisionError(error) instead of answering
        self.calls: list[tuple[bytes, str]] = []

    async def extract(self, image: bytes, mime: str) -> ReceiptExtraction:
        self.calls.append((image, mime))
        if self.error:
            raise VisionError(self.error)
        return self.result
