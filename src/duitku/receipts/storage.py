"""Receipt image files: type sniffing, size limits, safe write/read.

The type is decided by MAGIC BYTES, never by the file name or the MIME type the client claims.
`file_path` is stored ABSOLUTE and always inside RECEIPT_DIR, so the retention job (T-408) and
the reader here agree on it regardless of the process's working directory.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ImageType = Literal["jpeg", "png"]
MIME: dict[ImageType, str] = {"jpeg": "image/jpeg", "png": "image/png"}
_EXT: dict[ImageType, str] = {"jpeg": "jpg", "png": "png"}
_JPEG = b"\xff\xd8\xff"
_PNG = b"\x89PNG\r\n\x1a\n"


class ReceiptRejected(Exception):
    """The upload cannot be accepted. `reason` is safe to show the user (no content in it)."""

    def __init__(self, reason: Literal["type", "size"]) -> None:
        super().__init__(reason)
        self.reason = reason


class ReceiptFileError(Exception):
    """A stored receipt file cannot be used (gone, outside RECEIPT_DIR, wrong type, too big)."""


def detect_image_type(data: bytes) -> ImageType | None:
    if data.startswith(_JPEG):
        return "jpeg"
    if data.startswith(_PNG):
        return "png"
    return None


def validate_upload(data: bytes, max_bytes: int) -> ImageType:
    """Raise ReceiptRejected unless `data` is a non-empty JPEG/PNG within `max_bytes`."""
    if len(data) > max_bytes:
        raise ReceiptRejected("size")
    kind = detect_image_type(data)
    if kind is None:
        raise ReceiptRejected("type")
    return kind


def _write_new(receipt_dir: Path, name: str, data: bytes) -> Path:
    path = receipt_dir.resolve() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # "x": never overwrite; 0o600: receipts are private (best effort on Windows).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


async def save_image(data: bytes, receipt_dir: Path, max_bytes: int) -> Path:
    """Validate and write `data` to `<receipt_dir>/<uuid>.jpg|png`; returns the absolute path."""
    kind = validate_upload(data, max_bytes)
    return await asyncio.to_thread(_write_new, receipt_dir, f"{uuid.uuid4()}.{_EXT[kind]}", data)


def _read_checked(file_path: str, receipt_dir: Path, max_bytes: int) -> tuple[bytes, ImageType]:
    root = receipt_dir.resolve()
    p = Path(os.path.abspath(file_path))
    # Same guard as the retention job: parent must resolve inside the dir, never a symlink.
    if not p.parent.resolve().is_relative_to(root) or p.is_symlink():
        raise ReceiptFileError("path outside receipt dir")
    try:
        if p.stat().st_size > max_bytes:
            raise ReceiptFileError("file too large")
        data = p.read_bytes()
    except OSError as e:  # FileNotFoundError (purged by retention), PermissionError, ...
        raise ReceiptFileError(type(e).__name__) from None
    kind = detect_image_type(data)
    if kind is None:
        raise ReceiptFileError("not a jpeg/png")
    return data, kind


async def read_image(file_path: str, receipt_dir: Path, max_bytes: int) -> tuple[bytes, ImageType]:
    """Load a stored receipt image, re-checking location, size and magic bytes."""
    return await asyncio.to_thread(_read_checked, file_path, receipt_dir, max_bytes)


def remove_quietly(path: Path) -> None:
    """Best-effort cleanup of an orphan file (the DB row could not be written)."""
    try:
        path.unlink()
    except OSError as e:
        logger.warning("could not remove orphan receipt file (%s)", type(e).__name__)
