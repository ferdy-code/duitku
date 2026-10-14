"""Store an uploaded receipt: image file on disk + `receipts` row for the user."""

import logging
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.config import get_settings
from duitku.db.engine import get_sessionmaker
from duitku.db.repositories import receipts
from duitku.receipts.storage import remove_quietly, save_image

logger = logging.getLogger(__name__)


async def ingest_receipt(
    user_id: uuid.UUID,
    data: bytes,
    *,
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    receipt_dir: Path | None = None,
    max_bytes: int | None = None,
) -> uuid.UUID:
    """Validate (magic bytes, size), write the file, insert the row; returns the receipt id.

    Raises `ReceiptRejected` for a bad upload. If the row cannot be written the file is removed
    again, so a failed insert leaves no orphan image behind.
    """
    s = get_settings()
    path = await save_image(data, receipt_dir or s.receipt_dir, max_bytes or s.receipt_max_bytes)
    try:
        async with (sessionmaker or get_sessionmaker())() as session:
            receipt_id = (await receipts.create_receipt(session, user_id, str(path))).id
            await session.commit()
            return receipt_id
    except BaseException:
        remove_quietly(path)
        raise
