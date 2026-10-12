"""Receipts: stored file path + parse_receipt output. Images never go into state/checkpoints."""

import uuid
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from duitku.db.models import Receipt
from duitku.domain.errors import NotOwnedError


async def create_receipt(session: AsyncSession, user_id: uuid.UUID, file_path: str) -> Receipt:
    r = Receipt(user_id=user_id, file_path=file_path)
    session.add(r)
    await session.flush()
    return r


async def get_receipt(
    session: AsyncSession, user_id: uuid.UUID, receipt_id: uuid.UUID
) -> Receipt | None:
    q = select(Receipt).where(Receipt.user_id == user_id, Receipt.id == receipt_id)
    return (await session.execute(q)).scalar_one_or_none()


async def require_receipt(
    session: AsyncSession, user_id: uuid.UUID, receipt_id: uuid.UUID
) -> Receipt:
    r = await get_receipt(session, user_id, receipt_id)
    if r is None:
        raise NotOwnedError("receipt")
    return r


async def set_extraction(
    session: AsyncSession, user_id: uuid.UUID, receipt_id: uuid.UUID, extraction: dict[str, Any]
) -> Receipt | None:
    q = (
        update(Receipt)
        .where(Receipt.user_id == user_id, Receipt.id == receipt_id)
        .values(extraction=extraction)
        .returning(Receipt)
    )
    return (await session.execute(q)).scalar_one_or_none()


async def delete_receipt(session: AsyncSession, user_id: uuid.UUID, receipt_id: uuid.UUID) -> bool:
    """False if missing/foreign. Raises IntegrityError if an expense still references it."""
    q = (
        delete(Receipt)
        .where(Receipt.user_id == user_id, Receipt.id == receipt_id)
        .returning(Receipt.id)
    )
    return (await session.execute(q)).scalar_one_or_none() is not None
