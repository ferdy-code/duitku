"""Memories: plain CRUD + vector column storage only.

Embedding, dedupe and similarity recall are T-302 (memory/); this layer never computes vectors.
"""

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from duitku.db.models import Memory


async def add_memory(
    session: AsyncSession, user_id: uuid.UUID, kind: str, content: str, embedding: Sequence[float]
) -> Memory:
    m = Memory(user_id=user_id, kind=kind, content=content, embedding=list(embedding))
    session.add(m)
    await session.flush()
    return m


async def get_memory(
    session: AsyncSession, user_id: uuid.UUID, memory_id: uuid.UUID
) -> Memory | None:
    q = select(Memory).where(Memory.user_id == user_id, Memory.id == memory_id)
    return (await session.execute(q)).scalar_one_or_none()


async def list_memories(session: AsyncSession, user_id: uuid.UUID) -> Sequence[Memory]:
    q = select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at.desc())
    return (await session.execute(q)).scalars().all()


async def update_memory(
    session: AsyncSession,
    user_id: uuid.UUID,
    memory_id: uuid.UUID,
    *,
    content: str,
    embedding: Sequence[float],
) -> Memory | None:
    """Content and its embedding must change together, hence both are required."""
    values: dict[str, Any] = {"content": content, "embedding": list(embedding)}
    q = (
        update(Memory)
        .where(Memory.user_id == user_id, Memory.id == memory_id)
        .values(**values)
        .returning(Memory)
    )
    return (await session.execute(q)).scalar_one_or_none()


async def delete_memory(session: AsyncSession, user_id: uuid.UUID, memory_id: uuid.UUID) -> bool:
    q = delete(Memory).where(Memory.user_id == user_id, Memory.id == memory_id).returning(Memory.id)
    return (await session.execute(q)).scalar_one_or_none() is not None
