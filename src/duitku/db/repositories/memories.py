"""Memories: CRUD, vector storage and cosine nearest-neighbour search.

This layer never computes vectors (embedding, dedupe and recall policy live in memory/store.py).
"""

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select, update
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
        .execution_options(populate_existing=True)  # else the identity map returns the old row
    )
    return (await session.execute(q)).scalar_one_or_none()


async def delete_memory(session: AsyncSession, user_id: uuid.UUID, memory_id: uuid.UUID) -> bool:
    q = delete(Memory).where(Memory.user_id == user_id, Memory.id == memory_id).returning(Memory.id)
    return (await session.execute(q)).scalar_one_or_none() is not None


async def nearest_memories(
    session: AsyncSession,
    user_id: uuid.UUID,
    embedding: Sequence[float],
    *,
    limit: int,
    kind: str | None = None,
) -> list[tuple[Memory, float]]:
    """Closest memories of this user, best first, as (memory, cosine similarity = 1 - `<=>`)."""
    dist = Memory.embedding.cosine_distance(list(embedding))
    q = select(Memory, dist).where(Memory.user_id == user_id)
    if kind is not None:
        q = q.where(Memory.kind == kind)
    rows = (await session.execute(q.order_by(dist, Memory.id).limit(limit))).all()
    return [(m, 1.0 - float(d)) for m, d in rows]


async def mark_recalled(
    session: AsyncSession, user_id: uuid.UUID, memory_ids: Sequence[uuid.UUID]
) -> None:
    if not memory_ids:
        return
    q = (
        update(Memory)
        .where(Memory.user_id == user_id, Memory.id.in_(list(memory_ids)))
        .values(recall_count=Memory.recall_count + 1, last_recalled_at=func.now())
        .execution_options(synchronize_session=False)
    )
    await session.execute(q)
