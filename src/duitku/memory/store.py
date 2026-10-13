"""Long-term memory store: save (with dedupe), recall, forget.

The caller owns the session and the transaction (like repositories); nothing here commits.
Embedding runs BEFORE any DB write, so an `EmbeddingError` leaves no partial state.
Thresholds are parameters; `memory.thresholds` reads the T-303 calibrated config values.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from duitku.db.models import EMBEDDING_DIM, Memory
from duitku.db.repositories import memories
from duitku.memory.embeddings import Embedder, EmbeddingError
from duitku.memory.validation import validate_content, validate_kind


@dataclass(frozen=True)
class SaveResult:
    memory: Memory
    created: bool  # False = an existing near-duplicate was updated in place


@dataclass(frozen=True)
class RecalledMemory:
    id: uuid.UUID
    kind: str
    content: str
    similarity: float


def _check_dim(vec: list[float]) -> list[float]:
    if len(vec) != EMBEDDING_DIM:
        raise EmbeddingError(f"embedding has dim {len(vec)}, expected {EMBEDDING_DIM}")
    return vec


async def save(
    session: AsyncSession,
    user_id: uuid.UUID,
    kind: str,
    content: str,
    embedder: Embedder,
    *,
    dedupe_threshold: float,
) -> SaveResult:
    kind = validate_kind(kind)
    content = validate_content(content)
    vec = _check_dim(await embedder.embed_passage(content))
    # Same kind only: updating keeps the row's kind, so crossing kinds would let a "goal" text
    # overwrite a "preference" (or relabel it). memory.md says "nearest neighbor for this user".
    near = await memories.nearest_memories(session, user_id, vec, limit=1, kind=kind)
    if near and near[0][1] >= dedupe_threshold:
        updated = await memories.update_memory(
            session, user_id, near[0][0].id, content=content, embedding=vec
        )
        if updated is not None:
            return SaveResult(updated, created=False)
    return SaveResult(await memories.add_memory(session, user_id, kind, content, vec), created=True)


async def recall(
    session: AsyncSession,
    user_id: uuid.UUID,
    query: str,
    embedder: Embedder,
    *,
    top_k: int,
    min_similarity: float,
) -> list[RecalledMemory]:
    """Top-k memories of this user with similarity >= min_similarity, best first.

    Bookkeeping (`recall_count`, `last_recalled_at`) is updated in this same call, in the caller's
    transaction, for the returned memories only.
    """
    text = query.strip()
    if not text or top_k < 1:
        return []
    vec = _check_dim(await embedder.embed_query(text))
    near = await memories.nearest_memories(session, user_id, vec, limit=top_k)
    hits = [
        RecalledMemory(m.id, m.kind, m.content, sim) for m, sim in near if sim >= min_similarity
    ]
    await memories.mark_recalled(session, user_id, [h.id for h in hits])
    return hits


async def forget(session: AsyncSession, user_id: uuid.UUID, memory_id: uuid.UUID) -> bool:
    """False if missing or not yours (indistinguishable)."""
    return await memories.delete_memory(session, user_id, memory_id)
