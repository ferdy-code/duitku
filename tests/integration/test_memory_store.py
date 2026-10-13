"""T-302: memory store against the real test DB (pgvector) with a controllable fake embedder."""

import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.fake_embedder import FakeEmbedder, unit

from duitku.db.models import Memory
from duitku.db.repositories import memories, users
from duitku.memory import store
from duitku.memory.embeddings import EmbeddingError
from duitku.memory.validation import InvalidMemoryError

SM = async_sessionmaker[AsyncSession]
DEDUPE = 0.9

# cosine to unit(1.0) (= e0) is exactly the number; "far" is orthogonal.
EMB = FakeEmbedder(
    {
        "q": unit(1.0),
        "kopi = makan-minum": unit(0.99),
        "kopi itu kategori makan-minum": unit(0.97, axis=1),  # sim to previous ~ 0.99+: duplicate
        "gojek = transport": unit(0.0, axis=2),
        "m95": unit(0.95, axis=3),
        "m80": unit(0.80, axis=4),
        "m60": unit(0.60, axis=5),
        "m30": unit(0.30, axis=6),
    }
)


async def _user(sm: SM, tg: int) -> uuid.UUID:
    async with sm() as s:
        u, _ = await users.get_or_create_user(s, tg)
        await s.commit()
        return u.id


async def _count(sm: SM) -> int:
    async with sm() as s:
        return (await s.execute(select(func.count()).select_from(Memory))).scalar_one()


async def _save(sm: SM, uid: uuid.UUID, content: str, kind: str = "preference"):
    async with sm() as s:
        r = await store.save(s, uid, kind, content, EMB, dedupe_threshold=DEDUPE)
        await s.commit()
        return r


async def test_near_duplicate_updates_one_row(sm: SM) -> None:
    uid = await _user(sm, 1)
    first = await _save(sm, uid, "kopi = makan-minum")
    second = await _save(sm, uid, "kopi itu kategori makan-minum")
    assert first.created and not second.created
    assert second.memory.id == first.memory.id
    assert second.memory.created_at == first.memory.created_at
    assert await _count(sm) == 1
    async with sm() as s:
        row = await memories.get_memory(s, uid, first.memory.id)
        assert row is not None and row.content == "kopi itu kategori makan-minum"


async def test_different_content_inserts_second_row(sm: SM) -> None:
    uid = await _user(sm, 1)
    await _save(sm, uid, "kopi = makan-minum")
    r = await _save(sm, uid, "gojek = transport")
    assert r.created and await _count(sm) == 2


async def test_dedupe_is_per_user_and_per_kind(sm: SM) -> None:
    a, b = await _user(sm, 1), await _user(sm, 2)
    ra = await _save(sm, a, "kopi = makan-minum")
    rb = await _save(sm, b, "kopi = makan-minum")  # identical text, other user: must not merge
    rk = await _save(sm, a, "kopi = makan-minum", kind="goal")  # same text, other kind: new row
    assert rb.created and rk.created and rb.memory.id != ra.memory.id
    assert await _count(sm) == 3


async def test_recall_filters_by_user_threshold_topk_and_orders(sm: SM) -> None:
    a, b = await _user(sm, 1), await _user(sm, 2)
    for c in ("m30", "m95", "m60", "m80"):
        await _save(sm, a, c)
    await _save(sm, b, "m95")  # user B's near-identical memory
    async with sm() as s:
        hits = await store.recall(s, a, "q", EMB, top_k=3, min_similarity=0.5)
        await s.commit()
    assert [h.content for h in hits] == ["m95", "m80", "m60"]  # m30 < min_similarity
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)
    assert sims == pytest.approx([0.95, 0.80, 0.60], abs=1e-6)
    async with sm() as s:
        owners = {m.user_id for m in await memories.list_memories(s, a)}
        assert owners == {a}
        top1 = await store.recall(s, a, "q", EMB, top_k=1, min_similarity=0.0)
        assert [h.content for h in top1] == ["m95"]
        assert await store.recall(s, a, "q", EMB, top_k=3, min_similarity=0.99) == []


async def test_recall_updates_bookkeeping_only_for_returned(sm: SM) -> None:
    uid = await _user(sm, 1)
    hit = (await _save(sm, uid, "m95")).memory.id
    miss = (await _save(sm, uid, "m30")).memory.id
    for _ in range(2):
        async with sm() as s:
            await store.recall(s, uid, "q", EMB, top_k=3, min_similarity=0.5)
            await s.commit()
    async with sm() as s:
        h, m = await memories.get_memory(s, uid, hit), await memories.get_memory(s, uid, miss)
        assert h and m
        assert h.recall_count == 2 and h.last_recalled_at is not None
        assert m.recall_count == 0 and m.last_recalled_at is None


async def test_forget_and_cross_user_isolation(sm: SM) -> None:
    a, b = await _user(sm, 1), await _user(sm, 2)
    mine = (await _save(sm, a, "m95")).memory.id
    theirs = (await _save(sm, b, "m95")).memory.id
    async with sm() as s:
        assert await store.forget(s, a, theirs) is False  # foreign == missing
        assert await store.forget(s, a, uuid.uuid4()) is False
        # update by id across users is also "not found" and leaves B untouched
        assert (
            await memories.update_memory(s, a, theirs, content="pwned", embedding=unit(1.0)) is None
        )
        await s.commit()
    async with sm() as s:
        row = await memories.get_memory(s, b, theirs)
        assert row is not None and row.content == "m95"
        assert await store.forget(s, a, mine) is True
        await s.commit()
    assert await _count(sm) == 1


async def test_invalid_input_rejected_before_db_and_embedder(sm: SM) -> None:
    uid = await _user(sm, 1)
    EMB.calls.clear()
    async with sm() as s:
        for kind, content in (("nonsense", "m95"), ("fact", "   "), ("fact", "x" * 301)):
            with pytest.raises(InvalidMemoryError):
                await store.save(s, uid, kind, content, EMB, dedupe_threshold=DEDUPE)
    assert EMB.calls == [] and await _count(sm) == 0


async def test_embedding_failure_propagates_without_partial_write(sm: SM) -> None:
    uid = await _user(sm, 1)
    await _save(sm, uid, "m95")
    bad = FakeEmbedder({"m60": unit(0.6)}, fail=True)
    async with sm() as s:
        with pytest.raises(EmbeddingError):
            await store.save(s, uid, "fact", "m60", bad, dedupe_threshold=DEDUPE)
        with pytest.raises(EmbeddingError):
            await store.recall(s, uid, "m60", bad, top_k=3, min_similarity=0.0)
    assert await _count(sm) == 1


async def test_dimension_mismatch_caught(sm: SM) -> None:
    uid = await _user(sm, 1)
    short = FakeEmbedder({"m60": unit(0.6)}, dim=3)
    async with sm() as s:
        with pytest.raises(EmbeddingError, match="dim 3"):
            await store.save(s, uid, "fact", "m60", short, dedupe_threshold=DEDUPE)
        with pytest.raises(EmbeddingError, match="dim 3"):
            await store.recall(s, uid, "m60", short, top_k=3, min_similarity=0.0)
    assert await _count(sm) == 0


async def test_query_uses_cosine_operator_and_hnsw_index(sm: SM) -> None:
    uid = await _user(sm, 1)
    for c in ("m95", "m80", "m60"):
        await _save(sm, uid, c)
    async with sm() as s:
        near = await memories.nearest_memories(s, uid, unit(1.0), limit=3)
        assert [m.content for m, _ in near] == ["m95", "m80", "m60"]
        # With seqscan disabled the planner must pick the HNSW index for ORDER BY <=> ... LIMIT.
        await s.execute(text("SET LOCAL enable_seqscan = off"))
        plan = (
            (
                await s.execute(
                    text(
                        "EXPLAIN SELECT id FROM memories "
                        "ORDER BY embedding <=> (SELECT embedding FROM memories LIMIT 1) LIMIT 3"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert any("memories_embedding_idx" in line for line in plan), plan
