"""Retention against a real (test) DB: boundary at exactly N days, row + extraction kept."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.db.models import Receipt
from duitku.db.repositories import receipts, users
from duitku.ops.retention import purge_expired_receipts

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


async def test_purge_boundary_and_row_kept(
    sm: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    root = tmp_path / "receipts"
    root.mkdir()
    ages = {"old": timedelta(days=30, seconds=1), "edge": timedelta(days=30), "new": timedelta(1)}
    async with sm() as s:
        u, _ = await users.get_or_create_user(s, 1)
        uid = u.id
        for name, age in ages.items():
            (root / f"{name}.jpg").write_bytes(b"x")
            r = await receipts.create_receipt(s, uid, str(root / f"{name}.jpg"))
            await receipts.set_extraction(s, uid, r.id, {"total": 1})
            await s.execute(update(Receipt).where(Receipt.id == r.id).values(created_at=NOW - age))
        await s.commit()
    rep = await purge_expired_receipts(sm, NOW, root, 30)
    assert rep.deleted == 1
    assert not (root / "old.jpg").exists()
    assert (root / "edge.jpg").exists() and (root / "new.jpg").exists()
    async with sm() as s:
        rows = (await s.execute(select(Receipt))).scalars().all()
        assert len(rows) == 3 and all(r.extraction == {"total": 1} for r in rows)
