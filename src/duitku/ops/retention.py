"""Daily receipt retention (spec: interfaces-ops.md): delete image FILES older than
RECEIPT_RETENTION_DAYS; the `receipts` row and its `extraction` stay in the DB.

`receipts` has no "purged" column, so "file is gone" IS the purged marker: re-runs find the
file missing and count it `already_gone`. That makes the job idempotent without a migration.
shortcut: every run re-scans all receipts older than the cutoff (one stat each); fine at personal
scale, add `receipts.purged_at` + a partial index if this ever hurts.

Run: `python -m duitku.ops.retention`. systemd timer / cron example: docs/ops.md.
"""

import asyncio
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.config import ConfigError, get_settings
from duitku.db.models import Receipt

logger = logging.getLogger(__name__)


@dataclass
class PurgeReport:
    scanned: int = 0
    deleted: int = 0
    already_gone: int = 0
    refused: int = 0  # path outside RECEIPT_DIR, or a symlink
    errors: int = 0

    def summary(self) -> str:
        return (
            f"receipt retention: scanned={self.scanned} deleted={self.deleted} "
            f"already_gone={self.already_gone} refused={self.refused} errors={self.errors}"
        )


def cutoff_for(now: datetime, retention_days: int) -> datetime:
    """Receipts with created_at STRICTLY before this are expired (exactly N days old is kept)."""
    return now - timedelta(days=retention_days)


def _inside(file_path: str, root: Path) -> Path | None:
    """The file path if it sits directly under/inside `root`, else None.

    abspath normalises `..` without following links; resolving the PARENT catches a symlinked
    directory leading out. The file itself must not be a symlink (we never touch link targets).
    """
    p = Path(os.path.abspath(file_path))
    if not p.parent.resolve().is_relative_to(root.resolve()) or p.is_symlink():
        return None
    return p


def _purge_one(rid: uuid.UUID, file_path: str, root: Path, report: PurgeReport) -> None:
    try:
        p = _inside(file_path, root)
        if p is None:
            report.refused += 1
            logger.warning("retention: refused path outside receipt dir receipt_id=%s", rid)
        else:
            p.unlink()
            report.deleted += 1
    except FileNotFoundError:
        report.already_gone += 1
    except OSError as e:  # PermissionError, IsADirectoryError, ...
        report.errors += 1
        logger.error("retention: delete failed receipt_id=%s error=%s", rid, type(e).__name__)


async def purge_expired_receipts(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    now: datetime | None = None,
    receipt_dir: Path | None = None,
    retention_days: int | None = None,
) -> PurgeReport:
    """Raises DB errors (caller decides); per-file problems are counted, never raised."""
    if sessionmaker is None:
        from duitku.db.engine import get_sessionmaker  # lazy: needs DATABASE_URL

        sessionmaker = get_sessionmaker()
    if receipt_dir is None or retention_days is None:
        s = get_settings()
        receipt_dir = receipt_dir or s.receipt_dir
        retention_days = retention_days or s.receipt_retention_days
    cutoff = cutoff_for(now or datetime.now(UTC), retention_days)
    async with sessionmaker() as session:
        rows = (
            await session.execute(
                select(Receipt.id, Receipt.file_path).where(Receipt.created_at < cutoff)
            )
        ).all()
    report = PurgeReport(scanned=len(rows))
    for rid, file_path in rows:
        _purge_one(rid, file_path, receipt_dir, report)
    return report


def main() -> int:
    """Exit 0 even when some files failed (see the summary); non-zero only on config/DB failure."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = asyncio.run(purge_expired_receipts())
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    except (SQLAlchemyError, OSError) as e:  # connection refused surfaces as OSError/Operational
        print(f"receipt retention: database failure ({type(e).__name__})", file=sys.stderr)
        return 1
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
