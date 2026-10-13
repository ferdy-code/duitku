"""Receipt retention: file deletion rules against tmp_path with a stub sessionmaker (no DB)."""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.ops import retention
from duitku.ops.retention import PurgeReport, cutoff_for, purge_expired_receipts

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


class _Result:
    def __init__(self, rows: list[tuple[uuid.UUID, str]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[uuid.UUID, str]]:
        return self._rows


class _Session:
    def __init__(self, rows: list[tuple[uuid.UUID, str]], seen: list[Any]) -> None:
        self._rows, self._seen = rows, seen

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *a: object) -> None:
        return None

    async def execute(self, stmt: Any) -> _Result:
        self._seen.append(stmt)
        return _Result(self._rows)


def _sm(paths: list[Path | str], seen: list[Any] | None = None) -> async_sessionmaker[AsyncSession]:
    rows = [(uuid.uuid4(), str(p)) for p in paths]
    return cast(
        "async_sessionmaker[AsyncSession]", lambda: _Session(rows, seen if seen is not None else [])
    )


async def _purge(root: Path, paths: list[Path | str]) -> PurgeReport:
    return await purge_expired_receipts(_sm(paths), NOW, root, 30)


def _file(root: Path, name: str = "a.jpg") -> Path:
    root.mkdir(exist_ok=True)
    p = root / name
    p.write_bytes(b"img")
    return p


async def test_deletes_file_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    f = _file(root)
    r1 = await _purge(root, [f])
    assert (r1.scanned, r1.deleted, r1.already_gone) == (1, 1, 0)
    assert not f.exists()
    r2 = await _purge(root, [f])  # second run: already purged, no error
    assert (r2.deleted, r2.already_gone, r2.errors) == (0, 1, 0)


@pytest.mark.parametrize("rel", ["../outside.jpg", "sub/../../outside.jpg"])
async def test_refuses_traversal(tmp_path: Path, rel: str) -> None:
    root = tmp_path / "receipts"
    (root / "sub").mkdir(parents=True)
    victim = tmp_path / "outside.jpg"
    victim.write_bytes(b"keep")
    r = await _purge(root, [root / rel])
    assert (r.refused, r.deleted) == (1, 0)
    assert victim.exists()


async def test_refuses_absolute_path_outside(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    root.mkdir()
    victim = _file(tmp_path, "secret.txt")
    r = await _purge(root, [victim])
    assert (r.refused, r.deleted) == (1, 0)
    assert victim.exists()


async def test_never_follows_symlinks_out(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    root.mkdir()
    victim = _file(tmp_path, "secret.txt")
    (root / "link.jpg").symlink_to(victim)  # file symlink
    (root / "dir").symlink_to(tmp_path, target_is_directory=True)  # dir symlink out
    r = await _purge(root, [root / "link.jpg", root / "dir" / "secret.txt"])
    assert (r.refused, r.deleted) == (2, 0)
    assert victim.exists() and (root / "link.jpg").is_symlink()


async def test_one_failure_does_not_abort_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "receipts"
    bad = root / "adir.jpg"
    bad.mkdir(parents=True)  # unlink on a directory -> OSError
    good = _file(root, "good.jpg")
    with caplog.at_level(logging.INFO):
        r = await _purge(root, [bad, good, root / "missing.jpg"])
    assert (r.scanned, r.deleted, r.errors, r.already_gone) == (3, 1, 1, 1)
    assert not good.exists()
    assert str(root) not in caplog.text  # ids and error class only, no paths


def test_cutoff_boundary() -> None:
    cutoff = cutoff_for(NOW, 30)
    assert cutoff == NOW - timedelta(days=30)
    exact, older = NOW - timedelta(days=30), NOW - timedelta(days=30, seconds=1)
    assert not exact < cutoff  # exactly N days old is kept (strict <)
    assert older < cutoff


async def test_query_uses_cutoff_param(tmp_path: Path) -> None:
    seen: list[Any] = []
    await purge_expired_receipts(_sm([], seen), NOW, tmp_path, 7)
    assert NOW - timedelta(days=7) in seen[0].compile().params.values()


def test_main_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from sqlalchemy.exc import OperationalError

    from duitku.config import ConfigError

    async def ok() -> PurgeReport:
        return PurgeReport(scanned=2, deleted=1, already_gone=1)

    def failing(exc: Exception) -> Any:
        async def f() -> PurgeReport:
            raise exc

        return f

    monkeypatch.setattr(retention, "purge_expired_receipts", ok)
    assert retention.main() == 0
    assert "deleted=1" in capsys.readouterr().out
    monkeypatch.setattr(retention, "purge_expired_receipts", failing(ConfigError("bad")))
    assert retention.main() == 2
    monkeypatch.setattr(
        retention, "purge_expired_receipts", failing(OperationalError("s", {}, Exception("pw")))
    )
    assert retention.main() == 1
    assert "pw" not in capsys.readouterr().err
