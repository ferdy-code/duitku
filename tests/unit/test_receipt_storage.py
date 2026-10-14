"""T-404: receipt file storage (magic bytes, size, location). Real files in tmp_path, no DB."""

import os
import stat
from pathlib import Path

import pytest

from duitku.receipts.storage import (
    ReceiptFileError,
    ReceiptRejected,
    detect_image_type,
    read_image,
    save_image,
    validate_upload,
)

JPEG = b"\xff\xd8\xff\xe0" + b"j" * 50
PNG = b"\x89PNG\r\n\x1a\n" + b"p" * 50
MAX = 1000


def _real(p: Path) -> str:
    return os.path.realpath(p)


def test_detect_by_magic_bytes() -> None:
    assert detect_image_type(JPEG) == "jpeg"
    assert detect_image_type(PNG) == "png"
    for other in (b"GIF89a....", b"%PDF-1.7", b"RIFF....WEBP", b"<html>", b"", b"\xff\xd8"):
        assert detect_image_type(other) is None


def test_validate_rejects_wrong_type_and_size() -> None:
    assert validate_upload(JPEG, MAX) == "jpeg"
    with pytest.raises(ReceiptRejected) as t:
        validate_upload(b"%PDF-1.7" + b"x" * 20, MAX)
    assert t.value.reason == "type"
    with pytest.raises(ReceiptRejected) as s:
        validate_upload(JPEG + b"x" * MAX, MAX)
    assert s.value.reason == "size"
    with pytest.raises(ReceiptRejected):
        validate_upload(b"", MAX)


async def test_save_writes_absolute_path_inside_dir_with_extension_from_content(
    tmp_path: Path,
) -> None:
    d = tmp_path / "receipts"
    jpg, png = await save_image(JPEG, d, MAX), await save_image(PNG, d, MAX)
    assert jpg.suffix == ".jpg" and png.suffix == ".png" and jpg != png
    for p, data in ((jpg, JPEG), (png, PNG)):
        assert p.is_absolute() and p.parent == d.resolve() and p.read_bytes() == data
    if os.name == "posix":
        assert stat.S_IMODE(jpg.stat().st_mode) == 0o600


async def test_relative_receipt_dir_still_gives_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    p = await save_image(PNG, Path("data/receipts"), MAX)
    assert p.is_absolute() and str(p).startswith(_real(tmp_path / "data" / "receipts"))


async def test_save_refuses_bad_upload_and_leaves_no_file(tmp_path: Path) -> None:
    with pytest.raises(ReceiptRejected):
        await save_image(b"not an image", tmp_path, MAX)
    with pytest.raises(ReceiptRejected):
        await save_image(JPEG + b"x" * MAX, tmp_path, MAX)
    assert os.listdir(tmp_path) == []


async def test_read_roundtrip(tmp_path: Path) -> None:
    p = await save_image(PNG, tmp_path, MAX)
    assert await read_image(str(p), tmp_path, MAX) == (PNG, "png")


async def test_read_missing_file_is_a_file_error(tmp_path: Path) -> None:
    with pytest.raises(ReceiptFileError):
        await read_image(str(tmp_path / "gone.jpg"), tmp_path, MAX)


async def test_read_refuses_paths_outside_the_receipt_dir(tmp_path: Path) -> None:
    inside, outside = tmp_path / "in", tmp_path / "out"
    inside.mkdir()
    outside.mkdir()
    secret = outside / "x.jpg"
    secret.write_bytes(JPEG)
    with pytest.raises(ReceiptFileError):
        await read_image(str(secret), inside, MAX)
    with pytest.raises(ReceiptFileError):  # `..` does not escape either
        await read_image(str(inside / ".." / "out" / "x.jpg"), inside, MAX)


@pytest.mark.skipif(os.name != "posix", reason="symlinks")
async def test_read_refuses_symlinks(tmp_path: Path) -> None:
    inside, outside = tmp_path / "in", tmp_path / "out"
    inside.mkdir()
    outside.mkdir()
    (outside / "x.jpg").write_bytes(JPEG)
    (inside / "link.jpg").symlink_to(outside / "x.jpg")
    with pytest.raises(ReceiptFileError):
        await read_image(str(inside / "link.jpg"), inside, MAX)


async def test_read_rechecks_type_and_size(tmp_path: Path) -> None:
    fake = tmp_path / "a.jpg"
    fake.write_bytes(b"<html>not an image</html>")
    with pytest.raises(ReceiptFileError):
        await read_image(str(fake), tmp_path, MAX)
    big = tmp_path / "b.jpg"
    big.write_bytes(JPEG + b"x" * MAX)
    with pytest.raises(ReceiptFileError):
        await read_image(str(big), tmp_path, MAX)
