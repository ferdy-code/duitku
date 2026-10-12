"""T-301: embedding service without downloading a model (fake loader)."""

import asyncio
import subprocess
import sys
import threading
import time

import pytest

from duitku.memory import embeddings
from duitku.memory.embeddings import (
    EmbeddingError,
    SentenceTransformerEmbedder,
    check_embedding_dim,
)

DIM = 4


class _Out:
    def __init__(self, rows: list[list[float]]):
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


class FakeEncoder:
    def __init__(self, dim: int = DIM, delay: float = 0.0, fail: bool = False):
        self.dim, self.delay, self.fail = dim, delay, fail
        self.seen: list[list[str]] = []
        self.threads: list[str] = []

    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> _Out:
        assert normalize_embeddings is True
        self.seen.append(sentences)
        self.threads.append(threading.current_thread().name)
        if self.fail:
            raise RuntimeError("boom with secret text")
        time.sleep(self.delay)
        return _Out([[1.0] + [0.0] * (self.dim - 1) for _ in sentences])


def _embedder(enc: FakeEncoder, loads: list[str] | None = None) -> SentenceTransformerEmbedder:
    def loader(name: str) -> FakeEncoder:
        if loads is not None:
            loads.append(name)
        return enc

    return SentenceTransformerEmbedder("fake-model", DIM, loader=loader)


def test_import_does_not_load_torch() -> None:
    code = (
        "import sys, duitku.memory.embeddings;"
        "sys.exit(any(m in sys.modules for m in ('torch','sentence_transformers')))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


async def test_lazy_single_load_and_prefixes() -> None:
    enc, loads = FakeEncoder(), list[str]()
    e = _embedder(enc, loads)
    assert loads == []  # constructing does not load
    await e.embed_query("kopi")
    await e.embed_passage("aku suka kopi")
    await asyncio.gather(e.embed_query("a"), e.embed_passages(["b", "c"]))
    assert loads == ["fake-model"]
    assert enc.seen[0] == ["query: kopi"]
    assert enc.seen[1] == ["passage: aku suka kopi"]
    assert sorted(map(tuple, enc.seen[2:])) == [("passage: b", "passage: c"), ("query: a",)]


async def test_empty_batch_does_not_load() -> None:
    loads: list[str] = []
    assert await _embedder(FakeEncoder(), loads).embed_passages([]) == []
    assert loads == []


async def test_runs_off_the_event_loop() -> None:
    enc = FakeEncoder(delay=0.3)
    e = _embedder(enc)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    await e.embed_query("x")
    t.cancel()
    assert ticks >= 10  # loop stayed responsive during the 0.3s blocking encode
    assert enc.threads[0].startswith("embed")
    assert enc.threads[0] != threading.current_thread().name


async def test_dimension_mismatch_raises() -> None:
    e = _embedder(FakeEncoder(dim=DIM + 1))
    with pytest.raises(EmbeddingError, match="expected EMBEDDING_DIM=4"):
        await e.embed_query("x")


async def test_encode_failure_is_typed_and_hides_content() -> None:
    with pytest.raises(EmbeddingError) as ei:
        await _embedder(FakeEncoder(fail=True)).embed_query("x")
    assert "secret" not in str(ei.value)
    assert isinstance(ei.value.__cause__, RuntimeError)


async def test_loader_failure_is_typed_and_retried() -> None:
    calls = 0

    def loader(_: str) -> FakeEncoder:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("offline")
        return FakeEncoder()

    e = SentenceTransformerEmbedder("m", DIM, loader=loader)
    with pytest.raises(EmbeddingError):
        await e.embed_query("x")
    assert len(await e.embed_query("x")) == DIM  # next call retries the load


def test_check_embedding_dim_matches_frozen_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    check_embedding_dim(384)  # migration 0001 / ORM constant
    with pytest.raises(EmbeddingError, match=r"vector\(384\)"):
        check_embedding_dim(1024)
    monkeypatch.setattr(embeddings, "SCHEMA_EMBEDDING_DIM", 1024)
    check_embedding_dim(1024)
