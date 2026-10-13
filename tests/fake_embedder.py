"""Deterministic fake embedder with exactly controllable cosine similarities (no real model)."""

import math
from collections.abc import Sequence

from duitku.db.models import EMBEDDING_DIM
from duitku.memory.embeddings import EmbeddingError


def unit(angle_cos: float, axis: int = 1) -> list[float]:
    """Unit vector whose cosine similarity to unit(1.0) (= e0) is exactly `angle_cos`."""
    v = [0.0] * EMBEDDING_DIM
    v[0] = angle_cos
    v[axis] = math.sqrt(max(0.0, 1.0 - angle_cos**2))
    return v


class FakeEmbedder:
    """Maps known strings to vectors; the E5 prefix is ignored (that is the real embedder's job)."""

    def __init__(
        self, table: dict[str, list[float]], *, fail: bool = False, dim: int | None = None
    ):
        self.table, self.fail, self.dim = table, fail, dim
        self.calls: list[tuple[str, str]] = []

    def _vec(self, text: str) -> list[float]:
        if self.fail:
            raise EmbeddingError("embedding failed (fake)")
        v = self.table[text]
        return v if self.dim is None else v[: self.dim]

    async def embed_query(self, text: str) -> list[float]:
        self.calls.append(("query", text))
        return self._vec(text)

    async def embed_passage(self, text: str) -> list[float]:
        self.calls.append(("passage", text))
        return self._vec(text)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed_query(t) for t in texts]

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed_passage(t) for t in texts]
