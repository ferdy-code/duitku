"""Embedding service (T-301): multilingual-e5 with query/passage prefixes.

- Importing this module does not import torch / sentence-transformers or touch the network; the
  model is loaded on first use, once, under a lock (E5-small is ~470MB on disk).
- Encoding (and the first load) run on a dedicated single-thread executor so the event loop never
  blocks. One thread also keeps torch from oversubscribing the CPU when requests overlap.
- Vectors are L2-normalized, so cosine similarity == dot product (and pgvector `<=>` agrees).
- Any failure surfaces as `EmbeddingError` so callers (T-302/T-305) can degrade gracefully.
"""

import asyncio
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Protocol, cast

from duitku.config import get_settings
from duitku.db.models import EMBEDDING_DIM as SCHEMA_EMBEDDING_DIM

QUERY_PREFIX = "query: "  # E5: text used to search
PASSAGE_PREFIX = "passage: "  # E5: text that is stored


class EmbeddingError(RuntimeError):
    """Embedding failed (model load, encode, or unexpected output shape)."""


class Embedder(Protocol):
    """What the memory store depends on; tests inject a fake."""

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
    async def embed_passage(self, text: str) -> list[float]: ...


class _Matrix(Protocol):
    def tolist(self) -> list[list[float]]: ...


class _Encoder(Protocol):
    """The slice of SentenceTransformer we use (its own typing is loose/overloaded)."""

    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> _Matrix: ...


Loader = Callable[[str], _Encoder]


def _load_sentence_transformer(model_name: str) -> _Encoder:
    from sentence_transformers import SentenceTransformer  # heavy import: keep lazy

    return cast("_Encoder", SentenceTransformer(model_name))


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int, loader: Loader = _load_sentence_transformer):
        self._model_name = model_name
        self._dim = dim
        self._loader = loader
        self._model: _Encoder | None = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")

    def _get_model(self) -> _Encoder:
        with self._lock:  # runs in the single worker thread; lock guards direct/sync callers too
            if self._model is None:
                self._model = self._loader(self._model_name)
            return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._get_model().encode(texts, normalize_embeddings=True).tolist()
        except Exception as exc:
            raise EmbeddingError(f"embedding failed ({type(exc).__name__})") from exc
        if len(vectors) != len(texts) or any(len(v) != self._dim for v in vectors):
            got = len(vectors[0]) if vectors else 0
            raise EmbeddingError(
                f"model {self._model_name!r} returned dim {got}, expected EMBEDDING_DIM={self._dim}"
            )
        return vectors

    async def _embed(self, prefix: str, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [prefix + t for t in texts]
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._encode, prefixed
        )

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(QUERY_PREFIX, texts)

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(PASSAGE_PREFIX, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_queries([text]))[0]

    async def embed_passage(self, text: str) -> list[float]:
        return (await self.embed_passages([text]))[0]


def check_embedding_dim(dim: int) -> None:
    """Fail fast if EMBEDDING_DIM != the vector(N) frozen in migration 0001 / the ORM models."""
    if dim != SCHEMA_EMBEDDING_DIM:
        raise EmbeddingError(
            f"EMBEDDING_DIM={dim} but the schema is vector({SCHEMA_EMBEDDING_DIM}); "
            "changing the model needs a migration + re-embedding (architecture.md)"
        )


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    check_embedding_dim(s.embedding_dim)
    return SentenceTransformerEmbedder(s.embedding_model, s.embedding_dim)
