"""Embeddings over an OpenAI-compatible `/embeddings` endpoint (evaluation; not wired anywhere).

Same `Embedder` contract as the local E5 service: vectors are L2-normalized (gateways do not all
do it), the dimension is validated, and every SDK / network / HTTP failure becomes `EmbeddingError`
with no input text and no key in the message.

Per-model query/passage prefixes live in `API_MODELS`. Asymmetric models need them on exactly one
side; most hosted ones are symmetric (no prefix).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import openai

from duitku.memory.embeddings import EmbeddingError

# Qwen3-Embedding model card: queries are `Instruct: {task}\nQuery:{query}` (no space after
# "Query:"), documents are encoded as-is. The task description is recommended in English.
QWEN3_TASK = "Given a user message about personal finances, retrieve relevant remembered facts"
QWEN3_QUERY_PREFIX = f"Instruct: {QWEN3_TASK}\nQuery:"


@dataclass(frozen=True)
class ApiModel:
    dim: int  # default output size
    query_prefix: str = ""
    passage_prefix: str = ""
    can_truncate: bool = False  # honours the `dimensions` request param (verified on the gateway)


API_MODELS: dict[str, ApiModel] = {
    "bge-m3": ApiModel(1024),
    "qwen3-embedding-0.6b": ApiModel(1024, query_prefix=QWEN3_QUERY_PREFIX),
    "text-embedding-3-small": ApiModel(1536, can_truncate=True),
    "text-embedding-3-large": ApiModel(3072, can_truncate=True),
    "mistral-embed": ApiModel(1024),
}

MAX_BATCH = 64


def _normalize(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if not math.isfinite(norm) or norm == 0:
        raise EmbeddingError("embedding API returned a zero or non-finite vector")
    return [x / norm for x in v]


def _describe(exc: Exception) -> str:
    """Error summary without the exception text (it can echo the request body)."""
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")


class ApiEmbedder:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "",
        api_key: str = "",
        timeout: float = 30.0,
        dimensions: int | None = None,
        client: Any = None,  # AsyncOpenAI-like; injectable for tests
        max_batch: int = MAX_BATCH,
        query_prefix: str
        | None = None,  # override the table (experiments: Qwen3 without instruction)
    ):
        if model not in API_MODELS:
            raise EmbeddingError(f"unknown embedding model {model!r}; known: {sorted(API_MODELS)}")
        spec = API_MODELS[model]
        if dimensions is not None and not spec.can_truncate:
            raise EmbeddingError(f"model {model!r} does not support `dimensions`")
        self.model = model
        self._spec = spec
        self._query_prefix = spec.query_prefix if query_prefix is None else query_prefix
        self._dimensions = dimensions
        self.dim = dimensions or spec.dim
        self._max_batch = max(1, max_batch)
        self.tokens_used = 0  # sum of usage.total_tokens when the API reports it
        self._client = client or openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def _create(self, batch: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": batch, "encoding_format": "float"}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        try:
            resp = await self._client.embeddings.create(**kwargs)
            rows = sorted(resp.data, key=lambda d: d.index)
            vectors = [list(r.embedding) for r in rows]
            usage = getattr(resp, "usage", None)
        except openai.OpenAIError as exc:  # the SDK wraps transport errors in APIConnectionError
            raise EmbeddingError(f"embedding API failed: {_describe(exc)}") from exc
        except (AttributeError, TypeError, ValueError) as exc:  # malformed response body
            raise EmbeddingError(
                f"embedding API returned a malformed response ({type(exc).__name__})"
            ) from exc
        if len(vectors) != len(batch) or any(len(v) != self.dim for v in vectors):
            got = len(vectors[0]) if vectors else 0
            raise EmbeddingError(
                f"model {self.model!r} returned {len(vectors)} vectors of dim {got}, "
                f"expected {len(batch)} of dim {self.dim}"
            )
        self.tokens_used += getattr(usage, "total_tokens", None) or 0
        return [_normalize(v) for v in vectors]

    async def _embed(self, prefix: str, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        prefixed = [prefix + t for t in texts]
        for i in range(0, len(prefixed), self._max_batch):
            out += await self._create(prefixed[i : i + self._max_batch])
        return out

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(self._query_prefix, texts)

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(self._spec.passage_prefix, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_queries([text]))[0]

    async def embed_passage(self, text: str) -> list[float]:
        return (await self.embed_passages([text]))[0]
