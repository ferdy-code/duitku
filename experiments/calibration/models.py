"""Load a model once, embed texts with the right prefixes, measure latency and RSS (CPU)."""

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from duitku.memory.embeddings import PASSAGE_PREFIX, QUERY_PREFIX


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    dim: int
    query_prefix: str
    passage_prefix: str
    api_model: str = (
        ""  # non-empty: served by the OpenAI-compatible gateway (calibration/api_runner)
    )
    dimensions: int | None = None  # API only: request truncated output (3-small/3-large)
    no_instruction: bool = False  # API only: drop the model's query instruction (ablation)


# E5 needs prefixes (reused from production so they cannot drift). BGE-M3 needs NONE: its model
# card says queries are encoded as-is (no instruction), and passages too.
MODELS: dict[str, ModelSpec] = {
    "e5": ModelSpec("e5", "intfloat/multilingual-e5-small", 384, QUERY_PREFIX, PASSAGE_PREFIX),
    "bge": ModelSpec("bge", "BAAI/bge-m3", 1024, "", ""),
    # Gateway models; prefixes/instructions live in memory/embeddings_api.API_MODELS.
    "api-bge-m3": ModelSpec("api-bge-m3", "bge-m3 (API)", 1024, "", "", "bge-m3"),
    "api-qwen3": ModelSpec(
        "api-qwen3", "qwen3-embedding-0.6b (API)", 1024, "", "", "qwen3-embedding-0.6b"
    ),
    "api-qwen3-noinst": ModelSpec(
        "api-qwen3-noinst",
        "qwen3 no instruction",
        1024,
        "",
        "",
        "qwen3-embedding-0.6b",
        no_instruction=True,
    ),
    "api-3small": ModelSpec(
        "api-3small", "text-embedding-3-small", 1536, "", "", "text-embedding-3-small"
    ),
    "api-3small-384": ModelSpec(
        "api-3small-384", "text-embedding-3-small @384", 384, "", "", "text-embedding-3-small", 384
    ),
    "api-3large": ModelSpec(
        "api-3large", "text-embedding-3-large", 3072, "", "", "text-embedding-3-large"
    ),
    "api-3large-384": ModelSpec(
        "api-3large-384", "text-embedding-3-large @384", 384, "", "", "text-embedding-3-large", 384
    ),
    "api-mistral": ModelSpec("api-mistral", "mistral-embed", 1024, "", "", "mistral-embed"),
}


@dataclass(frozen=True)
class Perf:
    load_s: float  # includes the one-off download on a cold cache
    rss_base_mb: float  # after importing torch/sentence-transformers, before the model
    rss_loaded_mb: float
    rss_encoded_mb: float  # after warm-up + timing runs
    single_ms_median: float
    single_ms_p95: float
    batch32_ms: float
    torch_threads: int


class _Matrix(Protocol):
    def tolist(self) -> list[list[float]]: ...


class _Encoder(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> _Matrix: ...


def rss_mb() -> float:
    """Resident set size of this process (Linux /proc; 0.0 elsewhere)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Vectors are L2-normalized at encode time, so the dot product is the cosine."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class Runner:
    """One loaded model. Fails loudly (exception from the loader) if the download/load breaks."""

    def __init__(self, spec: ModelSpec):
        import torch
        from sentence_transformers import SentenceTransformer  # heavy: import lazily

        self.spec = spec
        self._base = rss_mb()
        t0 = time.perf_counter()
        self._model = cast("_Encoder", SentenceTransformer(spec.name))
        self._load_s = time.perf_counter() - t0
        self._loaded = rss_mb()
        self._threads = int(torch.get_num_threads())

    def encode(self, texts: Sequence[str]) -> dict[str, list[float]]:
        uniq = list(dict.fromkeys(texts))
        vecs = self._model.encode(uniq, normalize_embeddings=True).tolist()
        if any(len(v) != self.spec.dim for v in vecs):
            raise ValueError(f"{self.spec.name}: got dim {len(vecs[0])}, expected {self.spec.dim}")
        return dict(zip(uniq, vecs, strict=True))

    def embed(self, texts: Sequence[str], prefix: str) -> dict[str, list[float]]:
        """Map ORIGINAL text -> vector (prefix applied before encoding)."""
        by_prefixed = self.encode([prefix + t for t in texts])
        return {t: by_prefixed[prefix + t] for t in texts}

    def embed_queries(self, texts: Sequence[str]) -> dict[str, list[float]]:
        return self.embed(texts, self.spec.query_prefix)

    def embed_passages(self, texts: Sequence[str]) -> dict[str, list[float]]:
        return self.embed(texts, self.spec.passage_prefix)

    def measure(self, sample: Sequence[str], runs: int = 30) -> Perf:
        texts = [self.spec.passage_prefix + t for t in sample]
        for t in texts[:3]:  # warm-up (allocator, thread pool)
            self._model.encode([t], normalize_embeddings=True)
        times: list[float] = []
        for i in range(runs):
            t0 = time.perf_counter()
            self._model.encode([texts[i % len(texts)]], normalize_embeddings=True)
            times.append((time.perf_counter() - t0) * 1000)
        batch = (texts * 2)[:32]
        t0 = time.perf_counter()
        self._model.encode(batch, normalize_embeddings=True)
        batch_ms = (time.perf_counter() - t0) * 1000
        times.sort()
        return Perf(
            self._load_s,
            self._base,
            self._loaded,
            rss_mb(),
            statistics.median(times),
            times[int(0.95 * (len(times) - 1))],
            batch_ms,
            self._threads,
        )
