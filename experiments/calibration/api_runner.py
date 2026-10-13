"""Gateway-served embedding model: same scoring interface as `Runner`, plus network latency."""

import asyncio
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

from calibration.models import ModelSpec
from duitku.config import get_settings
from duitku.memory.embeddings_api import ApiEmbedder


@dataclass(frozen=True)
class ApiPerf:
    """Wall-clock as seen from this machine (includes network + gateway queueing)."""

    single_ms_median: float
    single_ms_p95: float
    batch32_ms: float
    tokens_used: int  # whole run (scoring + latency probes); 0 when the API reports no usage
    dim: int


class ApiRunner:
    def __init__(self, spec: ModelSpec):
        s = get_settings()
        self.spec = spec
        self._loop = asyncio.new_event_loop()  # one loop: keeps the HTTP connection pool alive
        self._emb = ApiEmbedder(
            spec.api_model,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key.get_secret_value(),
            timeout=s.llm_timeout_seconds,
            dimensions=spec.dimensions,
            query_prefix="" if spec.no_instruction else None,
        )

    def close(self) -> None:
        self._loop.run_until_complete(self._emb.aclose())  # else httpx closes on a dead loop
        self._loop.close()

    def _run(self, texts: Sequence[str], *, query: bool) -> dict[str, list[float]]:
        uniq = list(dict.fromkeys(texts))
        fn = self._emb.embed_queries if query else self._emb.embed_passages
        return dict(zip(uniq, self._loop.run_until_complete(fn(uniq)), strict=True))

    def embed_queries(self, texts: Sequence[str]) -> dict[str, list[float]]:
        return self._run(texts, query=True)

    def embed_passages(self, texts: Sequence[str]) -> dict[str, list[float]]:
        return self._run(texts, query=False)

    def measure(self, sample: Sequence[str], runs: int = 30) -> ApiPerf:
        texts = list(sample)  # keeps duplicates, like Runner.measure, so 32 texts really go out

        def call(batch: list[str]) -> None:
            self._loop.run_until_complete(self._emb.embed_passages(batch))

        for t in texts[:3]:  # warm-up: TLS handshake, gateway cold path
            call([t])
        times: list[float] = []
        for i in range(runs):
            t0 = time.perf_counter()
            call([texts[i % len(texts)]])
            times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        call((texts * 2)[:32])
        batch_ms = (time.perf_counter() - t0) * 1000
        times.sort()
        return ApiPerf(
            statistics.median(times),
            times[int(0.95 * (len(times) - 1))],
            batch_ms,
            self._emb.tokens_used,
            self._emb.dim,
        )
