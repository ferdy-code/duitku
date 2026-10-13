"""ApiEmbedder with a fake client and httpx.MockTransport: no network."""

import json
import math
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx  # the SDK's own HTTP layer (transitive dep of openai)
import openai
import pytest

from duitku.memory.embeddings import EmbeddingError
from duitku.memory.embeddings_api import QWEN3_QUERY_PREFIX, ApiEmbedder

SECRET = "sk-super-secret-key"
TEXT = "gaji masuk tgl 25 rahasia"


class FakeClient:
    """Records calls; returns `dim`-sized unnormalized vectors, optionally out of order."""

    def __init__(
        self, dim: int, *, scale: float = 3.0, reverse: bool = False, tokens: int | None = 5
    ):
        self.dim, self.scale, self.reverse, self.tokens = dim, scale, reverse, tokens
        self.calls: list[dict[str, Any]] = []
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        rows = [
            SimpleNamespace(index=i, embedding=[self.scale * (i + 1)] + [0.0] * (self.dim - 1))
            for i, _ in enumerate(kw["input"])
        ]
        if self.reverse:
            rows.reverse()
        usage = SimpleNamespace(total_tokens=self.tokens) if self.tokens is not None else None
        return SimpleNamespace(data=rows, usage=usage)


def make(model: str, client: FakeClient, **kw: Any) -> ApiEmbedder:
    return ApiEmbedder(model, client=client, **kw)


async def test_qwen3_instruction_on_query_side_only() -> None:
    c = FakeClient(1024)
    e = make("qwen3-embedding-0.6b", c)
    await e.embed_query("kapan gajian?")
    await e.embed_passage("Gaji masuk tanggal 25")
    assert c.calls[0]["input"] == [QWEN3_QUERY_PREFIX + "kapan gajian?"]
    assert c.calls[0]["input"][0].startswith("Instruct: Given a user message about personal")
    assert "\nQuery:" in c.calls[0]["input"][0]
    assert c.calls[1]["input"] == ["Gaji masuk tanggal 25"]


@pytest.mark.parametrize(
    ("model", "dim"),
    [("bge-m3", 1024), ("text-embedding-3-small", 1536), ("mistral-embed", 1024)],
)
async def test_symmetric_models_send_text_unchanged(model: str, dim: int) -> None:
    c = FakeClient(dim)
    e = make(model, c)
    await e.embed_query("a")
    await e.embed_passage("b")
    assert [x["input"] for x in c.calls] == [["a"], ["b"]]
    assert all(x["encoding_format"] == "float" and x["model"] == model for x in c.calls)
    assert all("dimensions" not in x for x in c.calls)


async def test_normalized_and_order_restored() -> None:
    c = FakeClient(1024, reverse=True)
    vecs = await make("bge-m3", c).embed_passages(["a", "b"])
    assert all(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0) for v in vecs)
    assert vecs[0][0] == 1.0  # index order, not response order (scales differ, norm hides it)
    assert c.calls[0]["input"] == ["a", "b"]


async def test_dimensions_param_only_when_requested_and_supported() -> None:
    c = FakeClient(384)
    e = make("text-embedding-3-small", c, dimensions=384)
    assert e.dim == 384
    assert len(await e.embed_query("x")) == 384
    assert c.calls[0]["dimensions"] == 384
    with pytest.raises(EmbeddingError, match="does not support"):
        make("bge-m3", FakeClient(1024), dimensions=384)
    with pytest.raises(EmbeddingError, match="unknown embedding model"):
        make("nope", FakeClient(1))


async def test_dim_mismatch_and_count_mismatch_raise() -> None:
    with pytest.raises(EmbeddingError, match="expected 1 of dim 1024"):
        await make("bge-m3", FakeClient(384)).embed_query("x")

    class Short(FakeClient):
        async def _create(self, **kw: Any) -> Any:
            r = await super()._create(**kw)
            r.data = r.data[:-1]
            return r

    with pytest.raises(EmbeddingError):
        await make("bge-m3", Short(1024)).embed_passages(["a", "b"])


async def test_zero_vector_rejected() -> None:
    with pytest.raises(EmbeddingError, match="zero or non-finite"):
        await make("bge-m3", FakeClient(1024, scale=0.0)).embed_query("x")


async def test_batching_and_empty_input() -> None:
    c = FakeClient(1024)
    e = make("bge-m3", c, max_batch=3)
    vecs = await e.embed_passages([str(i) for i in range(7)])
    assert [len(x["input"]) for x in c.calls] == [3, 3, 1]
    assert len(vecs) == 7
    assert await e.embed_passages([]) == []
    assert len(c.calls) == 3  # no request for empty input
    assert e.tokens_used == 15


async def test_missing_usage_is_fine() -> None:
    e = make("bge-m3", FakeClient(1024, tokens=None))
    await e.embed_query("x")
    assert e.tokens_used == 0


def _client(handler: Any) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url="http://gw.test/v1",
        api_key=SECRET,
        max_retries=0,
        timeout=1.0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_real_sdk_over_mock_transport_happy_path() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content), path=req.url.path)
        data = [{"object": "embedding", "index": 0, "embedding": [0.0, 2.0] + [0.0] * 1022}]
        return httpx.Response(200, json={"object": "list", "data": data, "model": "bge-m3"})

    v = await make("bge-m3", _client(handler)).embed_query("halo")  # type: ignore[arg-type]
    assert seen["path"] == "/v1/embeddings"
    assert seen["input"] == ["halo"] and seen["encoding_format"] == "float"
    assert v[1] == 1.0


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, {"error": {"message": f"bad input {TEXT}"}}),
        (401, {"error": {"message": f"bad key {SECRET} {TEXT}"}}),
        (429, {"error": {"message": "slow down"}}),
        (500, f"boom {TEXT}"),
        (200, "<html>not json</html>"),
        (200, {"data": "nope"}),
    ],
    ids=["400", "401", "429", "500", "non-json", "bad-shape"],
)
async def test_http_errors_wrapped_without_leakage(status: int, body: str | dict[str, Any]) -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raw = body if isinstance(body, str) else json.dumps(body)
        return httpx.Response(status, content=raw.encode())

    e = make("bge-m3", _client(handler))  # type: ignore[arg-type]
    with pytest.raises(EmbeddingError) as ei:
        await e.embed_query(TEXT)
    msg = str(ei.value)
    assert SECRET not in msg and TEXT not in msg and "rahasia" not in msg
    assert ei.value.__cause__ is not None


@pytest.mark.parametrize("exc", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused")])
async def test_timeout_and_connection_errors_wrapped(exc: Exception) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(EmbeddingError, match=r"APITimeoutError|APIConnectionError"):
        await make("bge-m3", _client(handler)).embed_query(TEXT)  # type: ignore[arg-type]


def test_timeout_is_passed_to_sdk_client() -> None:
    e = ApiEmbedder("bge-m3", base_url="http://gw.test/v1", api_key=SECRET, timeout=7.5)
    assert e._client.timeout == 7.5  # pyright: ignore[reportPrivateUsage]
