"""T-304: save_memory / recall_memory / forget_memory through the real StructuredTools.

FakeEmbedder (exact cosines, no model). Calls go through `ainvoke` with a tool-call dict.
Calibrated thresholds in effect: recall 0.835, dedupe 0.985.
"""

import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.fake_embedder import FakeEmbedder, unit

from duitku.db.models import Memory
from duitku.db.repositories import audit, memories, users
from duitku.tools.registry import build_tools

SM = async_sessionmaker[AsyncSession]
Run = Callable[..., Awaitable[dict[str, Any]]]

EMB = FakeEmbedder(
    {
        "q": unit(1.0),
        "kopi = makan-minum": unit(0.99),
        "kopi itu kategori makan-minum": unit(0.97, axis=1),  # cos to previous ~0.99 (>= 0.985)
        "gojek = transport": unit(0.0, axis=2),
        "m95": unit(0.95, axis=3),  # cos to q: recalled
        "m80": unit(0.80, axis=4),  # below 0.835: not recalled
        "m90": unit(0.90, axis=5),
        "m88": unit(0.88, axis=6),
        "m86": unit(0.86, axis=7),
        "m85": unit(0.85, axis=8),
    }
)
SECRET = "pin gue 123456"


@pytest.fixture
def emb() -> FakeEmbedder:
    EMB.calls.clear()
    EMB.fail = False
    return EMB


@pytest.fixture
def run(sm: SM, emb: FakeEmbedder) -> Run:
    tools = {t.name: t for t in build_tools(sm, emb)}

    async def _run(name: str, args: dict[str, Any], uid: uuid.UUID) -> dict[str, Any]:
        config: RunnableConfig = {"configurable": {"user_id": str(uid), "timezone": "Asia/Jakarta"}}
        msg = await tools[name].ainvoke(
            {"type": "tool_call", "name": name, "args": args, "id": "c1"}, config
        )
        assert msg.tool_call_id == "c1"
        return json.loads(msg.content)

    return _run


async def make_user(sm: SM, tg: int) -> uuid.UUID:
    async with sm() as s:
        user, _ = await users.get_or_create_user(s, tg)
        await s.commit()
        return user.id


async def count(sm: SM, uid: uuid.UUID | None = None) -> int:
    async with sm() as s:
        q = select(func.count()).select_from(Memory)
        if uid:
            q = q.where(Memory.user_id == uid)
        return (await s.execute(q)).scalar_one()


def err_code(res: dict[str, Any]) -> str:
    assert res["ok"] is False
    assert res["error"]["message"] and res["error"]["hint"]
    return res["error"]["code"]


async def save(run: Run, uid: uuid.UUID, content: str, kind: str = "fact") -> dict[str, Any]:
    return await run("save_memory", {"content": content, "kind": kind}, uid)


async def test_save_then_recall_round_trip(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    saved = await save(run, uid, "m95", "preference")
    assert saved["ok"] and saved["data"]["status"] == "created"
    assert saved["data"]["kind"] == "preference"

    res = await run("recall_memory", {"query": "q"}, uid)
    assert res["ok"]
    (hit,) = res["data"]["memories"]
    assert hit["id"] == saved["data"]["memory_id"]
    assert (hit["kind"], hit["content"], hit["similarity"]) == ("preference", "m95", 0.95)


async def test_recall_applies_threshold_and_orders_by_similarity(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    for c in ("m80", "m90", "m95"):
        await save(run, uid, c)
    res = await run("recall_memory", {"query": "q", "top_k": 5}, uid)
    assert [m["content"] for m in res["data"]["memories"]] == ["m95", "m90"]  # m80 < 0.835


async def test_near_duplicate_save_reports_existing_row_updated(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    first = await save(run, uid, "kopi = makan-minum", "category_rule")
    second = await save(run, uid, "kopi itu kategori makan-minum", "category_rule")
    assert first["data"]["status"] == "created"
    assert second["data"]["status"] == "updated_existing"
    assert second["data"]["memory_id"] == first["data"]["memory_id"]
    assert await count(sm) == 1
    # different content is a new row
    third = await save(run, uid, "gojek = transport", "category_rule")
    assert third["data"]["status"] == "created" and await count(sm) == 2


async def test_agent_driven_update_forget_then_save(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    old = await save(run, uid, "m95")
    found = (await run("recall_memory", {"query": "q"}, uid))["data"]["memories"][0]
    assert (await run("forget_memory", {"memory_id": found["id"]}, uid))["ok"]
    new = await save(run, uid, "m90")
    assert (
        new["data"]["status"] == "created" and new["data"]["memory_id"] != old["data"]["memory_id"]
    )
    contents = [
        m["content"] for m in (await run("recall_memory", {"query": "q"}, uid))["data"]["memories"]
    ]
    assert contents == ["m90"]


async def test_user_isolation(sm: SM, run: Run) -> None:
    a, b = await make_user(sm, 1), await make_user(sm, 2)
    saved = await save(run, a, "m95")
    assert (await run("recall_memory", {"query": "q"}, b))["data"]["memories"] == []
    # same text for B is its own row, not a dedupe against A's
    assert (await save(run, b, "m95"))["data"]["status"] == "created"

    res = await run("forget_memory", {"memory_id": saved["data"]["memory_id"]}, b)
    assert err_code(res) == "NOT_FOUND"
    async with sm() as s:
        assert await memories.get_memory(s, a, uuid.UUID(saved["data"]["memory_id"])) is not None
    assert await count(sm, a) == 1


async def test_forget_unknown_and_malformed_ids(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    missing = await run("forget_memory", {"memory_id": str(uuid.uuid4())}, uid)
    assert err_code(missing) == "NOT_FOUND"
    assert (
        err_code(await run("forget_memory", {"memory_id": "not-a-uuid"}, uid)) == "INVALID_ARGUMENT"
    )
    ok_res = await save(run, uid, "m95")
    gone = await run("forget_memory", {"memory_id": ok_res["data"]["memory_id"]}, uid)
    assert gone["ok"] and gone["data"]["forgotten"] is True and await count(sm) == 0
    # second forget of the same id looks exactly like a never-existing id
    assert await run("forget_memory", {"memory_id": ok_res["data"]["memory_id"]}, uid) == {
        **missing
    }


@pytest.mark.parametrize(
    "content",
    [
        SECRET,
        "kartu 4111 1111 1111 1111",
        "nik 3174051503900001",
        "otp 482913",
        "password wifi: kopi1",
    ],
)
async def test_sensitive_data_refused_without_row_or_embedding(
    sm: SM, run: Run, emb: FakeEmbedder, content: str
) -> None:
    uid = await make_user(sm, 1)
    res = await save(run, uid, content, "fact")
    assert err_code(res) == "SENSITIVE_DATA"
    assert "Indonesian" in res["error"]["hint"]
    assert content not in json.dumps(res)  # label only, never the secret
    assert await count(sm) == 0
    assert emb.calls == []  # refused before embedding


@pytest.mark.parametrize(
    "args",
    [
        {"content": "m95", "kind": "transaction"},
        {"content": "m95", "kind": ""},
        {"content": "   ", "kind": "fact"},
        {"content": "", "kind": "fact"},
        {"content": "x" * 301, "kind": "fact"},
    ],
)
async def test_invalid_save_args(sm: SM, run: Run, emb: FakeEmbedder, args: dict[str, Any]) -> None:
    uid = await make_user(sm, 1)
    assert err_code(await run("save_memory", args, uid)) == "INVALID_ARGUMENT"
    assert await count(sm) == 0 and emb.calls == []


async def test_embedding_failure_is_an_observation_and_writes_nothing(
    sm: SM, run: Run, emb: FakeEmbedder
) -> None:
    uid = await make_user(sm, 1)
    emb.fail = True
    res = await save(run, uid, "m95")
    assert err_code(res) == "MEMORY_UNAVAILABLE" and "m95" not in json.dumps(res)
    assert await count(sm) == 0
    res = await run("recall_memory", {"query": "q"}, uid)
    assert err_code(res) == "MEMORY_UNAVAILABLE"


@pytest.mark.parametrize("top_k", [0, -1, 6, 100])
async def test_top_k_out_of_bounds_rejected(sm: SM, run: Run, top_k: int) -> None:
    uid = await make_user(sm, 1)
    assert err_code(await run("recall_memory", {"query": "q", "top_k": top_k}, uid)) == (
        "INVALID_ARGUMENT"
    )


async def test_top_k_limits_results_and_empty_query_rejected(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    for c in ("m95", "m90", "m88", "m86", "m85"):
        await save(run, uid, c)
    for k, n in ((1, 1), (3, 3), (5, 5)):
        res = await run("recall_memory", {"query": "q", "top_k": k}, uid)
        assert len(res["data"]["memories"]) == n
    assert (
        len((await run("recall_memory", {"query": "q"}, uid))["data"]["memories"]) == 3
    )  # default
    assert err_code(await run("recall_memory", {"query": "  "}, uid)) == "INVALID_ARGUMENT"


async def test_recall_bumps_recall_count(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    saved = await save(run, uid, "m95")
    mid = uuid.UUID(saved["data"]["memory_id"])
    for _ in range(2):
        await run("recall_memory", {"query": "q"}, uid)
    async with sm() as s:
        row = await memories.get_memory(s, uid, mid)
        assert row is not None and row.recall_count == 2 and row.last_recalled_at is not None


async def test_audit_rows_written_with_redacted_args(sm: SM, run: Run) -> None:
    uid = await make_user(sm, 1)
    await save(run, uid, SECRET)
    await save(run, uid, "m95")
    await run("recall_memory", {"query": "q"}, uid)
    await run("forget_memory", {"memory_id": str(uuid.uuid4())}, uid)
    async with sm() as s:
        rows = list(await audit.list_tool_calls(s, uid))
    got = sorted((r.tool_name, r.status, r.error_code) for r in rows)
    assert got == [
        ("forget_memory", "error", "NOT_FOUND"),
        ("recall_memory", "ok", None),
        ("save_memory", "error", "SENSITIVE_DATA"),
        ("save_memory", "ok", None),
    ]
    dump = json.dumps([r.args for r in rows])
    assert "123456" not in dump and "m95" not in dump  # LOG_CONTENT=false: values redacted
