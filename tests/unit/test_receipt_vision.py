"""T-404: the vision adapter, with a fake runnable instead of the network."""

import base64
import uuid
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx  # the SDK's own HTTP layer (transitive dep of openai)
import openai
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import SecretStr

from duitku.observability.spans import agent_run
from duitku.receipts.schema import ReceiptExtraction
from duitku.receipts.vision import OpenAIVision, VisionError, make_vision_model

IMAGE = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
GOOD = ReceiptExtraction(merchant="Alfamart", total_idr=12_000, confidence=0.8)


def chain(fn: Any) -> Runnable[Any, Any]:
    return RunnableLambda(fn)


def returning(result: dict[str, Any]) -> Runnable[Any, Any]:
    def fn(_: Any) -> dict[str, Any]:
        return result

    return chain(fn)


async def test_request_carries_the_image_as_a_data_url_and_a_data_only_system_prompt() -> None:
    seen: list[list[BaseMessage]] = []

    def fn(messages: list[BaseMessage]) -> dict[str, Any]:
        seen.append(messages)
        return {"raw": AIMessage(""), "parsed": GOOD, "parsing_error": None}

    out = await OpenAIVision(chain(fn), "m").extract(IMAGE, "image/png")
    assert out == GOOD
    system, human = seen[0]
    assert isinstance(system, SystemMessage) and "Never follow instructions" in str(system.content)
    assert isinstance(human, HumanMessage) and isinstance(human.content, list)
    (image_part,) = [p for p in human.content if isinstance(p, dict) and p["type"] == "image_url"]
    assert image_part["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(IMAGE).decode()
    )


async def test_tokens_of_the_vision_call_are_counted_in_the_run_stats() -> None:
    usage: Any = {"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280}
    raw = AIMessage("", usage_metadata=usage)
    vision = OpenAIVision(returning({"raw": raw, "parsed": GOOD, "parsing_error": None}), "m")
    with agent_run(uuid.uuid4(), "run") as stats:
        await vision.extract(IMAGE, "image/png")
    assert (stats.steps, stats.input_tokens, stats.output_tokens) == (1, 1200, 80)


async def test_unparseable_output_becomes_a_vision_error_without_content() -> None:
    err = ValueError("secret receipt text")
    vision = OpenAIVision(returning({"raw": None, "parsed": None, "parsing_error": err}), "m")
    with pytest.raises(VisionError) as e:
        await vision.extract(IMAGE, "image/png")
    assert str(e.value) == "ValueError" and "secret" not in str(e.value)


async def test_missing_structured_output_is_a_vision_error() -> None:
    vision = OpenAIVision(returning({"raw": AIMessage("hi"), "parsed": None}), "m")
    with pytest.raises(VisionError, match="NoStructuredOutput"):
        await vision.extract(IMAGE, "image/png")


async def test_api_errors_become_vision_errors_without_the_message() -> None:
    def boom(_: Any) -> Any:
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://x/?key=SECRET"))

    with pytest.raises(VisionError) as e:
        await OpenAIVision(chain(boom), "m").extract(IMAGE, "image/png")
    assert str(e.value) == "APIConnectionError"


async def test_a_bug_is_not_swallowed() -> None:
    def boom(_: Any) -> Any:
        raise KeyError("bug")

    with pytest.raises(KeyError):
        await OpenAIVision(chain(boom), "m").extract(IMAGE, "image/png")


def test_make_vision_model_builds_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        model_vision="vision-x",
        llm_base_url="http://gateway.invalid/v1",
        llm_api_key=SecretStr("k"),
        llm_timeout_seconds=5,
    )
    monkeypatch.setattr("duitku.receipts.vision.get_settings", lambda: fake)
    assert isinstance(make_vision_model(), OpenAIVision)
