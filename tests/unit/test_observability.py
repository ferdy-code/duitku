"""T-406: redaction, span tree/attributes, metrics, JSON logs, setup (no network, no DB)."""

from __future__ import annotations

import json
import logging
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import NoOpTracer, SpanContext, StatusCode
from tests.conftest import Otel, ScriptedChat
from tests.fake_embedder import FakeEmbedder

from duitku.agent import graph as graph_mod
from duitku.agent.graph import build_graph, run_turn
from duitku.interfaces.telegram.handlers import on_error
from duitku.observability import metrics, tracing
from duitku.observability.logs import JsonFormatter
from duitku.observability.redact import content_attr, redact_args, user_hash
from duitku.observability.spans import agent_run, last_trace_id, tool_event
from duitku.tools import base
from duitku.tools.base import error, ok

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import Responder

UID = uuid.UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
CONFIG: RunnableConfig = {"configurable": {"user_id": str(UID), "timezone": "Asia/Jakarta"}}
SECRETS = ("SECRET-USER-MSG", "SECRET-ARG", "SECRET-REPLY", "SECRET-EXC")


class Named(ScriptedChat):
    model_name: str = "test-model"


def _graph(
    scripted_llm: Callable[[Responder], ScriptedChat],
    monkeypatch: pytest.MonkeyPatch,
    *,
    log_content: bool,
    tool_fn: Callable[..., Any] | None = None,
) -> Any:
    async def echo(x: str, config: RunnableConfig) -> dict[str, Any]:
        return ok({"x": x})

    async def slugs(config: RunnableConfig) -> list[str]:
        return ["makan-minum"]

    monkeypatch.setattr(base, "get_settings", lambda: SimpleNamespace(log_content=log_content))

    async def no_audit(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(base, "_write_audit", no_audit)
    monkeypatch.setattr(
        graph_mod,
        "get_settings",
        lambda: SimpleNamespace(max_agent_steps=5, short_term_max_tokens=6000),
    )
    tool = StructuredTool.from_function(
        coroutine=base.audited("echo")(tool_fn or echo), name="echo", description="echo"
    )
    script = iter(
        [
            AIMessage(
                "",
                tool_calls=[{"id": "c1", "name": "echo", "args": {"x": "SECRET-ARG"}}],
                usage_metadata={"input_tokens": 100, "output_tokens": 7, "total_tokens": 107},
            ),
            AIMessage(
                "jawab SECRET-REPLY",
                usage_metadata={"input_tokens": 50, "output_tokens": 3, "total_tokens": 53},
            ),
        ]
    )
    llm = Named(responder=lambda _: next(script))
    return build_graph(
        llm=llm,
        tools=[tool],
        category_slugs=slugs,
        embedder=FakeEmbedder({}, fail=True),
    )


def _ctx(span: ReadableSpan) -> SpanContext:
    ctx = span.get_span_context()
    assert ctx is not None
    return ctx


def _attr_text(spans: tuple[ReadableSpan, ...]) -> str:
    return json.dumps(
        [[dict(s.attributes or {}), [e.name for e in s.events], s.name] for s in spans], default=str
    )


# --- redaction -------------------------------------------------------------------------------


def test_redaction_on_and_off() -> None:
    args = {"note": "kopi", "n": 1}
    assert redact_args(args, False) == {"note": "[redacted]", "n": "[redacted]"}
    assert redact_args(args, True) == args
    assert content_attr(args, False) is None
    assert json.loads(content_attr(args, True) or "") == args
    long = content_attr({"a": "x" * 2000}, True) or ""
    assert len(long) == 500 and long.endswith("…")


def test_user_hash_is_not_the_raw_id() -> None:
    h = user_hash(UID)
    assert len(h) == 16 and str(UID) not in h and h == user_hash(str(UID))
    assert h != user_hash(uuid.uuid4())


# --- span tree + attributes -----------------------------------------------------------------


@pytest.mark.parametrize("log_content", [False, True])
async def test_span_tree_attributes_and_privacy(
    otel: Otel,
    scripted_llm: Callable[[Responder], ScriptedChat],
    monkeypatch: pytest.MonkeyPatch,
    log_content: bool,
) -> None:
    graph = _graph(scripted_llm, monkeypatch, log_content=log_content)
    with agent_run(UID, "run") as stats:
        out = await run_turn(graph, [HumanMessage("SECRET-USER-MSG")], CONFIG)
    assert "SECRET-REPLY" in out["messages"][-1].content

    spans = otel.spans.get_finished_spans()
    by_name: dict[str, list[ReadableSpan]] = {}
    for s in spans:
        by_name.setdefault(s.name, []).append(s)
    assert sorted(by_name) == ["agent.llm", "agent.run", "agent.tool.echo"]
    (run,) = by_name["agent.run"]
    llms, (tool,) = by_name["agent.llm"], by_name["agent.tool.echo"]
    assert len(llms) == 2
    assert run.parent is None
    assert all(s.parent and s.parent.span_id == _ctx(run).span_id for s in (*llms, tool))
    assert len({_ctx(s).trace_id for s in spans}) == 1

    assert run.attributes
    assert run.attributes["duitku.steps"] == 2
    assert run.attributes["duitku.user_hash"] == user_hash(UID) != str(UID)
    assert [(s.attributes or {})["llm.model"] for s in llms] == ["test-model"] * 2
    assert [(s.attributes or {})["llm.tokens.input"] for s in llms] == [100, 50]
    assert [(s.attributes or {})["llm.tokens.output"] for s in llms] == [7, 3]
    t_attrs = dict(tool.attributes or {})
    assert t_attrs["duitku.tool.name"] == "echo" and t_attrs["duitku.tool.status"] == "ok"
    assert (stats.steps, stats.input_tokens, stats.output_tokens) == (2, 150, 10)
    assert stats.trace_id == format(_ctx(run).trace_id, "032x")

    dump = _attr_text(spans)
    assert str(UID) not in dump and "SECRET-USER-MSG" not in dump and "SECRET-REPLY" not in dump
    if log_content:
        assert "SECRET-ARG" in t_attrs["duitku.tool.args"]  # type: ignore[operator]
    else:
        assert "duitku.tool.args" not in t_attrs and "SECRET-ARG" not in dump

    assert otel.metric_total("llm_tokens_total", direction="input") == 150
    assert otel.metric_total("llm_tokens_total", direction="output") == 10
    assert otel.metric_total("agent_steps") == 1  # histogram count: one run recorded
    assert otel.metric_total("agent_run_duration_ms") == 1


async def test_tool_error_rejected_and_exception_spans(
    otel: Otel,
    scripted_llm: Callable[[Responder], ScriptedChat],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def bad(x: str, config: RunnableConfig) -> dict[str, Any]:
        return error("NOT_FOUND", "no SECRET-ARG", "h")

    graph = _graph(scripted_llm, monkeypatch, log_content=False, tool_fn=bad)
    with agent_run(UID, "run"):
        await run_turn(graph, [HumanMessage("hi")], CONFIG)
        tool_event("delete_expense", "rejected")
    tool = next(s for s in otel.spans.get_finished_spans() if s.name == "agent.tool.echo")
    assert tool.attributes and tool.attributes["duitku.tool.status"] == "error"
    assert tool.attributes["duitku.tool.error_code"] == "NOT_FOUND"
    assert tool.status.status_code == StatusCode.ERROR
    rejected = next(s for s in otel.spans.get_finished_spans() if s.name.endswith("delete_expense"))
    assert rejected.attributes and rejected.attributes["duitku.tool.status"] == "rejected"
    assert otel.metric_total("tool_errors_total", tool="echo") == 1
    assert otel.metric_total("approval_rejected_total") == 1

    with pytest.raises(RuntimeError), agent_run(UID, "run"):
        raise RuntimeError("SECRET-EXC")
    failed = otel.spans.get_finished_spans()[-1]
    assert failed.status.status_code == StatusCode.ERROR
    assert (failed.attributes or {})["duitku.error.type"] == "RuntimeError"
    assert "SECRET-EXC" not in _attr_text((failed,)) and not failed.events


# --- no-op without setup --------------------------------------------------------------------


def test_everything_is_a_noop_without_setup() -> None:
    assert isinstance(tracing.get_tracer(), NoOpTracer)
    metrics.record_run(5, 1)
    metrics.record_tokens(1, 1)
    metrics.record_tool_error("t")
    metrics.record_approval_rejected()
    with agent_run(UID, "run") as stats:
        tool_event("t", "error", "X")
    assert stats.trace_id is None and last_trace_id() is None


# --- JSON logs ------------------------------------------------------------------------------


def _record(**extra: Any) -> logging.LogRecord:
    r = logging.LogRecord("duitku.x", logging.ERROR, __file__, 1, "boom %s", ("ValueError",), None)
    r.__dict__.update(extra)
    return r


def test_json_formatter_carries_trace_ids_when_a_span_is_active(otel: Otel) -> None:
    fmt = JsonFormatter()
    outside = json.loads(fmt.format(_record()))
    assert outside["msg"] == "boom ValueError" and outside["level"] == "ERROR"
    assert "trace_id" not in outside and "exc" not in outside
    with tracing.get_tracer().start_as_current_span("s") as span:
        inside = json.loads(fmt.format(_record()))
        ctx = span.get_span_context()
    assert inside["trace_id"] == format(ctx.trace_id, "032x")
    assert inside["span_id"] == format(ctx.span_id, "016x")
    assert json.loads(fmt.format(_record(trace_id="abc")))["trace_id"] == "abc"


async def test_telegram_error_log_has_the_runs_trace_id(
    otel: Otel, caplog: pytest.LogCaptureFixture
) -> None:
    with agent_run(UID, "run") as stats:
        pass
    ctx: Any = SimpleNamespace(error=ValueError("SECRET-EXC"), bot=None)
    with caplog.at_level(logging.ERROR):
        await on_error(object(), ctx)
    rec = next(r for r in caplog.records if "update handler failed" in r.message)
    assert getattr(rec, "trace_id") == stats.trace_id is not None  # noqa: B009
    assert "SECRET-EXC" not in caplog.text


# --- setup ----------------------------------------------------------------------------------


def _settings(endpoint: str) -> Any:
    return SimpleNamespace(
        otel_exporter_otlp_endpoint=endpoint, otel_service_name="duitku-test", log_level="INFO"
    )


@pytest.fixture
def fresh_setup(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Reset module state; keep pytest's log handlers (configure_logging replaces root's)."""
    monkeypatch.setattr(tracing, "_done", False)
    monkeypatch.setattr(tracing, "_exporting", False)

    def no_logging(level: str) -> None:
        return None

    def no_atexit(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(tracing, "configure_logging", no_logging)
    monkeypatch.setattr(tracing.atexit, "register", no_atexit)
    yield monkeypatch
    metrics_mod: Any = metrics
    if metrics_mod._provider is not None:
        metrics_mod._provider.shutdown()  # stop the reader thread
    tracing.install_tracing(None)
    metrics.install_metrics(None)


def test_setup_without_endpoint_is_a_noop_and_idempotent(fresh_setup: pytest.MonkeyPatch) -> None:
    fresh_setup.setattr(tracing, "get_settings", lambda: _settings(""))
    assert tracing.setup_observability() is False
    assert tracing.setup_observability() is False
    assert isinstance(tracing.get_tracer(), NoOpTracer)


def test_setup_with_endpoint_configures_http_exporter_once(fresh_setup: pytest.MonkeyPatch) -> None:
    urls: list[str] = []
    real = tracing.OTLPSpanExporter

    def spy(endpoint: str) -> Any:
        urls.append(endpoint)
        return real(endpoint)

    fresh_setup.setattr(tracing, "get_settings", lambda: _settings("http://127.0.0.1:4318/"))
    fresh_setup.setattr(tracing, "OTLPSpanExporter", spy)
    assert tracing.setup_observability() is True
    provider = tracing._provider  # pyright: ignore[reportPrivateUsage]
    assert tracing.setup_observability() is True
    assert urls == ["http://127.0.0.1:4318/v1/traces"]  # trailing slash stripped, path appended
    assert tracing._provider is provider  # pyright: ignore[reportPrivateUsage]
    assert not isinstance(tracing.get_tracer(), NoOpTracer)
