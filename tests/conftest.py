"""Shared test doubles. `scripted_llm` is a fake chat model: no network, tools "bind" to itself."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field

from duitku.observability.metrics import install_metrics
from duitku.observability.tracing import install_tracing

Responder = Callable[[list[BaseMessage]], AIMessage]


class ScriptedChat(BaseChatModel):
    """Answers each call with `responder(messages)`; records what it was shown in `seen`."""

    responder: Responder
    seen: list[list[BaseMessage]] = Field(default_factory=lambda: list[list[BaseMessage]]())

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChat":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responder(messages))])


@pytest.fixture
def scripted_llm() -> Callable[[Responder], ScriptedChat]:
    return lambda responder: ScriptedChat(responder=responder)


@dataclass
class Otel:
    spans: InMemorySpanExporter
    reader: InMemoryMetricReader

    def metric_total(self, name: str, **attrs: str) -> float:
        """Sum of a counter's data points (or histogram counts) whose attributes include `attrs`."""
        total = 0.0
        data: Any = self.reader.get_metrics_data()
        rms: list[Any] = data.resource_metrics if data else []
        points: list[Any] = [
            p
            for rm in rms
            for scope in rm.scope_metrics
            for m in scope.metrics
            if m.name == name
            for p in m.data.data_points
        ]
        for p in points:
            if all(p.attributes.get(k) == v for k, v in attrs.items()):
                total += getattr(p, "value", None) or getattr(p, "count", 0)
        return total


@pytest.fixture
def otel() -> Iterator[Otel]:
    """In-memory tracer + meter provider installed for one test (restored to no-op after)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    install_tracing(provider)
    install_metrics(MeterProvider(metric_readers=[reader]))
    yield Otel(exporter, reader)
    install_tracing(None)
    install_metrics(None)
