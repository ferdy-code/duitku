"""OTel metrics (spec: interfaces-ops.md). No-op until `tracing.setup_observability()` installs a
MeterProvider, so tests and local runs need no collector."""

from dataclasses import dataclass

from opentelemetry.metrics import Counter, Histogram, MeterProvider, NoOpMeterProvider, get_meter

_provider: MeterProvider | None = None


@dataclass(frozen=True)
class _Instruments:
    run_duration: Histogram
    steps: Histogram
    tool_errors: Counter
    tokens: Counter
    approval_rejected: Counter


_cached: _Instruments | None = None


def install_metrics(provider: MeterProvider | None) -> None:
    global _provider, _cached
    _provider, _cached = provider, None


def _instruments() -> _Instruments:
    global _cached
    if _cached is None:
        m = get_meter("duitku", meter_provider=_provider or NoOpMeterProvider())
        _cached = _Instruments(
            run_duration=m.create_histogram("agent_run_duration_ms", unit="ms"),
            steps=m.create_histogram("agent_steps"),
            tool_errors=m.create_counter("tool_errors_total"),
            tokens=m.create_counter("llm_tokens_total"),
            approval_rejected=m.create_counter("approval_rejected_total"),
        )
    return _cached


def record_run(duration_ms: int, steps: int) -> None:
    i = _instruments()
    i.run_duration.record(duration_ms)
    i.steps.record(steps)


def record_tokens(input_tokens: int, output_tokens: int) -> None:
    i = _instruments()
    i.tokens.add(input_tokens, {"direction": "input"})
    i.tokens.add(output_tokens, {"direction": "output"})


def record_tool_error(tool: str) -> None:
    _instruments().tool_errors.add(1, {"tool": tool})


def record_approval_rejected() -> None:
    _instruments().approval_rejected.add(1)
