"""Tracer provider state + `setup_observability()` (spec: interfaces-ops.md, Observability).

Exporter choice: OTLP over HTTP (`opentelemetry-exporter-otlp-proto-http`, no grpcio dependency).
HTTP listens on 4318 by default (gRPC on 4317), so OTEL_EXPORTER_OTLP_ENDPOINT is the BASE URL of
a collector's HTTP receiver, e.g. http://localhost:4318; `/v1/traces` and `/v1/metrics` are
appended here. Empty endpoint (the default) = tracing/metrics stay no-ops.

The providers are kept in this package, not registered as OTel globals: nothing else in the
process (libraries) gets traced by accident, and tests can swap them freely with `install_*`.
"""

import atexit
import logging
from urllib.parse import urlsplit

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NoOpTracer, Tracer

from duitku.config import get_settings
from duitku.observability.logs import configure_logging
from duitku.observability.metrics import install_metrics

logger = logging.getLogger(__name__)

TRACER_NAME = "duitku"
_provider: TracerProvider | None = None
_done = False
_exporting = False


def install_tracing(provider: TracerProvider | None) -> None:
    global _provider
    _provider = provider


def get_tracer() -> Tracer:
    return _provider.get_tracer(TRACER_NAME) if _provider else NoOpTracer()


def _shutdown(*providers: TracerProvider | MeterProvider) -> None:
    for p in providers:  # flush buffered spans/metrics before the process exits
        p.shutdown()


def setup_observability() -> bool:
    """Configure JSON logging and, if an OTLP endpoint is set, tracing + metrics export.

    Idempotent (second call returns the first result). Returns True when exporting."""
    global _done, _exporting
    if _done:
        return _exporting
    s = get_settings()
    configure_logging(s.log_level)
    _done = True
    endpoint = s.otel_exporter_otlp_endpoint.strip().rstrip("/")
    if not endpoint:
        logger.info("observability: no OTLP endpoint set; spans and metrics are no-ops")
        return False
    resource = Resource.create({"service.name": s.otel_service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(f"{endpoint}/v1/traces")))
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(f"{endpoint}/v1/metrics"))
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    install_tracing(provider)
    install_metrics(meter_provider)
    atexit.register(_shutdown, provider, meter_provider)
    where = urlsplit(endpoint)  # host:port only; a URL may carry credentials
    logger.info(
        "observability: OTLP/HTTP export to %s:%s as service %s",
        where.hostname,
        where.port,
        s.otel_service_name,
    )
    _exporting = True
    return True
