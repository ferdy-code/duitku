"""Calibrated thresholds from settings. No defaults in code: T-303 decides the numbers."""

from duitku.config import ConfigError, Settings, get_settings


def _required(value: float | None, env: str) -> float:
    if value is None:
        raise ConfigError(f"{env} is not set; calibrate it first (T-303) and put it in .env")
    return value


def dedupe_threshold(settings: Settings | None = None) -> float:
    return _required(
        (settings or get_settings()).memory_dedupe_threshold, "MEMORY_DEDUPE_THRESHOLD"
    )


def recall_threshold(settings: Settings | None = None) -> float:
    return _required(
        (settings or get_settings()).memory_recall_threshold, "MEMORY_RECALL_THRESHOLD"
    )
