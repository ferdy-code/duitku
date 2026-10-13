"""Calibrated thresholds (T-303) from settings; env overrides MEMORY_*_THRESHOLD win."""

from duitku.config import Settings, get_settings


def dedupe_threshold(settings: Settings | None = None) -> float:
    return (settings or get_settings()).memory_dedupe_threshold


def recall_threshold(settings: Settings | None = None) -> float:
    return (settings or get_settings()).memory_recall_threshold
