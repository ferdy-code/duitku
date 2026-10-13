"""T-302: kind/content hygiene and the threshold helper (no DB, no model)."""

import pytest

from duitku.config import ConfigError, Settings, load_settings
from duitku.memory.thresholds import dedupe_threshold, recall_threshold
from duitku.memory.validation import (
    MAX_CONTENT_CHARS,
    InvalidMemoryError,
    validate_content,
    validate_kind,
)


@pytest.mark.parametrize("kind", ["preference", "category_rule", "goal", "fact"])
def test_valid_kinds(kind: str) -> None:
    assert validate_kind(kind) == kind


@pytest.mark.parametrize("kind", ["", "Fact", "transaction", "fact; DROP TABLE memories"])
def test_invalid_kind(kind: str) -> None:
    with pytest.raises(InvalidMemoryError):
        validate_kind(kind)


def test_content_stripped_and_limited() -> None:
    assert validate_content("  Gojek = transport \n") == "Gojek = transport"
    assert validate_content("x" * MAX_CONTENT_CHARS) == "x" * MAX_CONTENT_CHARS
    for bad in ("", "   \n\t", "x" * (MAX_CONTENT_CHARS + 1)):
        with pytest.raises(InvalidMemoryError):
            validate_content(bad)


def _settings(monkeypatch: pytest.MonkeyPatch, **thresholds: str) -> Settings:
    for k, v in {
        "LLM_BASE_URL": "http://x/v1",
        "LLM_API_KEY": "k",
        "DATABASE_URL": "postgresql://x/y",
        "CHECKPOINT_DATABASE_URL": "postgresql://x/y",
    }.items():
        monkeypatch.setenv(k, v)
    for k in ("MEMORY_DEDUPE_THRESHOLD", "MEMORY_RECALL_THRESHOLD"):
        monkeypatch.delenv(k, raising=False)
    for k, v in thresholds.items():
        monkeypatch.setenv(k, v)
    return load_settings(env_file=None)


def test_thresholds_unset_raise_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch)
    for fn in (dedupe_threshold, recall_threshold):
        with pytest.raises(ConfigError, match="THRESHOLD"):
            fn(s)


def test_thresholds_read_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, MEMORY_DEDUPE_THRESHOLD="0.95", MEMORY_RECALL_THRESHOLD="0.85")
    assert dedupe_threshold(s) == 0.95
    assert recall_threshold(s) == 0.85
