import re
from pathlib import Path

import pytest

from duitku.config import ConfigError, Settings, load_settings

SECRET = "sk-super-secret-value"
DB_PW = "hunter2"
REQUIRED = {
    "LLM_BASE_URL": "https://llm.example/v1",
    "LLM_API_KEY": SECRET,
    "DATABASE_URL": f"postgresql+psycopg://u:{DB_PW}@localhost:5432/d",
    "CHECKPOINT_DATABASE_URL": f"postgresql://u:{DB_PW}@localhost:5432/d",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for k, v in REQUIRED.items():
        monkeypatch.setenv(k, v)
    return monkeypatch


def test_loads_with_required_only_and_defaults(env: pytest.MonkeyPatch) -> None:
    s = load_settings(env_file=None)
    assert s.llm_base_url == "https://llm.example/v1"
    assert s.llm_api_key.get_secret_value() == SECRET
    assert s.model_agent == "claude-sonnet-5-5"
    assert s.max_agent_steps == 8
    assert s.llm_timeout_seconds == 30
    assert s.telegram_bot_token is None
    assert s.telegram_allowed_user_ids == []
    assert s.memory_recall_threshold == 0.835  # calibrated defaults (T-303)
    assert s.memory_dedupe_threshold == 0.985
    assert s.memory_recall_top_k == 3
    assert s.embedding_dim == 384
    assert s.default_timezone == "Asia/Jakarta"
    assert s.receipt_dir == Path("./data/receipts")
    assert s.log_content is False
    assert s.log_level == "INFO"


def test_overrides_and_types(env: pytest.MonkeyPatch) -> None:
    env.setenv("MAX_AGENT_STEPS", "5")
    env.setenv("MEMORY_RECALL_THRESHOLD", "0.87")
    env.setenv("LOG_CONTENT", "true")
    env.setenv("TELEGRAM_BOT_TOKEN", "123:tok")
    s = load_settings(env_file=None)
    assert s.max_agent_steps == 5
    assert s.memory_recall_threshold == 0.87
    assert s.log_content is True
    assert s.telegram_bot_token is not None
    assert s.telegram_bot_token.get_secret_value() == "123:tok"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("111,222, 333 ", [111, 222, 333]), ("42", [42]), ("", []), (" , ", [])],
)
def test_allowed_user_ids_comma_separated(
    env: pytest.MonkeyPatch, raw: str, expected: list[int]
) -> None:
    env.setenv("TELEGRAM_ALLOWED_USER_IDS", raw)
    assert load_settings(env_file=None).telegram_allowed_user_ids == expected


def test_allowed_user_ids_non_numeric_rejected(env: pytest.MonkeyPatch) -> None:
    env.setenv("TELEGRAM_ALLOWED_USER_IDS", "1,abc")
    with pytest.raises(ConfigError, match="TELEGRAM_ALLOWED_USER_IDS"):
        load_settings(env_file=None)


@pytest.mark.parametrize("name", ["MEMORY_RECALL_THRESHOLD", "MEMORY_DEDUPE_THRESHOLD"])
@pytest.mark.parametrize("bad", ["1.0", "1.2", "-0.1"])
def test_thresholds_must_be_in_unit_interval_below_one(
    env: pytest.MonkeyPatch, name: str, bad: str
) -> None:
    env.setenv(name, bad)
    with pytest.raises(ConfigError, match=name):
        load_settings(env_file=None)


def test_blank_optional_vars_in_env_file_are_unset(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text(
        "\n".join(f"{k}={v}" for k, v in REQUIRED.items())
        + "\nMEMORY_RECALL_THRESHOLD=\nTELEGRAM_BOT_TOKEN=\nTELEGRAM_ALLOWED_USER_IDS=\n"
    )
    s = load_settings(env_file=str(f))
    assert s.memory_recall_threshold == 0.835  # blank = unset -> calibrated default
    assert s.telegram_bot_token is None
    assert s.telegram_allowed_user_ids == []


def test_missing_required_lists_all_names_without_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", SECRET)
    with pytest.raises(ConfigError) as exc:
        load_settings(env_file=None)
    msg = str(exc.value)
    for name in ("LLM_BASE_URL", "DATABASE_URL", "CHECKPOINT_DATABASE_URL"):
        assert name in msg
    assert "LLM_API_KEY" not in msg
    assert SECRET not in msg


def test_blank_required_is_missing(env: pytest.MonkeyPatch) -> None:
    env.setenv("LLM_API_KEY", "")
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        load_settings(env_file=None)


def test_invalid_value_error_hides_secret(env: pytest.MonkeyPatch) -> None:
    env.setenv("MAX_AGENT_STEPS", "zero")
    env.setenv("DEFAULT_TIMEZONE", "Mars/Olympus")
    with pytest.raises(ConfigError) as exc:
        load_settings(env_file=None)
    msg = str(exc.value)
    assert "MAX_AGENT_STEPS" in msg
    assert "DEFAULT_TIMEZONE" in msg
    assert SECRET not in msg
    assert DB_PW not in msg


def test_secrets_not_in_repr(env: pytest.MonkeyPatch) -> None:
    env.setenv("TELEGRAM_BOT_TOKEN", "123:tok")
    s = load_settings(env_file=None)
    text = repr(s) + str(s) + str(s.model_dump())
    for leaked in (SECRET, DB_PW, "123:tok"):
        assert leaked not in text


def test_env_example_vars_all_covered() -> None:
    example = Path(__file__).parents[2] / ".env.example"
    names = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", example.read_text(), flags=re.M))
    assert names <= {n.upper() for n in Settings.model_fields}


def test_no_env_access_outside_config() -> None:
    src = Path(__file__).parents[2] / "src"
    pattern = re.compile(r"\b(os\.environ|os\.getenv|getenv|environ)\b")
    offenders = [
        str(p) for p in src.rglob("*.py") if p.name != "config.py" and pattern.search(p.read_text())
    ]
    assert offenders == []
