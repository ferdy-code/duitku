"""Typed settings. The ONLY module that reads the environment (CLAUDE.md hard rule).

Required (fail fast): LLM_BASE_URL, LLM_API_KEY, DATABASE_URL, CHECKPOINT_DATABASE_URL.
Optional / lazily required: Telegram vars (only the bot needs them; CLI and tests do not) and
Blank values in `.env` (e.g. `TELEGRAM_BOT_TOKEN=`) count as unset (-> the default applies).
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ConfigError(RuntimeError):
    """Invalid or missing configuration. The message names variables, never values."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    # --- LLM (OpenAI-compatible only) ---
    llm_base_url: str = Field(min_length=1)
    llm_api_key: SecretStr
    model_agent: str = "claude-sonnet-5-5"
    model_light: str = "claude-haiku-5-5"
    model_vision: str = "claude-sonnet-5-5"
    llm_timeout_seconds: float = Field(default=30, gt=0)
    tool_timeout_seconds: float = Field(default=10, gt=0)
    max_agent_steps: int = Field(default=8, ge=1)

    # --- Telegram (optional until the bot starts) ---
    telegram_bot_token: SecretStr | None = None
    # NoDecode: pydantic-settings would otherwise demand JSON; we accept "1,2,3".
    # Empty = nobody allowed (R-14 default-deny).
    telegram_allowed_user_ids: Annotated[list[int], NoDecode] = []
    rate_limit_messages_per_minute: int = Field(default=20, ge=1)

    # --- Database (URLs carry passwords -> SecretStr; no credentials baked into code) ---
    database_url: SecretStr
    checkpoint_database_url: SecretStr

    # --- Memory / embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = Field(default=384, ge=1)
    # Calibrated for intfloat/multilingual-e5-small in T-303 (experiments/results/); re-calibrate
    # when EMBEDDING_MODEL changes. Dedupe must stay < 1.0 (an identical vector can score 0.99999).
    memory_recall_threshold: float = Field(default=0.835, ge=0, lt=1)
    memory_dedupe_threshold: float = Field(default=0.985, ge=0, lt=1)
    memory_recall_top_k: int = Field(default=3, ge=1)

    # --- App ---
    default_timezone: str = "Asia/Jakarta"
    receipt_dir: Path = Path("./data/receipts")
    receipt_retention_days: int = Field(default=30, ge=1)
    short_term_max_tokens: int = Field(default=6000, ge=1)

    # --- Observability ---
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "duitku"
    log_content: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("llm_api_key", "database_url", "checkpoint_database_url")
    @classmethod
    def _secret_not_blank(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("telegram_allowed_user_ids", mode="before")
    @classmethod
    def _split_ids(cls, v: object) -> object:
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    @field_validator("default_timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (KeyError, ValueError):  # ZoneInfoNotFoundError is a KeyError, not a ValueError
            raise ValueError("unknown IANA timezone") from None
        return v


def load_settings(env_file: str | None = ".env") -> Settings:
    """Build Settings or raise ConfigError listing offending var names (no values)."""
    try:
        return Settings(_env_file=env_file)  # pyright: ignore[reportCallIssue]
    except ValidationError as e:
        lines = [
            f"  - {'.'.join(str(p) for p in err['loc']).upper()}: {err['msg']}"
            for err in e.errors(include_input=False)
        ]
        raise ConfigError("Invalid configuration:\n" + "\n".join(lines)) from None


@lru_cache
def get_settings() -> Settings:
    return load_settings()
