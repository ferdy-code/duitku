"""`python -m duitku.interfaces.telegram.bot`: long-polling Telegram bot over `TurnRunner`.

Fails fast (exit 2) when TELEGRAM_BOT_TOKEN is unset or TELEGRAM_ALLOWED_USER_IDS is empty: an
empty allowlist means nobody is allowed, never open to everyone. Updates are processed one after
another (PTB default), which keeps a user's messages in order; the per-thread lock in
`TurnRunner` already makes concurrency safe if that is ever relaxed. The token is never printed
or logged (httpx's INFO request lines contain it, so those loggers are clamped to WARNING).
"""

# pyright: reportUnknownMemberType=false

import logging
import sys
from contextlib import AsyncExitStack

from telegram.ext import Application

from duitku.agent.checkpointer import open_checkpointer
from duitku.agent.graph import build_graph
from duitku.config import ConfigError, Settings, get_settings
from duitku.interfaces.runner import TurnRunner
from duitku.interfaces.telegram.app import ALLOWED_UPDATES, App, register
from duitku.memory.embeddings import warm_up_embedder
from duitku.observability.tracing import setup_observability

logger = logging.getLogger(__name__)


def require_telegram_config(settings: Settings) -> tuple[str, list[int]]:
    token = settings.telegram_bot_token
    if token is None or not token.get_secret_value().strip():
        raise ConfigError("TELEGRAM_BOT_TOKEN is not set")
    if not settings.telegram_allowed_user_ids:
        raise ConfigError("TELEGRAM_ALLOWED_USER_IDS is empty (empty allowlist = nobody allowed)")
    return token.get_secret_value().strip(), settings.telegram_allowed_user_ids


def quiet_http_loggers() -> None:
    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


async def _post_init(app: App) -> None:
    await warm_up_embedder()  # a cold model load must not eat the first user's recall timeout
    stack = AsyncExitStack()
    try:
        saver = await stack.enter_async_context(open_checkpointer())
        app.bot_data["runner"] = TurnRunner(build_graph(checkpointer=saver))
    except BaseException:
        await stack.aclose()
        raise
    app.bot_data["stack"] = stack


async def _post_shutdown(app: App) -> None:
    stack: AsyncExitStack | None = app.bot_data.pop("stack", None)
    if stack is not None:
        await stack.aclose()


def build_app(settings: Settings) -> App:
    token, allowed = require_telegram_config(settings)
    app = (
        Application.builder()
        .token(token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    return register(app, allowed, settings.rate_limit_messages_per_minute)


def main() -> int:
    try:
        settings = get_settings()
        setup_observability()  # JSON logs; OTLP export only if OTEL_EXPORTER_OTLP_ENDPOINT is set
        quiet_http_loggers()
        app = build_app(settings)
        app.run_polling(allowed_updates=ALLOWED_UPDATES)
    except ConfigError as e:
        print(f"Konfigurasi belum lengkap: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # entry point: type only; messages may embed the token or a URL
        logger.error("bot failed to start or crashed (%s)", type(e).__name__)
        print(f"Bot berhenti ({type(e).__name__}). Cek konfigurasi dan koneksi.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
