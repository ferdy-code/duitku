"""T-401: Telegram interface through the REAL handler chain (Application.process_update), with a
fake HTTP layer and a fake runner: no network, no DB, no LLM."""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, NetworkError
from tests.telegram_fakes import FakeBot, FakeRunner, make_app, text_update

from duitku.config import ConfigError, Settings
from duitku.interfaces.telegram import bot as bot_mod
from duitku.interfaces.telegram import texts
from duitku.interfaces.telegram.formatting import split_message, typing_indicator
from duitku.interfaces.telegram.guards import RateLimiter

ALLOWED = [1]


async def _send(
    text: str = "kopi 18rb", runner: FakeRunner | None = None, **kw: Any
) -> tuple[FakeBot, FakeRunner]:
    bot, runner = FakeBot(), runner or FakeRunner()
    app = await make_app(bot, runner, kw.pop("allowed", ALLOWED), **kw.pop("app_kw", {}))
    await app.process_update(text_update(bot, text, **kw))
    return bot, runner


# --- (1) allowlist: nothing reaches the runner ---------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"user_id": 999},  # not allowlisted
        {"user_id": 1, "with_user": False},  # no effective_user
        {"user_id": 1, "chat_type": "group", "chat_id": -100},  # allowed user, but a group
    ],
)
async def test_unauthorized_update_never_reaches_runner_and_gets_no_reply(
    kw: dict[str, Any],
) -> None:
    bot, runner = await _send(**kw)
    assert runner.resolved == [] and runner.runs == []
    assert bot.calls == []  # not even a "forbidden" reply


async def test_unauthorized_commands_and_update_without_message_are_dropped() -> None:
    bot, runner = FakeBot(), FakeRunner()
    app = await make_app(bot, runner, ALLOWED)
    await app.process_update(text_update(bot, "/start", user_id=999))
    await app.process_update(Update.de_json({"update_id": 77}, bot))  # no user, no chat
    assert bot.calls == [] and runner.runs == []


async def test_empty_allowlist_denies_everyone() -> None:
    bot, runner = await _send(allowed=[])
    assert bot.calls == [] and runner.runs == []


async def test_allowlisted_user_reaches_runner_with_chat_thread() -> None:
    bot, runner = await _send("kopi 18rb", FakeRunner("Dicatat"))
    assert runner.runs == [("tg:1", "kopi 18rb")] and runner.resolved == [1]
    assert bot.sent()[-1]["text"] == "Dicatat"


# --- (2) errors never escape ----------------------------------------------------------------


async def test_runner_exception_sends_generic_apology() -> None:
    bot, _ = await _send(runner=FakeRunner(boom=RuntimeError("secret detail")))
    assert [m["text"] for m in bot.sent()] == [texts.GENERIC_APOLOGY]
    assert "secret detail" not in str(bot.calls)


async def test_reply_send_failure_does_not_propagate() -> None:
    bot = FakeBot()
    bot.fail["sendMessage"] = [NetworkError("down")]  # the reply fails, the apology succeeds
    app = await make_app(bot, FakeRunner("hi"), ALLOWED)
    await app.process_update(text_update(bot, "halo"))
    assert [m["text"] for m in bot.sent()] == ["hi", texts.GENERIC_APOLOGY]


async def test_apology_failure_does_not_propagate() -> None:
    bot = FakeBot()
    bot.fail["sendMessage"] = [NetworkError("a"), NetworkError("b")]
    app = await make_app(bot, FakeRunner("hi"), ALLOWED)
    await app.process_update(text_update(bot, "halo"))  # must not raise


async def test_error_log_has_class_only(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    await _send(runner=FakeRunner(boom=ValueError("kopi 18rb token 123456:TEST-TOKEN-NOT-REAL")))
    assert "ValueError" in caplog.text
    assert "kopi" not in caplog.text and "TEST-TOKEN" not in caplog.text


# --- (3) rate limit -------------------------------------------------------------------------


async def test_rate_limit_blocks_then_resets_after_window() -> None:
    now = [1000.0]
    bot, runner = FakeBot(), FakeRunner()
    app = await make_app(bot, runner, ALLOWED, limit=3, clock=lambda: now[0])
    for i in range(4):
        await app.process_update(text_update(bot, f"m{i}"))
    assert len(runner.runs) == 3  # 4th blocked, agent not invoked
    assert bot.sent()[-1]["text"] == texts.RATE_LIMITED
    now[0] += 61
    await app.process_update(text_update(bot, "again"))
    assert len(runner.runs) == 4


def test_rate_limiter_is_per_user_and_sliding() -> None:
    now = [0.0]
    rl = RateLimiter(2, clock=lambda: now[0])
    assert rl.allow(1) and rl.allow(1) and not rl.allow(1)
    assert rl.allow(2)  # other user unaffected
    now[0] = 59
    assert not rl.allow(1)  # blocked tries do not extend the window
    now[0] = 60
    assert rl.allow(1)


# --- (4) input validation -------------------------------------------------------------------


async def test_long_text_refused_without_agent() -> None:
    bot, runner = await _send("x" * (texts.MAX_TEXT_CHARS + 1))
    assert runner.runs == [] and bot.sent()[0]["text"] == texts.TOO_LONG
    bot, runner = await _send("x" * texts.MAX_TEXT_CHARS)  # boundary is accepted
    assert len(runner.runs) == 1


async def test_blank_text_ignored() -> None:
    bot, runner = await _send("   ")
    assert runner.runs == [] and bot.calls == []


# --- (5) HTML --------------------------------------------------------------------------------


async def test_reply_is_html_escaped() -> None:
    bot, _ = await _send(runner=FakeRunner("<b>Rp35.000</b> & 'kopi'"))
    sent = bot.sent()[0]
    assert sent["text"] == "&lt;b&gt;Rp35.000&lt;/b&gt; &amp; 'kopi'"
    assert sent["parse_mode"] == ParseMode.HTML


async def test_bad_request_falls_back_to_plain_text_once() -> None:
    bot = FakeBot()
    bot.fail["sendMessage"] = [BadRequest("can't parse entities")]
    app = await make_app(bot, FakeRunner("a < b"), ALLOWED)
    await app.process_update(text_update(bot, "halo"))
    first, second = bot.sent()
    assert first["text"] == "a &lt; b" and second["text"] == "a < b"
    assert "parse_mode" not in second or second["parse_mode"] is None


async def test_long_reply_split_under_4096() -> None:
    long = "\n".join(f"baris {i} " + "x" * 90 for i in range(100))  # ~9.5k chars
    bot, _ = await _send(runner=FakeRunner(long))
    texts_sent = [m["text"] for m in bot.sent()]
    assert len(texts_sent) >= 3 and all(len(t) <= 4096 for t in texts_sent)
    assert "".join(texts_sent).replace("\n", "") == long.replace("\n", "")


def test_split_message_boundaries() -> None:
    assert split_message("short") == ["short"]
    assert split_message("a" * 10 + " " + "b" * 10, limit=15) == ["a" * 10, "b" * 10]
    assert split_message("x" * 25, limit=10) == ["x" * 10, "x" * 10, "x" * 5]  # no break point


# --- (6) startup ------------------------------------------------------------------------------


def _settings(token: str | None, ids: list[int]) -> Settings:
    tok = SecretStr(token) if token else None
    return cast(
        Settings,
        SimpleNamespace(
            telegram_bot_token=tok,
            telegram_allowed_user_ids=ids,
            rate_limit_messages_per_minute=20,
            log_level="INFO",
        ),
    )


def test_startup_refuses_missing_token_or_empty_allowlist() -> None:
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        bot_mod.build_app(_settings(None, [1]))
    with pytest.raises(ConfigError, match="ALLOWED_USER_IDS"):
        bot_mod.build_app(_settings("123:abc", []))


@pytest.mark.parametrize("token, ids", [(None, [1]), ("123:SECRET-TOK", [])])
def test_main_exits_2_without_polling_and_without_leaking(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    token: str | None,
    ids: list[int],
) -> None:
    monkeypatch.setattr(bot_mod, "get_settings", lambda: _settings(token, ids))
    monkeypatch.setattr(bot_mod, "setup_observability", lambda: False)  # no exporter/log rewiring
    monkeypatch.setattr(bot_mod.Application, "builder", None)  # building an app would crash
    assert bot_mod.main() == 2
    out = capsys.readouterr()
    assert "SECRET-TOK" not in out.out + out.err and "TELEGRAM" in out.err


def test_build_app_registers_guard_first_and_only_three_commands() -> None:
    app = bot_mod.build_app(_settings("123:abc", [1]))
    assert min(app.handlers) == -1 and len(app.handlers[-1]) == 1
    cmds = {c for h in app.handlers[0] for c in getattr(h, "commands", ())}
    assert cmds == {"start", "bantuan", "batal"}


def test_http_loggers_clamped() -> None:
    bot_mod.quiet_http_loggers()
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


# --- (7) commands -----------------------------------------------------------------------------


async def test_start_and_bantuan_reply() -> None:
    bot, runner = await _send("/start")
    assert "kopi 18rb" in bot.sent()[0]["text"] and runner.runs == []
    bot, runner = await _send("/bantuan")
    assert "/bantuan" in bot.sent()[0]["text"] and "hari ini" in bot.sent()[0]["text"]
    assert runner.runs == []


# --- (8) typing --------------------------------------------------------------------------------


async def test_typing_sent_while_runner_is_slow() -> None:
    release = asyncio.Event()

    async def slow() -> str:
        await release.wait()
        return "selesai"

    bot, runner = FakeBot(), FakeRunner(slow)
    app = await make_app(bot, runner, ALLOWED)
    task = asyncio.create_task(app.process_update(text_update(bot, "halo")))
    await asyncio.sleep(0.05)
    assert "sendChatAction" in bot.endpoints() and "sendMessage" not in bot.endpoints()
    assert bot.calls[0][1]["action"] == "typing"
    release.set()
    await task
    assert bot.sent()[-1]["text"] == "selesai"


async def test_typing_repeats_and_stops_cleanly() -> None:
    bot = FakeBot()
    async with typing_indicator(bot, 1, interval=0.01):
        await asyncio.sleep(0.08)
    n = len(bot.calls)
    assert n >= 3
    await asyncio.sleep(0.05)
    assert len(bot.calls) == n  # cancelled: no pings after the context exits


async def test_typing_failure_does_not_break_turn() -> None:
    bot = FakeBot()
    bot.fail["sendChatAction"] = [NetworkError("x")] * 5
    async with typing_indicator(bot, 1, interval=0.01):
        await asyncio.sleep(0.03)
