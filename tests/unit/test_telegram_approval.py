"""T-403: approval buttons, /batal and callback guarding through the REAL handler chain (fake HTTP
layer, fake runner): no network, no DB, no LLM."""

import logging
from typing import Any

import pytest
from telegram.error import BadRequest, NetworkError
from tests.telegram_fakes import (
    PENDING,
    FakeBot,
    FakeRunner,
    callback_update,
    fake_result,
    make_app,
    text_update,
)

from duitku.interfaces.runner import NOTHING_PENDING_REPLY
from duitku.interfaces.telegram import texts
from duitku.interfaces.telegram.app import ALLOWED_UPDATES

ALLOWED = [1]
QUESTION = "Hapus pengeluaran: Kopi &lt;Rp18.000&gt; (12 Okt)?"  # PENDING's summary, escaped


def buttons(msg: dict[str, Any]) -> list[tuple[str, str]]:
    markup = msg["reply_markup"]
    return [(b.text, b.callback_data) for row in markup.inline_keyboard for b in row]


async def setup(runner: FakeRunner, **kw: Any) -> tuple[FakeBot, FakeRunner, Any]:
    bot = FakeBot()
    return bot, runner, await make_app(bot, runner, ALLOWED, **kw)


# --- (1) pending result -> question with exactly two buttons ------------------------------------


async def test_pending_run_sends_escaped_question_with_two_fixed_buttons() -> None:
    bot, _, app = await setup(FakeRunner(pending=PENDING))
    await app.process_update(text_update(bot, "hapus yang tadi"))
    (msg,) = bot.sent()
    assert msg["text"] == QUESTION
    assert buttons(msg) == [(texts.BTN_APPROVE, "appr:y"), (texts.BTN_REJECT, "appr:n")]
    assert all(len(d.encode()) <= 64 for _, d in buttons(msg))


async def test_plain_reply_has_no_keyboard() -> None:
    bot, _, app = await setup(FakeRunner("Dicatat"))
    await app.process_update(text_update(bot, "kopi 18rb"))
    assert bot.sent()[0].get("reply_markup") is None


def test_callback_query_is_polled() -> None:
    assert "callback_query" in ALLOWED_UPDATES and "message" in ALLOWED_UPDATES


# --- (2) callbacks -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, decision, label",
    [("appr:y", "approve", texts.OUTCOME_APPROVED), ("appr:n", "reject", texts.OUTCOME_REJECTED)],
)
async def test_callback_resumes_once_edits_message_and_sends_final_reply(
    data: str, decision: str, label: str
) -> None:
    bot, runner, app = await setup(FakeRunner(resumes=[fake_result("Sudah dihapus")]))
    await app.process_update(callback_update(bot, data, user_id=1))
    assert runner.resume_calls == [("tg:1", decision)] and runner.runs == []
    assert bot.endpoints()[0] == "answerCallbackQuery"
    (edit,) = [d for e, d in bot.calls if e == "editMessageText"]
    assert edit["text"].endswith(f"\n\n{label}") and edit["text"].startswith("Hapus pengeluaran")
    assert edit.get("reply_markup") is None  # no reply_markup on edit = keyboard removed
    assert bot.sent()[-1]["text"] == "Sudah dihapus" and bot.sent()[-1].get("reply_markup") is None


async def test_thread_comes_from_chat_not_callback_data() -> None:
    bot, runner, app = await setup(FakeRunner())
    await app.process_update(callback_update(bot, "appr:y", user_id=1, chat_id=1))
    assert runner.resume_calls[0][0] == "tg:1"  # data carries no thread/ids at all


async def test_callback_pending_again_sends_next_question_with_buttons() -> None:
    nxt = dict(PENDING, human_summary="Hapus pengeluaran: Bensin Rp50.000?", tool_call_id="c2")
    bot, _, app = await setup(FakeRunner(resumes=[fake_result(nxt["human_summary"], nxt)]))
    await app.process_update(callback_update(bot, "appr:y"))
    last = bot.sent()[-1]
    assert last["text"] == "Hapus pengeluaran: Bensin Rp50.000?"
    assert [d for _, d in buttons(last)] == ["appr:y", "appr:n"]


async def test_stale_or_double_press_is_harmless() -> None:
    stale = fake_result(NOTHING_PENDING_REPLY)
    bot, _, app = await setup(FakeRunner(resumes=[stale, stale]))
    for _ in range(2):
        await app.process_update(callback_update(bot, "appr:y"))
    assert bot.endpoints().count("answerCallbackQuery") == 2
    edits = [d for e, d in bot.calls if e == "editMessageText"]
    assert len(edits) == 2 and all(texts.OUTCOME_APPROVED not in d["text"] for d in edits)
    assert [m["text"] for m in bot.sent()] == [NOTHING_PENDING_REPLY] * 2
    assert all(m.get("reply_markup") is None for m in bot.sent())


async def test_failed_resume_keeps_buttons_and_reports_error() -> None:
    bot, _, app = await setup(FakeRunner(resumes=[fake_result("Maaf, gangguan", failed=True)]))
    await app.process_update(callback_update(bot, "appr:y"))
    assert "editMessageText" not in bot.endpoints()
    assert bot.sent()[-1]["text"] == "Maaf, gangguan"


async def test_edit_failure_does_not_break_final_reply(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    bot, _, app = await setup(FakeRunner(resumes=[fake_result("Sudah dihapus")]))
    bot.fail["editMessageText"] = [BadRequest("message can't be edited")]
    await app.process_update(callback_update(bot, "appr:y"))
    assert bot.sent()[-1]["text"] == "Sudah dihapus"
    assert "BadRequest" in caplog.text and "can't be edited" not in caplog.text


async def test_inaccessible_old_message_skips_edit_but_still_replies() -> None:
    bot, runner, app = await setup(FakeRunner(resumes=[fake_result("Sudah dihapus")]))
    await app.process_update(callback_update(bot, "appr:y", accessible=False))
    assert "editMessageText" not in bot.endpoints()
    assert runner.resume_calls == [("tg:1", "approve")]
    assert bot.sent()[-1]["text"] == "Sudah dihapus"


async def test_answer_failure_does_not_stop_the_resume() -> None:
    bot, runner, app = await setup(FakeRunner())
    bot.fail["answerCallbackQuery"] = [BadRequest("query is too old")]
    await app.process_update(callback_update(bot, "appr:n"))
    assert runner.resume_calls == [("tg:1", "reject")]


# --- (3) guard applies to callbacks ------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"user_id": 999},  # not allowlisted
        {"user_id": 1, "chat_type": "group", "chat_id": -100},  # allowed user in a group
    ],
)
async def test_unauthorized_callback_never_reaches_runner_and_gets_no_answer(
    kw: dict[str, Any],
) -> None:
    bot, runner, app = await setup(FakeRunner())
    await app.process_update(callback_update(bot, "appr:y", **kw))
    assert runner.resolved == [] and runner.resume_calls == []
    assert bot.calls == []  # not even answerCallbackQuery


@pytest.mark.parametrize("data", ["appr:x", "appr:yy", "evil", "appr:y;drop", "", None])
async def test_malformed_callback_data_is_ignored(data: str | None) -> None:
    bot, runner, app = await setup(FakeRunner())
    await app.process_update(callback_update(bot, data))
    assert runner.resume_calls == [] and "sendMessage" not in bot.endpoints()


async def test_rate_limit_applies_to_callbacks() -> None:
    bot, runner, app = await setup(FakeRunner(), limit=2, clock=lambda: 1000.0)
    for _ in range(3):
        await app.process_update(callback_update(bot, "appr:y"))
    assert len(runner.resume_calls) == 2  # third press blocked, agent not touched
    answers = [d for e, d in bot.calls if e == "answerCallbackQuery"]
    assert answers[-1]["text"] == texts.RATE_LIMITED and len(answers) == 3


# --- (4) text while pending --------------------------------------------------------------


async def test_text_while_pending_resends_question_with_buttons() -> None:
    runner = FakeRunner(pending=PENDING)  # the real runner re-asks without running the agent
    bot, _, app = await setup(runner)
    await app.process_update(text_update(bot, "eh salah, 15rb"))
    (msg,) = bot.sent()
    assert msg["text"] == QUESTION and len(buttons(msg)) == 2
    assert runner.resume_calls == []  # typing a message never approves anything


# --- (5) /batal --------------------------------------------------------------------------


async def test_batal_with_pending_rejects() -> None:
    bot, runner, app = await setup(FakeRunner(resumes=[fake_result("Oke, tidak jadi dihapus")]))
    await app.process_update(text_update(bot, "/batal"))
    assert runner.resume_calls == [("tg:1", "reject")] and runner.runs == []
    assert bot.sent()[-1]["text"] == "Oke, tidak jadi dihapus"


async def test_batal_without_pending_says_nothing_pending() -> None:
    bot, _, app = await setup(FakeRunner(resumes=[fake_result(NOTHING_PENDING_REPLY)]))
    await app.process_update(text_update(bot, "/batal"))
    assert [m["text"] for m in bot.sent()] == [NOTHING_PENDING_REPLY]


async def test_batal_from_stranger_is_dropped() -> None:
    bot, runner, app = await setup(FakeRunner())
    await app.process_update(text_update(bot, "/batal", user_id=999))
    assert runner.resume_calls == [] and bot.calls == []


async def test_bantuan_mentions_batal() -> None:
    bot, _, app = await setup(FakeRunner())
    await app.process_update(text_update(bot, "/bantuan"))
    assert "/batal" in bot.sent()[0]["text"]


async def test_send_failure_of_question_falls_to_apology() -> None:
    bot, _, app = await setup(FakeRunner(pending=PENDING))
    bot.fail["sendMessage"] = [NetworkError("down")]
    await app.process_update(text_update(bot, "hapus"))
    assert [m["text"] for m in bot.sent()][-1] == texts.GENERIC_APOLOGY
