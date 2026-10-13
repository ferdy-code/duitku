"""T-205: REPL loop with injected streams and a fake runner (no DB, no network)."""

import io

import pytest

from duitku.interfaces.cli import cli_telegram_id, repl
from duitku.interfaces.runner import TurnResult
from duitku.interfaces.trace import ModelStep, ToolStep


class Fake:
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on

    async def __call__(self, text: str) -> TurnResult:
        self.calls.append(text)
        if text == self.fail_on:
            return TurnResult("Maaf, ada gangguan.", failed=True)
        trace = [ModelStep(calls=[ToolStep("add_expense", {"amount": "1rb"}, '{"ok":true}')])]
        return TurnResult(f"echo {text}", trace)


async def _run(script: str, fake: Fake, verbose: bool = False) -> str:
    out = io.StringIO()
    await repl(fake, stdin=io.StringIO(script), out=out, verbose=verbose)
    return out.getvalue()


@pytest.mark.parametrize("script", ["a\n/exit\nnever\n", "a\n\nnever\n", "a\n", "a"])
async def test_quits_on_exit_empty_line_or_eof(script: str) -> None:
    fake = Fake()
    out = await _run(script, fake)
    assert fake.calls == ["a"]
    assert "echo a" in out


async def test_immediate_eof_runs_nothing() -> None:
    fake = Fake()
    assert await _run("", fake) == ""
    assert fake.calls == []


async def test_error_result_is_printed_and_loop_continues() -> None:
    fake = Fake(fail_on="boom")
    out = await _run("one\nboom\ntwo\n/exit\n", fake)
    assert fake.calls == ["one", "boom", "two"]
    assert "Maaf, ada gangguan." in out and "echo two" in out


async def test_verbose_shows_trace_only_when_asked() -> None:
    quiet = await _run("x\n", Fake())
    loud = await _run("x\n", Fake(), verbose=True)
    assert "add_expense" not in quiet
    assert 'add_expense({"amount": "1rb"})' in loud


def test_cli_identity_is_stable_distinct_and_never_a_telegram_id() -> None:
    assert cli_telegram_id("a") == cli_telegram_id("a")
    assert cli_telegram_id("a") != cli_telegram_id("b")
    assert cli_telegram_id("a") < 0


async def test_pending_approval_asks_and_resumes() -> None:
    answers: list[str] = []

    async def run(text: str) -> TurnResult:
        return TurnResult("Hapus kopi Rp18.000?", pending={"human_summary": "x"})

    async def decide(decision: str) -> TurnResult:
        answers.append(decision)
        return TurnResult(f"hasil {decision}")

    out = io.StringIO()
    await repl(run, stdin=io.StringIO("hapus\ny\nhapus\nnggak\nhapus\n"), out=out, decide=decide)
    # 3rd turn: stdin is exhausted at the prompt (EOF) -> rejected, not approved
    assert answers == ["approve", "reject", "reject"]
    assert "Setuju? [y/n]" in out.getvalue() and "hasil approve" in out.getvalue()
