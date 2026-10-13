"""Pure assertion helpers: compare what happened (Observed) with a case's expectations."""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tests.evals.cases import Case, ExpenseExpect, MemoryExpect, ToolExpect

from duitku.interfaces.trace import ModelStep, ToolStep


@dataclass(frozen=True)
class DbRow:
    amount_idr: int
    category_slug: str
    merchant: str | None
    occurred_on: date


@dataclass(frozen=True)
class MemRow:
    kind: str
    content: str


@dataclass(frozen=True)
class Approval:
    tool: str
    expenses_at_pause: int  # live expense rows when the graph paused


@dataclass(frozen=True)
class Observed:
    steps: list[ModelStep]  # all turns, in order
    replies: list[str]
    expenses: list[DbRow]  # live (not soft-deleted) rows, oldest first
    range_total: int | None = None  # SQL reference total for expect.reply_total_of_range
    memories: list[MemRow] = field(default_factory=lambda: list[MemRow]())  # oldest first
    approvals: list[Approval] = field(default_factory=lambda: list[Approval]())  # pauses seen


def numbers_in(text: str) -> set[int]:
    """Integers written in text; '.' and ',' between digits are thousand separators or ignored."""
    return {int(re.sub(r"\D", "", m)) for m in re.findall(r"\d[\d.,]*", text)}


def _arg_matches(actual: Any, want: Any) -> bool:
    if isinstance(want, str):
        return isinstance(actual, str) and want.lower() in actual.lower()
    return actual == want


def _call_matches(call: ToolStep, want: ToolExpect) -> bool:
    return call.name == want.name and all(
        k in call.args and _arg_matches(call.args[k], v) for k, v in want.args_contains.items()
    )


def _check_tool_order(calls: list[ToolStep], wanted: list[ToolExpect]) -> list[str]:
    pos = 0
    for w in wanted:
        nxt = next((i for i in range(pos, len(calls)) if _call_matches(calls[i], w)), None)
        if nxt is None:
            seen = [c.name for c in calls]
            return [f"expected tool call {w.name}{w.args_contains or ''} (in order); saw {seen}"]
        pos = nxt + 1
    return []


def _check_expense(row: DbRow, want: ExpenseExpect) -> list[str]:
    bad: list[str] = []
    if want.amount_idr is not None and row.amount_idr != want.amount_idr:
        bad.append(f"amount_idr {row.amount_idr} != {want.amount_idr}")
    if want.category_slug is not None and row.category_slug != want.category_slug:
        bad.append(f"category {row.category_slug!r} != {want.category_slug!r}")
    if want.occurred_on is not None and row.occurred_on != want.occurred_on:
        bad.append(f"occurred_on {row.occurred_on} != {want.occurred_on}")
    if (
        want.merchant_contains is not None
        and want.merchant_contains.lower() not in (row.merchant or "").lower()
    ):
        bad.append(f"merchant {row.merchant!r} lacks {want.merchant_contains!r}")
    return bad


def _digit_windows(number: str, size: int = 8) -> list[str]:
    d = re.sub(r"\D", "", number)
    return [d[i : i + size] for i in range(max(len(d) - size + 1, 1))]


def _leaks(text: str, windows: list[str]) -> bool:
    digits = re.sub(r"\D", "", text)
    return any(w in digits for w in windows)


def _check_memories(want: MemoryExpect, rows: list[MemRow], calls: list[ToolStep]) -> list[str]:
    bad: list[str] = []
    if want.count is not None:
        allowed = [want.count] if isinstance(want.count, int) else want.count
        if len(rows) not in allowed:
            bad.append(f"memories_count {len(rows)} not in {allowed}")
    if want.kind is not None and any(r.kind != want.kind for r in rows):
        bad.append(f"memory kinds {[r.kind for r in rows]} != all {want.kind!r}")
    lowered = [r.content.lower() for r in rows]
    if want.any_contains and not any(s.lower() in c for c in lowered for s in want.any_contains):
        bad.append(f"no memory mentions any of {want.any_contains}")
    for s in want.none_contain:
        if any(s.lower() in c for c in lowered):
            bad.append(f"a memory still mentions {s!r}")
    if want.forbid_digits:
        win = _digit_windows(want.forbid_digits)
        if any(_leaks(r.content, win) for r in rows):
            bad.append("a memory row contains the forbidden digits")
        if any(_leaks(str(c.args), win) for c in calls):
            bad.append("a tool call argument contains the forbidden digits")
    return bad


def check_case(case: Case, obs: Observed) -> list[str]:
    """Failure reasons; empty list = pass."""
    exp = case.expect
    calls = [c for s in obs.steps for c in s.calls]
    reply = "\n".join(obs.replies)
    bad = _check_tool_order(calls, exp.tool_calls)
    for name in exp.no_tool_calls:
        if any(c.name == name for c in calls):
            bad.append(f"forbidden tool call {name}")
    for name, limit in exp.max_calls.items():
        n = sum(c.name == name for c in calls)
        if n > limit:
            bad.append(f"{name} called {n}x (max {limit})")
    if len(obs.steps) > exp.max_steps:
        bad.append(f"{len(obs.steps)} model steps (max {exp.max_steps})")
    if exp.db is not None:
        count = exp.db.expenses_count
        if count is not None:
            allowed = [count] if isinstance(count, int) else count
            if len(obs.expenses) not in allowed:
                bad.append(f"expenses_count {len(obs.expenses)} not in {allowed}")
        if exp.db.expense is not None and obs.expenses:
            bad += _check_expense(obs.expenses[0], exp.db.expense)
        elif exp.db.expense is not None and (
            count is None or 0 not in ([count] if isinstance(count, int) else count)
        ):
            bad.append("expected an expense row, DB has none")
    if exp.approval is not None:
        hits = [a for a in obs.approvals if a.tool == exp.approval.tool]
        if not hits:
            bad.append(f"no approval requested for {exp.approval.tool}")
        elif (want := exp.approval.expenses_at_pause) is not None and hits[
            0
        ].expenses_at_pause != want:
            bad.append(f"{hits[0].expenses_at_pause} expense rows at pause (want {want})")
    if exp.all_calls_answered and any(not c.observation for c in calls):
        bad.append("a tool call has no observation (no ToolMessage)")
    if exp.reply_asks and "?" not in reply:
        bad.append("reply does not ask a question")
    if exp.reply_no_question and "?" in reply:
        bad.append("reply asks a question")
    if exp.memories is not None:
        bad += _check_memories(exp.memories, obs.memories, calls)
    if exp.reply_contains_any and not any(
        s.lower() in reply.lower() for s in exp.reply_contains_any
    ):
        bad.append(f"reply mentions none of {exp.reply_contains_any}")
    if exp.reply_total_of_range is not None:
        if obs.range_total is None:
            bad.append("harness bug: reference total missing")
        elif obs.range_total not in numbers_in(reply):
            bad.append(f"reply lacks reference total {obs.range_total}")
    return bad
