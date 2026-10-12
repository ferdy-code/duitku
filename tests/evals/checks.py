"""Pure assertion helpers: compare what happened (Observed) with a case's expectations."""

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from tests.evals.cases import Case, ExpenseExpect, ToolExpect

from duitku.interfaces.trace import ModelStep, ToolStep


@dataclass(frozen=True)
class DbRow:
    amount_idr: int
    category_slug: str
    merchant: str | None
    occurred_on: date


@dataclass(frozen=True)
class Observed:
    steps: list[ModelStep]  # all turns, in order
    replies: list[str]
    expenses: list[DbRow]  # live (not soft-deleted) rows, oldest first
    range_total: int | None = None  # SQL reference total for expect.reply_total_of_range


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
    if exp.reply_asks and "?" not in reply:
        bad.append("reply does not ask a question")
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
