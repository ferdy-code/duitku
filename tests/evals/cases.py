"""Eval case schema + loader (pure: no DB, no network). Typos in cases.yaml fail loudly."""

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeedMemory(_Strict):
    kind: str
    content: str


class Seed(_Strict):
    fixture: str | None = None  # name in runner.SEED_FIXTURES
    memories: list[SeedMemory] = []  # saved through memory.store.save with the REAL embedder
    remove_categories: list[str] = []  # slugs deleted from the user's default categories


class Turn(_Strict):
    user: str
    thread: str = "main"  # turns with the same name share a thread; a new name = a NEW thread
    # Answer to an approval the turn pauses on; None = leave it pending (the case ends paused).
    decide: Literal["approve", "reject"] | None = None


class ToolExpect(_Strict):
    name: str
    # string -> case-insensitive substring of the actual arg; anything else -> equality
    args_contains: dict[str, Any] = {}


class ExpenseExpect(_Strict):
    amount_idr: int | None = None
    category_slug: str | None = None
    occurred_on: date | None = None
    merchant_contains: str | None = None


class DbExpect(_Strict):
    expenses_count: int | list[int] | None = None  # list = any of
    expense: ExpenseExpect | None = None  # checked against the oldest stored row


class MemoryExpect(_Strict):
    count: int | list[int] | None = None  # stored memory rows of the user (any of, if list)
    kind: str | None = None  # every stored row must have this kind
    any_contains: list[str] = []  # case-insensitive; at least one row mentions at least one
    none_contain: list[str] = []  # case-insensitive; no row mentions any of these
    # Digit string (e.g. a card number) that must not leak: no 8-digit window of it may appear,
    # digits-only, in any memory row or in any tool call argument.
    forbid_digits: str | None = None


class ApprovalExpect(_Strict):
    tool: str  # a pause for this tool's approval must have been seen
    expenses_at_pause: int | None = None  # live expense rows when the pause happened


class Expect(_Strict):
    tool_calls: list[ToolExpect] = []  # must appear in this order (other calls may interleave)
    no_tool_calls: list[str] = []
    max_calls: dict[str, int] = {}  # tool name -> max number of calls
    db: DbExpect | None = None
    reply_asks: bool = False  # reply contains a question mark (asks for clarification)
    reply_no_question: bool = False  # reply contains no question mark
    memories: MemoryExpect | None = None
    approval: ApprovalExpect | None = None
    all_calls_answered: bool = False  # every tool call of every step got an observation
    reply_contains_any: list[str] = []  # case-insensitive
    reply_total_of_range: tuple[date, date] | None = None  # reply states SQL total of this range
    max_steps: int = Field(gt=0)  # model calls over all turns


class Case(_Strict):
    id: str
    tags: list[str] = []
    seed: Seed = Seed()
    turns: list[Turn] = Field(min_length=1)
    expect: Expect


def load_cases(path: Path) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = [Case.model_validate(item) for item in raw]
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids in cases.yaml")
    return cases
