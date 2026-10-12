"""Eval case schema + loader (pure: no DB, no network). Typos in cases.yaml fail loudly."""

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Seed(_Strict):
    fixture: str | None = None  # name in runner.SEED_FIXTURES
    remove_categories: list[str] = []  # slugs deleted from the user's default categories


class Turn(_Strict):
    user: str


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


class Expect(_Strict):
    tool_calls: list[ToolExpect] = []  # must appear in this order (other calls may interleave)
    no_tool_calls: list[str] = []
    max_calls: dict[str, int] = {}  # tool name -> max number of calls
    db: DbExpect | None = None
    reply_asks: bool = False  # reply contains a question mark (asks for clarification)
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
