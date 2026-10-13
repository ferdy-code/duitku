"""T-104: tool wiring that needs no DB (schemas, descriptions, clock)."""

from typing import get_args

import pytest

from duitku.domain.periods import PERIODS
from duitku.memory.validation import MEMORY_KINDS
from duitku.tools import clock
from duitku.tools.registry import build_tools
from duitku.tools.resolve_period import PeriodName


def test_period_literal_matches_domain() -> None:
    assert get_args(PeriodName) == PERIODS


def test_registry_schemas_hide_injected_args() -> None:
    tools = {t.name: t for t in build_tools()}
    assert set(tools) == {
        "add_expense",
        "query_expenses",
        "resolve_period",
        "update_expense",
        "get_exchange_rate",
        "save_memory",
        "recall_memory",
        "forget_memory",
    }
    for t in tools.values():
        assert not {"config", "tool_call_id", "user_id", "timezone"} & set(t.args)
        assert "Example" in t.description  # evals depend on example args in descriptions
    assert tools["add_expense"].args["amount"]["type"] == "string"
    assert set(tools["query_expenses"].args["group_by"]["enum"]) == {
        "none",
        "category",
        "day",
        "week",
        "month",
    }


def test_memory_tool_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise AssertionError("embedder must be resolved at call time, not build time")

    monkeypatch.setattr("duitku.tools._memory.get_embedder", boom)
    tools = {t.name: t for t in build_tools()}  # builds without loading the embedder or torch
    assert set(tools["save_memory"].args["kind"]["enum"]) == set(MEMORY_KINDS)
    assert tools["recall_memory"].args["top_k"]["maximum"] == 5
    assert tools["recall_memory"].args["top_k"]["minimum"] == 1
    assert "recall_memory first" in tools["save_memory"].description
    assert "forget_memory" in tools["save_memory"].description


def test_clock_is_aware() -> None:
    assert clock.now().tzinfo is not None
