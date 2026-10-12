"""T-104: tool wiring that needs no DB (schemas, descriptions, clock)."""

from typing import get_args

from duitku.domain.periods import PERIODS
from duitku.tools import clock
from duitku.tools.registry import build_tools
from duitku.tools.resolve_period import PeriodName


def test_period_literal_matches_domain() -> None:
    assert get_args(PeriodName) == PERIODS


def test_registry_schemas_hide_injected_args() -> None:
    tools = {t.name: t for t in build_tools()}
    assert set(tools) == {"add_expense", "query_expenses", "resolve_period"}
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


def test_clock_is_aware() -> None:
    assert clock.now().tzinfo is not None
