"""Pure parts of the eval harness (case loading, checks, summary, DB-name safety). No DB/network."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.evals.cases import Case, load_cases
from tests.evals.checks import DbRow, Observed, check_case, numbers_in
from tests.evals.report import CaseResult, render_table, summarize
from tests.evals.runner import eval_db_url, reset_eval_db

from duitku.interfaces.trace import ModelStep, ToolStep

CASES_YAML = Path(__file__).parents[1] / "evals" / "cases.yaml"


def _case(**expect: object) -> Case:
    return Case.model_validate(
        {"id": "X", "turns": [{"user": "hi"}], "expect": {"max_steps": 3, **expect}}
    )


def _step(*calls: tuple[str, dict[str, object]]) -> ModelStep:
    return ModelStep(calls=[ToolStep(n, a, "{}") for n, a in calls], input_tokens=10)


def test_shipped_cases_load_and_cover_e01_to_e10() -> None:
    cases = load_cases(CASES_YAML)
    assert [c.id for c in cases] == [f"E-{i:02d}" for i in range(1, 11)]


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _case(tool_call=[])


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    item = "- {id: A, turns: [{user: x}], expect: {max_steps: 1}}\n"
    p.write_text(item * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(p)


def test_tool_order_and_substring_args() -> None:
    case = _case(
        tool_calls=[
            {"name": "resolve_period", "args_contains": {"period": "TODAY"}},
            {"name": "query_expenses"},
        ]
    )
    good = [_step(("resolve_period", {"period": "today"})), _step(("query_expenses", {}))]
    assert check_case(case, Observed(good, ["ok"], [])) == []
    swapped = [good[1], good[0]]
    assert check_case(case, Observed(swapped, ["ok"], []))


def test_forbidden_and_max_calls_and_max_steps() -> None:
    case = _case(no_tool_calls=["add_expense"], max_calls={"query_expenses": 1}, max_steps=1)
    steps = [_step(("add_expense", {}), ("query_expenses", {})), _step(("query_expenses", {}))]
    msgs = check_case(case, Observed(steps, ["x"], []))
    assert len(msgs) == 3


def test_db_expectations() -> None:
    case = _case(
        db={
            "expenses_count": 1,
            "expense": {
                "amount_idr": 35000,
                "category_slug": "makan-minum",
                "merchant_contains": "WAR",
            },
        }
    )
    row = DbRow(35000, "makan-minum", "Warteg", date(2026, 10, 12))
    assert check_case(case, Observed([], ["x"], [row])) == []
    wrong = DbRow(3500, "transport", None, date(2026, 10, 12))
    assert len(check_case(case, Observed([], ["x"], [wrong]))) == 3
    assert check_case(case, Observed([], ["x"], []))  # count mismatch + missing row


def test_expense_expectation_tolerates_zero_rows_when_count_allows_it() -> None:
    case = _case(db={"expenses_count": [0, 1], "expense": {"amount_idr": 50000}})
    assert check_case(case, Observed([], ["x"], [])) == []


def test_reply_checks_and_numbers() -> None:
    assert numbers_in("Total Rp125.000, 3 transaksi, 1,5 juta") == {125000, 3, 15}
    case = _case(
        reply_asks=True, reply_contains_any=["Juta"], reply_total_of_range=[date(2026, 10, 12)] * 2
    )
    assert check_case(case, Observed([], ["Maksudnya 5 juta?"], [], range_total=5)) == []
    assert len(check_case(case, Observed([], ["oke"], [], range_total=125000))) == 3


def test_summary_and_table() -> None:
    results = [
        CaseResult("E-01", ["a"], True, steps=2, input_tokens=100, output_tokens=10),
        CaseResult("E-02", ["a", "b"], False, ["boom"], steps=4, input_tokens=50, output_tokens=5),
    ]
    s = summarize(results, {"model": "m"})
    assert (s["passed"], s["pass_rate"], s["avg_steps"]) == (1, 0.5, 3.0)
    assert (s["input_tokens"], s["output_tokens"]) == (150, 15)
    assert s["pass_rate_by_tag"] == {"a": 0.5, "b": 0.0}
    assert "FAIL" in render_table(s) and "boom" in render_table(s)


def test_eval_db_name_ends_with_test_and_drop_guard() -> None:
    url = eval_db_url("postgresql+psycopg://u:p@h:1/duitku")
    assert url.endswith("/duitku_eval_test")
    with pytest.raises(ValueError, match="_test"):
        reset_eval_db("postgresql+psycopg://u:p@h:1/duitku")
