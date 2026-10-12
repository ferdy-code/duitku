"""Real-LLM evals (`uv run pytest -m eval`). Collection needs no network or DB."""

from pathlib import Path

import pytest
from tests.evals.cases import Case, load_cases
from tests.evals.report import RESULTS
from tests.evals.runner import run_case

# One event loop for all cases: langchain-openai caches its httpx client process-wide, and a
# client bound to a closed loop fails ("Event loop is closed") on the next case.
pytestmark = [pytest.mark.eval, pytest.mark.asyncio(loop_scope="session")]

CASES = load_cases(Path(__file__).parent / "cases.yaml")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
async def test_eval_case(case: Case) -> None:
    result = await run_case(case)
    RESULTS.append(result)
    assert result.passed, "; ".join(result.failures)
