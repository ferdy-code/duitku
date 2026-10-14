"""Tool `compare_amounts`: difference and % change of two totals (arithmetic is never the LLM's)."""

from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.tools.base import audited, ok, validation_error_envelope

DESCRIPTION = (
    "Compare two rupiah totals you already got from query_expenses (e.g. this month vs last "
    "month). Returns difference_idr (current - previous), direction (higher|lower|same) and "
    "percent_change (null if previous is 0). Never subtract or divide yourself. "
    "Example: current_idr=1200000, previous_idr=900000."
)


def compare(current: int, previous: int) -> dict[str, Any]:
    diff = current - previous
    pct = (
        str((Decimal(diff) * 100 / Decimal(previous)).quantize(Decimal("0.1"), ROUND_HALF_UP))
        if previous
        else None
    )
    direction = "higher" if diff > 0 else "lower" if diff < 0 else "same"
    return {"difference_idr": diff, "direction": direction, "percent_change": pct}


def build_compare_amounts_tool(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> StructuredTool:
    async def compare_amounts(
        current_idr: Annotated[int, Field(ge=0, description="Newer/main total in rupiah.")],
        previous_idr: Annotated[int, Field(ge=0, description="Total to compare against.")],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        return ok(compare(current_idr, previous_idr))

    return StructuredTool.from_function(
        coroutine=audited("compare_amounts", sessionmaker=sessionmaker)(compare_amounts),
        name="compare_amounts",
        description=DESCRIPTION,
        handle_validation_error=validation_error_envelope,
    )
