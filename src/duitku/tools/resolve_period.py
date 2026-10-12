"""Tool `resolve_period`: deterministic natural-period -> date range (never done by the LLM)."""

from typing import Annotated, Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.domain.periods import InvalidPeriodError
from duitku.domain.periods import resolve_period as resolve
from duitku.tools import clock
from duitku.tools.base import audited, error, get_context, ok, validation_error_envelope

PeriodName = Literal[
    "today",
    "yesterday",
    "this_week",
    "last_week",
    "this_month",
    "last_month",
    "last_7_days",
    "last_30_days",
    "this_year",
]

DESCRIPTION = (
    "Convert a relative period into exact dates in the user's timezone. Call this before "
    "query_expenses for words like 'bulan ini' or 'minggu lalu'; never compute dates yourself. "
    "Returns start_date and end_date (inclusive, YYYY-MM-DD) and a label. "
    "Example: period='last_month'."
)


def build_resolve_period_tool(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> StructuredTool:
    async def resolve_period(
        period: Annotated[PeriodName, Field(description="Relative period name.")],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        ctx = get_context(config)
        try:
            p = resolve(period, clock.now(), ctx.timezone)
        except InvalidPeriodError as e:
            return error("INVALID_ARGUMENT", str(e), e.hint)
        return ok(
            {
                "start_date": p.start_date.isoformat(),
                "end_date": p.end_date.isoformat(),
                "label": p.label,
            }
        )

    return StructuredTool.from_function(
        coroutine=audited("resolve_period", sessionmaker=sessionmaker)(resolve_period),
        name="resolve_period",
        description=DESCRIPTION,
        handle_validation_error=validation_error_envelope,
    )
