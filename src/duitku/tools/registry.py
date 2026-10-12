"""All tools in one list. `sessionmaker` is injectable for tests; default = app engine."""

from langchain_core.tools import StructuredTool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.tools.add_expense import build_add_expense_tool
from duitku.tools.get_exchange_rate import build_get_exchange_rate_tool
from duitku.tools.query_expenses import build_query_expenses_tool
from duitku.tools.resolve_period import build_resolve_period_tool
from duitku.tools.update_expense import build_update_expense_tool


def build_tools(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
) -> list[StructuredTool]:
    return [
        build_add_expense_tool(sessionmaker),
        build_query_expenses_tool(sessionmaker),
        build_resolve_period_tool(sessionmaker),
        build_update_expense_tool(sessionmaker),
        build_get_exchange_rate_tool(sessionmaker),
    ]
