"""All tools in one list. `sessionmaker`/`embedder`/`vision`/`receipt_dir` are injectable for
tests; defaults = app engine, the lazily loaded real embedder and the configured vision model."""

from pathlib import Path

from langchain_core.tools import StructuredTool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.memory.embeddings import Embedder
from duitku.receipts.vision import VisionModel
from duitku.tools.add_expense import build_add_expense_tool
from duitku.tools.delete_expense import build_delete_expense_tool
from duitku.tools.forget_memory import build_forget_memory_tool
from duitku.tools.get_exchange_rate import build_get_exchange_rate_tool
from duitku.tools.parse_receipt import build_parse_receipt_tool
from duitku.tools.query_expenses import build_query_expenses_tool
from duitku.tools.recall_memory import build_recall_memory_tool
from duitku.tools.resolve_period import build_resolve_period_tool
from duitku.tools.save_memory import build_save_memory_tool
from duitku.tools.update_expense import build_update_expense_tool


def build_tools(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    embedder: Embedder | None = None,
    vision: VisionModel | None = None,
    receipt_dir: Path | None = None,
) -> list[StructuredTool]:
    return [
        build_add_expense_tool(sessionmaker),
        build_query_expenses_tool(sessionmaker),
        build_resolve_period_tool(sessionmaker),
        build_update_expense_tool(sessionmaker),
        build_delete_expense_tool(sessionmaker),
        build_get_exchange_rate_tool(sessionmaker),
        build_save_memory_tool(sessionmaker, embedder),
        build_recall_memory_tool(sessionmaker, embedder),
        build_forget_memory_tool(sessionmaker),
        build_parse_receipt_tool(sessionmaker, vision, receipt_dir),
    ]
