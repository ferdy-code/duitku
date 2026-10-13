"""Tool `save_memory`: store one long-term fact/preference/rule/goal (memory.md).

The AGENT reconciles updates (T-303: embeddings cannot tell "budget 1,5 juta" -> "2 juta" from a
paraphrase), so the description tells it to recall first and forget the stale row. The store's
dedupe (cosine >= 0.985) only collapses near-verbatim repeats; the result says which happened.
"""

from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from duitku.db.engine import get_sessionmaker
from duitku.memory import store
from duitku.memory.embeddings import Embedder, EmbeddingError
from duitku.memory.thresholds import dedupe_threshold
from duitku.memory.validation import InvalidMemoryError, MemoryKind
from duitku.tools._memory import MEMORY_UNAVAILABLE, memory_error, resolve_embedder
from duitku.tools.base import Envelope, audited, get_context, ok, validation_error_envelope

DESCRIPTION = (
    "Save ONE durable memory the user stated explicitly: a preference, category rule, goal or "
    "recurring fact. Never transactions, and never card/ID/account numbers, PIN, OTP, passwords "
    "or keys (refused). Before saving a fact that may update an existing one (budget, merchant, "
    "date...), call recall_memory first; if a conflicting memory exists, call "
    "forget_memory(id) and then save the new text. Result status 'created' = new row; "
    "'updated_existing' = a near-identical memory was overwritten. "
    "Example: content='Gojek = transport', kind='category_rule'."
)


def build_save_memory_tool(
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    embedder: Embedder | None = None,
) -> StructuredTool:
    async def save_memory(
        content: Annotated[
            str, Field(description="Short self-contained sentence, max 300 chars, Indonesian.")
        ],
        kind: Annotated[MemoryKind, Field(description="preference|category_rule|goal|fact")],
        config: RunnableConfig,
    ) -> Envelope:
        ctx = get_context(config)
        try:
            async with (sessionmaker or get_sessionmaker())() as session:
                res = await store.save(
                    session,
                    ctx.user_id,
                    kind,
                    content,
                    resolve_embedder(embedder),
                    dedupe_threshold=dedupe_threshold(),
                )
                await session.commit()
        except InvalidMemoryError as e:
            return memory_error(e)
        except EmbeddingError:
            return MEMORY_UNAVAILABLE
        return ok(
            {
                "memory_id": str(res.memory.id),
                "kind": res.memory.kind,
                "status": "created" if res.created else "updated_existing",
            }
        )

    return StructuredTool.from_function(
        coroutine=audited("save_memory", sessionmaker=sessionmaker)(save_memory),
        name="save_memory",
        description=DESCRIPTION,
        handle_validation_error=validation_error_envelope,
    )
