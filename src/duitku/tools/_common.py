"""Small helpers shared by the expense tools."""

import uuid
from typing import Annotated

from pydantic import BeforeValidator
from sqlalchemy.ext.asyncio import AsyncSession

from duitku.db.repositories import categories
from duitku.tools.base import Envelope, error

CategorySlug = Annotated[
    str, BeforeValidator(lambda v: v.strip().lower() if isinstance(v, str) else v)
]


async def unknown_category(session: AsyncSession, user_id: uuid.UUID, slug: str) -> Envelope:
    """Observation listing the valid slugs so the model can pick one on its next step."""
    slugs = ", ".join(c.slug for c in await categories.list_categories(session, user_id))
    return error(
        "UNKNOWN_CATEGORY",
        f"Category {slug!r} does not exist.",
        f"Use one of these slugs (or 'lainnya' if unsure): {slugs}",
    )
