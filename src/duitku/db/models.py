"""SQLAlchemy models, mirroring docs/spec/data-model.md (and migration 0001) 1:1.

LangGraph checkpoint tables are NOT here; `AsyncPostgresSaver.setup()` owns them.
Money is integer rupiah (BIGINT), never float.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector  # pyright: ignore[reportMissingTypeStubs]
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Tied to vector(N) in migration 0001 / EMBEDDING_DIM. Changing the embedding model means a new
# migration plus re-embedding (architecture.md), so this is a constant, not a runtime setting.
EMBEDDING_DIM = 384

_NOW = text("now()")


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


def _user_fk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = _pk()
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    display_name: Mapped[str | None] = mapped_column(Text)
    timezone: Mapped[str] = mapped_column(Text, server_default=text("'Asia/Jakarta'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)


class Category(Base):
    __tablename__ = "categories"
    __table_args__ = (UniqueConstraint("user_id", "slug"),)
    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = _user_fk()
    slug: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)


class Receipt(Base):
    __tablename__ = "receipts"
    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = _user_fk()
    file_path: Mapped[str] = mapped_column(Text)
    extraction: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)


class Expense(Base):
    __tablename__ = "expenses"
    __table_args__ = (
        CheckConstraint("amount_idr > 0"),
        CheckConstraint("source IN ('text','receipt')"),
        Index(
            "expenses_user_time_idx",
            "user_id",
            text("occurred_at DESC"),
            postgresql_where=text("deleted_at IS NULL"),
        ),
        Index(
            "expenses_idem_idx",
            "user_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
    )
    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = _user_fk()
    amount_idr: Mapped[int] = mapped_column(BigInteger)
    category_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("categories.id"))
    merchant: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(Text)
    receipt_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("receipts.id")
    )
    items: Mapped[list[Any] | None] = mapped_column(JSONB)
    orig_currency: Mapped[str | None] = mapped_column(Text)
    orig_amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    fx_rate: Mapped[Decimal | None] = mapped_column(Numeric(18, 6))
    idempotency_key: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Budget(Base):
    __tablename__ = "budgets"
    __table_args__ = (
        UniqueConstraint("user_id", "category_id"),
        CheckConstraint("monthly_limit_idr > 0"),
    )
    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = _user_fk()
    category_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("categories.id"))
    monthly_limit_idr: Mapped[int] = mapped_column(BigInteger)


class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (
        CheckConstraint("kind IN ('preference','category_rule','goal','fact')"),
        Index(
            "memories_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("memories_user_idx", "user_id"),
    )
    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = _user_fk()
    kind: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)
    last_recalled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    recall_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = _user_fk()
    thread_id: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    steps: Mapped[int] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(Text)  # ok | interrupted | step_limit | error
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)


class ToolAuditLog(Base):
    __tablename__ = "tool_audit_log"
    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = _user_fk()
    tool_name: Mapped[str] = mapped_column(Text)
    args: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # redacted if LOG_CONTENT=false
    status: Mapped[str] = mapped_column(Text)  # ok | error | rejected
    error_code: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_NOW)
