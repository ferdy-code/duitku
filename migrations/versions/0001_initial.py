"""initial schema (docs/spec/data-model.md)

Revision ID: 0001
Revises:
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen on purpose: a migration must not follow runtime config. Changing the embedding model
# (EMBEDDING_DIM) = a NEW migration + re-embed (architecture.md).
EMBEDDING_DIM = 384

# Raw DDL copied from data-model.md so the schema matches the spec exactly.
_UP = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE users (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  telegram_user_id bigint UNIQUE NOT NULL,
  display_name     text,
  timezone         text NOT NULL DEFAULT 'Asia/Jakarta',
  created_at       timestamptz NOT NULL DEFAULT now()
)""",
    """CREATE TABLE categories (
  id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  slug    text NOT NULL,
  name    text NOT NULL,
  UNIQUE (user_id, slug)
)""",
    """CREATE TABLE receipts (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  file_path  text NOT NULL,
  extraction jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
)""",
    """CREATE TABLE expenses (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  amount_idr      bigint NOT NULL CHECK (amount_idr > 0),
  category_id     uuid NOT NULL REFERENCES categories(id),
  merchant        text,
  note            text,
  occurred_at     timestamptz NOT NULL,
  source          text NOT NULL CHECK (source IN ('text','receipt')),
  receipt_id      uuid REFERENCES receipts(id),
  items           jsonb,
  orig_currency   text,
  orig_amount     numeric(18,2),
  fx_rate         numeric(18,6),
  idempotency_key text,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now(),
  deleted_at      timestamptz
)""",
    "CREATE INDEX expenses_user_time_idx ON expenses (user_id, occurred_at DESC)"
    " WHERE deleted_at IS NULL",
    "CREATE UNIQUE INDEX expenses_idem_idx ON expenses (user_id, idempotency_key)"
    " WHERE idempotency_key IS NOT NULL",
    """CREATE TABLE budgets (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  category_id       uuid NOT NULL REFERENCES categories(id),
  monthly_limit_idr bigint NOT NULL CHECK (monthly_limit_idr > 0),
  UNIQUE (user_id, category_id)
)""",
    f"""CREATE TABLE memories (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind             text NOT NULL CHECK (kind IN ('preference','category_rule','goal','fact')),
  content          text NOT NULL,
  embedding        vector({EMBEDDING_DIM}) NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  last_recalled_at timestamptz,
  recall_count     int NOT NULL DEFAULT 0
)""",
    "CREATE INDEX memories_embedding_idx ON memories USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX memories_user_idx ON memories (user_id)",
    """CREATE TABLE agent_runs (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  thread_id     text NOT NULL,
  model         text NOT NULL,
  steps         int NOT NULL,
  input_tokens  int,
  output_tokens int,
  duration_ms   int NOT NULL,
  status        text NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now()
)""",
    """CREATE TABLE tool_audit_log (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  tool_name  text NOT NULL,
  args       jsonb,
  status     text NOT NULL,
  error_code text,
  latency_ms int NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
)""",
]

# Reverse dependency order. Indexes go with their tables.
_DOWN = [
    "DROP TABLE tool_audit_log",
    "DROP TABLE agent_runs",
    "DROP TABLE memories",
    "DROP TABLE budgets",
    "DROP TABLE expenses",
    "DROP TABLE receipts",
    "DROP TABLE categories",
    "DROP TABLE users",
    # Fails (no CASCADE) if something outside this migration still uses vector: that is intended.
    "DROP EXTENSION IF EXISTS vector",
]


def upgrade() -> None:
    for stmt in _UP:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _DOWN:
        op.execute(stmt)
