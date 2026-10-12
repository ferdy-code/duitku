# Spec: Data model

Reference DDL (translate to SQLAlchemy models + Alembic). Every domain table has `user_id`; repositories require it on every function.
LangGraph checkpoint tables come from `AsyncPostgresSaver.setup()`; exclude them from Alembic autogenerate.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  telegram_user_id bigint UNIQUE NOT NULL,
  display_name     text,
  timezone         text NOT NULL DEFAULT 'Asia/Jakarta',
  created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE categories (
  id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  slug    text NOT NULL,                 -- 'makan-minum'
  name    text NOT NULL,                 -- 'Makan & Minum'
  UNIQUE (user_id, slug),
  UNIQUE (user_id, id)                   -- FK target (migration 0002)
);
-- Seed on user creation: makan-minum, transport, belanja, tagihan, hiburan, kesehatan, pendidikan, lainnya

CREATE TABLE receipts (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  file_path  text NOT NULL,
  extraction jsonb,                      -- parse_receipt output; null until parsed
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, id)                   -- FK target (migration 0002)
);

CREATE TABLE expenses (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  amount_idr      bigint NOT NULL CHECK (amount_idr > 0),
  category_id     uuid NOT NULL,
  merchant        text,
  note            text,
  occurred_at     timestamptz NOT NULL,
  source          text NOT NULL CHECK (source IN ('text','receipt')),
  receipt_id      uuid,
  items           jsonb,                 -- receipt line items (optional)
  orig_currency   text,                  -- e.g. 'USD' when converted
  orig_amount     numeric(18,2),
  fx_rate         numeric(18,6),
  idempotency_key text,                  -- tool_call_id; prevents duplicates on retry
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now(),
  deleted_at      timestamptz,
  -- composite FKs: DB rejects another user's category/receipt (migration 0002)
  FOREIGN KEY (user_id, category_id) REFERENCES categories (user_id, id),
  FOREIGN KEY (user_id, receipt_id)  REFERENCES receipts (user_id, id)
);
CREATE INDEX expenses_user_time_idx ON expenses (user_id, occurred_at DESC) WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX expenses_idem_idx ON expenses (user_id, idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE budgets (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  category_id       uuid NOT NULL,
  monthly_limit_idr bigint NOT NULL CHECK (monthly_limit_idr > 0),
  UNIQUE (user_id, category_id),
  FOREIGN KEY (user_id, category_id) REFERENCES categories (user_id, id)
);

CREATE TABLE memories (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind             text NOT NULL CHECK (kind IN ('preference','category_rule','goal','fact')),
  content          text NOT NULL,
  embedding        vector(384) NOT NULL, -- match EMBEDDING_DIM
  created_at       timestamptz NOT NULL DEFAULT now(),
  last_recalled_at timestamptz,
  recall_count     int NOT NULL DEFAULT 0
);
CREATE INDEX memories_embedding_idx ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX memories_user_idx ON memories (user_id);

CREATE TABLE agent_runs (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  thread_id     text NOT NULL,
  model         text NOT NULL,
  steps         int NOT NULL,
  input_tokens  int,
  output_tokens int,
  duration_ms   int NOT NULL,
  status        text NOT NULL,           -- ok | interrupted | step_limit | error
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE tool_audit_log (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  tool_name  text NOT NULL,
  args       jsonb,                      -- redacted when LOG_CONTENT=false
  status     text NOT NULL,              -- ok | error | rejected
  error_code text,
  latency_ms int NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
```
