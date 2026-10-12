"""composite FKs so Postgres rejects cross-user category/receipt references

Revision ID: 0002
Revises: 0001

(user_id, category_id) -> categories(user_id, id); (user_id, receipt_id) -> receipts(user_id, id).

Default MATCH SIMPLE: a NULL receipt_id skips the check (user_id is NOT NULL, so only that
column can opt out). The originals had no ON DELETE action (NO ACTION), kept as is.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, column, referenced table, old single-column FK name (PG default), new composite name)
_FKS = [
    (
        "expenses",
        "category_id",
        "categories",
        "expenses_category_id_fkey",
        "expenses_user_category_fkey",
    ),
    (
        "expenses",
        "receipt_id",
        "receipts",
        "expenses_receipt_id_fkey",
        "expenses_user_receipt_fkey",
    ),
    (
        "budgets",
        "category_id",
        "categories",
        "budgets_category_id_fkey",
        "budgets_user_category_fkey",
    ),
]
_TARGETS = [
    ("categories", "categories_user_id_id_key"),
    ("receipts", "receipts_user_id_id_key"),
]


def upgrade() -> None:
    for table, name in _TARGETS:
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} UNIQUE (user_id, id)")
    for table, col, ref, old, new in _FKS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {old}")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {new}"
            f" FOREIGN KEY (user_id, {col}) REFERENCES {ref} (user_id, id)"
        )


def downgrade() -> None:
    for table, col, ref, old, new in _FKS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {new}")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {old} FOREIGN KEY ({col}) REFERENCES {ref} (id)"
        )
    for table, name in _TARGETS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {name}")
