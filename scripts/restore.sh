#!/usr/bin/env bash
# Restore a backup.sh dump into a NAMED database (created if missing).
#   scripts/restore.sh <dump-file> <target-db> [--force]
# Refuses the database named in DATABASE_URL unless --force (that overwrites live data).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_dburl.sh
source "$here/_dburl.sh"

force=0
args=()
for a in "$@"; do [[ "$a" == "--force" ]] && force=1 || args+=("$a"); done
[[ ${#args[@]} -eq 2 ]] || { echo "usage: $0 <dump-file> <target-db> [--force]" >&2; exit 2; }
dump="${args[0]}"; target="${args[1]}"

[[ -f "$dump" ]] || { echo "dump file not found" >&2; exit 2; }
[[ "$target" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || { echo "invalid target database name" >&2; exit 2; }
load_db_env
if [[ "$target" == "$DB_NAME" && $force -ne 1 ]]; then
  echo "refusing to restore into the live database '$target' (use --force)" >&2
  exit 3
fi

exists="$(psql --no-password -d postgres -Atc "SELECT 1 FROM pg_database WHERE datname = '$target'")"
[[ "$exists" == 1 ]] || createdb --no-password "$target"

# --clean --if-exists: drop-then-create, so re-restoring into a used target works.
# Non-zero exit can be harmless noise (e.g. extension already exists): pg_restore's message is shown.
pg_restore --no-password --clean --if-exists --no-owner -d "$target" "$dump"
echo "restore ok: into database '$target'"
