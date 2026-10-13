# Sourced by backup.sh / restore.sh. Parses DATABASE_URL (SQLAlchemy form
# postgresql+psycopg://user:pass@host:port/db) into libpq env vars PGHOST/PGPORT/PGUSER/PGPASSWORD
# and DB_NAME. Never prints the URL. DATABASE_URL comes from the environment, else from .env.

load_db_env() {
  local url="${DATABASE_URL:-}"
  if [[ -z "$url" && -f .env ]]; then
    # Read the single line only; never `source` the file.
    url="$(grep -m1 '^DATABASE_URL=' .env | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//')"
  fi
  [[ -n "$url" ]] || { echo "DATABASE_URL is not set (env or .env)" >&2; return 1; }
  local rest="${url#*://}" userinfo hostpath hostport
  [[ "$rest" == *@* ]] || { echo "DATABASE_URL has no user@host part" >&2; return 1; }
  userinfo="${rest%@*}"
  hostpath="${rest##*@}"
  hostport="${hostpath%%/*}"
  DB_NAME="${hostpath#*/}"
  DB_NAME="${DB_NAME%%\?*}"
  PGUSER="${userinfo%%:*}"
  PGPASSWORD="${userinfo#*:}"
  [[ "$userinfo" == *:* ]] || PGPASSWORD=""
  PGHOST="${hostport%%:*}"
  PGPORT="${hostport#*:}"
  [[ "$hostport" == *:* ]] || PGPORT=5432
  # URL-decode user/password (%40 etc.)
  PGUSER="$(printf '%b' "${PGUSER//%/\\x}")"
  PGPASSWORD="$(printf '%b' "${PGPASSWORD//%/\\x}")"
  [[ -n "$DB_NAME" && -n "$PGHOST" && -n "$PGUSER" ]] || { echo "DATABASE_URL is incomplete" >&2; return 1; }
  export PGHOST PGPORT PGUSER PGPASSWORD DB_NAME
}
