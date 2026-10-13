#!/usr/bin/env bash
# pg_dump -Fc of the DATABASE_URL database into ${BACKUP_DIR:-./data/backups}/duitku-YYYYmmdd-HHMMSS.dump
# (mode 600), then keep only the newest BACKUP_KEEP (default 14) matching dumps.
# Run from the repo root (reads ./.env if DATABASE_URL is not exported). Schedule: docs/ops.md.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_dburl.sh
source "$here/_dburl.sh"
load_db_env

dir="${BACKUP_DIR:-./data/backups}"
keep="${BACKUP_KEEP:-14}"
[[ "$keep" =~ ^[1-9][0-9]*$ ]] || { echo "BACKUP_KEEP must be a positive integer" >&2; exit 2; }

umask 077
mkdir -p "$dir"
out="$dir/duitku-$(date +%Y%m%d-%H%M%S).dump"
tmp="$out.partial"
trap 'rm -f "$tmp"' EXIT   # a failed dump must never leave a file that looks like a backup

pg_dump -Fc --no-password -f "$tmp" "$DB_NAME"
chmod 600 "$tmp"
mv "$tmp" "$out"

# Rotation: only regular files named exactly duitku-<date>-<time>.dump; names sort chronologically.
shopt -s nullglob
files=("$dir"/duitku-????????-??????.dump)
excess=$(( ${#files[@]} - keep ))
(( excess > 0 )) || excess=0
for ((i = 0; i < excess; i++)); do
  [[ -f "${files[i]}" && ! -L "${files[i]}" ]] && rm -f -- "${files[i]}"
done

echo "backup ok: $out (${#files[@]} found, deleted $excess)"
