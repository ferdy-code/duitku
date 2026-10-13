# Ops notes (no Docker; local/VPS Postgres)

Run all commands from the repo root (they read `./.env`).

## Receipt retention (T-408)

`uv run python -m duitku.ops.retention` deletes receipt image files older than
`RECEIPT_RETENTION_DAYS`. The `receipts` row and `extraction` stay. Idempotent; prints one summary
line. Exit 1 = DB failure, 2 = config error; per-file errors are only counted in the summary.

## Backup / restore

```bash
scripts/backup.sh                                   # -> ${BACKUP_DIR:-./data/backups}/duitku-YYYYmmdd-HHMMSS.dump, keeps BACKUP_KEEP (14)
scripts/restore.sh data/backups/duitku-....dump duitku_restore_check   # named target DB (created if missing)
scripts/restore.sh <dump> duitku --force            # live DB: only with --force
```

Requires `pg_dump`/`pg_restore`/`psql`/`createdb` (postgresql-client, same major version as the
server) on PATH. Dumps hold all user data: keep `data/` uncommitted, mode 600, off shared disks.
Test a restore into a scratch DB periodically, not only after a disaster.

## Scheduling (systemd timer; cron alternative below)

`/etc/systemd/system/duitku-maintenance.service`:

```ini
[Service]
Type=oneshot
WorkingDirectory=/opt/duitku
ExecStart=/opt/duitku/scripts/backup.sh
ExecStart=/usr/bin/env uv run python -m duitku.ops.retention
```

`/etc/systemd/system/duitku-maintenance.timer`:

```ini
[Timer]
OnCalendar=*-*-* 03:15:00
Persistent=true
[Install]
WantedBy=timers.target
```

Enable: `systemctl enable --now duitku-maintenance.timer`. Cron equivalent:
`15 3 * * * cd /opt/duitku && scripts/backup.sh && uv run python -m duitku.ops.retention >> data/ops.log 2>&1`.
