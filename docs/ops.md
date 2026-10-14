# Ops notes

Run all commands from the repo root (they read `./.env`). Local development has no Docker (see
DECISIONS T-003); the Docker deploy below targets a VPS with Postgres on the host.

## Receipt retention (T-408)

`uv run python -m duitku.ops.retention` (local) or
`docker compose run --rm app python -m duitku.ops.retention` (deployed) deletes receipt image files older than
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

## Scheduling (host-level; Postgres is on the host)

Two jobs: `scripts/backup.sh` (host, needs `pg_dump`) and the retention job (inside the app image, because
`receipts.file_path` is an absolute path in the container's `/data/receipts`; run on the host it would
count every file as `refused`). `-T` = no TTY, needed under systemd/cron.

`/etc/systemd/system/duitku-maintenance.service`:

```ini
[Service]
Type=oneshot
WorkingDirectory=/opt/duitku
ExecStart=/opt/duitku/scripts/backup.sh
ExecStart=/usr/bin/docker compose run --rm -T app python -m duitku.ops.retention
```

`/etc/systemd/system/duitku-maintenance.timer`:

```ini
[Timer]
OnCalendar=*-*-* 03:15:00
Persistent=true
[Install]
WantedBy=timers.target
```

Enable: `systemctl enable --now duitku-maintenance.timer`; check `systemctl list-timers` and
`journalctl -u duitku-maintenance`. Cron equivalent (root or a user in the `docker` group):
`15 3 * * * cd /opt/duitku && scripts/backup.sh && docker compose run --rm -T app python -m duitku.ops.retention >> data/ops.log 2>&1`.
Backups land in `/opt/duitku/data/backups` (outside Docker). Copy them off the VPS too (a dump on the
same disk is not a backup against disk loss).

## Deploy (Docker; Postgres external)

Compose runs only the `app` service. Postgres 16+ with the `pgvector` extension is installed on the VPS
host (or elsewhere). The container reaches the host DB as `host.docker.internal` (compose adds
`extra_hosts: host.docker.internal:host-gateway`).

### One-time host setup

1. Database and role (the role needs to create the `vector` extension once, or create it as superuser first):
   `CREATE ROLE duitku LOGIN PASSWORD '...'; CREATE DATABASE duitku OWNER duitku; \c duitku; CREATE EXTENSION vector;`
2. Let Postgres accept connections from the Docker bridge: `listen_addresses = 'localhost,172.17.0.1'`
   (or `'*'` behind a firewall) in `postgresql.conf`, and a `pg_hba.conf` line such as
   `host duitku duitku 172.16.0.0/12 scram-sha-256` (covers docker0 and compose networks). Never open 5432 to
   the internet. Reload Postgres.
3. `.env` (copy `.env.example`, `chmod 600`). Use `host.docker.internal` as the DB host in BOTH URLs, e.g.
   `DATABASE_URL=postgresql+psycopg://duitku:PASS@host.docker.internal:5432/duitku` and
   `CHECKPOINT_DATABASE_URL=postgresql://duitku:PASS@host.docker.internal:5432/duitku`. Set
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USER_IDS`, `LLM_*`. `RECEIPT_DIR` and `HF_HOME` are forced to
   `/data/...` by compose, whatever `.env` says. A real DB host works too (no `extra_hosts` needed then).
4. So the host-side `scripts/backup.sh` can read the same `.env`, make the name resolve on the host:
   `echo '127.0.0.1 host.docker.internal' | sudo tee -a /etc/hosts` (or run the script with `DATABASE_URL`
   exported using `localhost`). Install `postgresql-client` (same major version as the server).

### First deploy

```bash
git clone <repo> /opt/duitku && cd /opt/duitku      # then create .env as above
docker compose build
docker compose up -d
docker compose logs -f app
```

Start order inside the container: `alembic upgrade head` (a no-op when already at head, so every
restart is safe), then the bot. If migrations fail the container exits non-zero and `restart:
unless-stopped` retries it in a loop: read the log, fix the DB/URL, `docker compose up -d` again.
Expect a slow FIRST start: the E5 embedding model (~470 MB) is downloaded into the `duitku-data`
volume (`/data/hf`) and loaded by the warm-up before polling begins; later starts reuse the cache.
The image is about 2.2 GB (CPU-only torch + sentence-transformers are required by the local embedder;
`ApiEmbedder` exists but is not wired into `get_embedder()`, see DECISIONS).

Smoke test: send `/start` to the bot from an allowlisted account, then "kopi 18rb".

### Update

```bash
cd /opt/duitku && scripts/backup.sh                  # always back up before a migration
git pull && docker compose build && docker compose up -d
docker compose logs --tail 50 app
```

### Rollback

Code only (no new migration in the bad release): `git checkout <previous-tag-or-sha> && docker compose
build && docker compose up -d`. If the bad release ran a migration, the old code may not match the
schema: restore the pre-update backup into the live DB (see below, `--force`) or run
`docker compose run --rm app alembic downgrade -1` BEFORE switching code (only if that migration has a
tested downgrade). Tag good releases (`git tag deploy-YYYYmmdd`) so there is something to return to.

### Logs, status, stop

```bash
docker compose ps
docker compose logs -f --tail 100 app        # JSON lines; rotated by compose (5 x 10 MB)
docker compose restart app
docker compose down                          # keeps the volume (receipts + model cache)
```

### Restore test (do this once before trusting backups, then every few months)

```bash
scripts/backup.sh
scripts/restore.sh data/backups/duitku-<latest>.dump duitku_restore_check   # scratch DB, not the live one
export PGPASSWORD=...   # or ~/.pgpass
psql -h localhost -U duitku duitku_restore_check -c "select count(*) from expenses"   # compare with the live DB
psql -h localhost -U duitku duitku_restore_check -c "select count(*) from memories"
```

Row counts must match the live DB (allow for rows written since the dump). Check that the `memories`
vector index exists (`\d memories`). Then `DROP DATABASE duitku_restore_check`. Disaster restore into the
live DB: `docker compose stop app`, `scripts/restore.sh <dump> duitku --force`, `docker compose up -d`.

## Observability (T-406)

Structured JSON logs (one object per line on stderr; `trace_id`/`span_id` appear while a span is
active) are always on. Traces and metrics are exported only when an OTLP endpoint is set:

| Env var | Meaning |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | BASE URL of an OTLP/**HTTP** receiver, e.g. `http://localhost:4318`. Empty (default) = spans and metrics are no-ops. `/v1/traces` and `/v1/metrics` are appended for you. |
| `OTEL_SERVICE_NAME` | `service.name` in the backend (default `duitku`). |
| `LOG_CONTENT` | `false` (default): no message text, tool args or exception text in spans/logs. `true`: adds truncated tool args (`duitku.tool.args`) and tracebacks to logs. Message text is never put on spans. |
| `LOG_LEVEL` | Log level of the JSON logs. |

Port gotcha: the exporter is OTLP over HTTP, so use port **4318**. 4317 is the gRPC receiver and will
not accept these requests (you would only see export warnings in the logs). An old `.env` that still
has `http://localhost:4317` must be changed.

Any OTLP/HTTP collector works. SigNoz (self-hosted, official docker-compose install) exposes 4318 on
the host; so does a plain Jaeger (`docker run -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one`)
or an OpenTelemetry Collector with the `debug` exporter.

`agent_runs` (per graph call: steps, input/output tokens, duration_ms, status, model) is written to
Postgres regardless of the collector. A turn that pauses for approval gets one row with status
`interrupted` and its resume gets another; sum a thread's rows for the true cost (no call is counted
twice). Cost per message example:

```sql
SELECT date_trunc('day', created_at) d, count(*) FILTER (WHERE status <> 'interrupted') turns,
       sum(input_tokens) tin, sum(output_tokens) tout FROM agent_runs GROUP BY 1 ORDER BY 1 DESC;
```

### Manual check (NOT yet verified against SigNoz)

1. Start the collector/SigNoz and set `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` in `.env`.
2. `uv run python -m duitku.interfaces.cli --verbose`; send "makan siang 35rb" (one tool call), then
   "hapus pengeluaran tadi" and answer `y` (approval pause + resume).
3. In the backend (service `duitku`, allow ~10 s for the batch exporter) expect, per user message:
   one `agent.run` trace (`duitku.run.kind` = `run`, or `resume` after the approval answer) with
   `duitku.user_hash` (a 16-hex hash, never the raw id) and `duitku.steps`; child spans `agent.llm`
   (`llm.model`, `llm.tokens.input/output`) and `agent.tool.<name>` (`duitku.tool.name`,
   `duitku.tool.status` = ok | error | rejected, `duitku.tool.error_code` on errors). The whole trace
   shares one trace id, and the JSON log lines of that turn carry the same `trace_id`.
4. With `LOG_CONTENT=false`, search the trace attributes for your message text and amounts: nothing
   may match. Metrics to find: `agent_run_duration_ms`, `agent_steps`, `tool_errors_total{tool}`,
   `llm_tokens_total{direction}`, `approval_rejected_total` (they export every 60 s).
5. Cross-check with `agent_runs`: the token sums of the turn's rows match the `llm.tokens.*` of its spans.

The exporter wiring itself was checked against a local stub receiver (POSTs reach `/v1/traces` and
`/v1/metrics` with the service name), not against SigNoz.
