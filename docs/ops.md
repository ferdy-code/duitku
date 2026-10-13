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
