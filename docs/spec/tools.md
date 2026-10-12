# Spec: Tools

## Contract (all tools)

```json
{"ok": true,  "data": {...}}
{"ok": false, "error": {"code": "INVALID_ARGUMENT", "message": "...", "hint": "..."}}
```

- Validation/business errors are returned as **observations** (not exceptions) so the model can self-correct in the next ReAct step. `hint` says how to fix it.
- Unexpected exceptions: log in full, return generic `INTERNAL_ERROR` without internals.
- Descriptions: English, short, with example args. Replies to users are Indonesian (system prompt handles it).
- `user_id`/`timezone` from `RunnableConfig` (`config["configurable"]`); `tool_call_id` via `InjectedToolCallId`. Never exposed as LLM-visible args.
- A decorator in `tools/base.py` writes each call to `tool_audit_log` and emits an OTel span.
- Tools stay thin: validate → call domain/repo → return envelope.

## Tool list

| # | Tool | Args | Behavior | Writes | Approval |
|---|---|---|---|---|---|
| 1 | `add_expense` | `amount: str` (raw text e.g. `"35rb"`), `category: slug`, `merchant?`, `note?`, `occurred_on?: date` (default today in user TZ), `source: "text"\|"receipt"="text"`, `receipt_id?`, `items?` | Parse via `parse_idr_amount`; validate category; save with `idempotency_key = tool_call_id`; return `expense_id`, parsed amount, category, date | DB | if `source="receipt"` |
| 2 | `update_expense` | `expense_id`, optional `amount`, `category`, `merchant`, `note`, `occurred_on` | Change only given fields; return old and new values | DB | no |
| 3 | `delete_expense` | `expense_id` | Soft delete; verify ownership | DB | **yes** |
| 4 | `query_expenses` | `start_date`, `end_date` (inclusive, user TZ), `category?`, `group_by: none\|category\|day\|week\|month="none"`, `limit ≤ 50 = 20` | Aggregate **in SQL**. Return `total_idr`, `count`, `groups[]`, `rows[]` (when `group_by=none`) | no | no |
| 5 | `resolve_period` | `period: today\|yesterday\|this_week\|last_week\|this_month\|last_month\|last_7_days\|last_30_days\|this_year` | Deterministic `start_date`, `end_date`, `label`; week starts Monday; TZ from context | no | no |
| 6 | `parse_receipt` | `receipt_id` | Vision model with structured output `ReceiptExtraction{merchant, date, total_idr, items[{name, qty, price_idr}], confidence, warnings[]}`; add warning if items sum ≠ total | `receipts.extraction` | no |
| 7 | `get_exchange_rate` | `from_currency`, `to_currency="IDR"`, `amount?` | Public FX API (suggest Frankfurter, no key; **verify endpoint at implementation time**); cache 1h; 5s timeout; retry once | HTTP out | no |
| 8 | `set_budget` | `category`, `monthly_limit: str` | Upsert monthly limit | DB | no |
| 9 | `check_budget` | `category?`, `month?: "YYYY-MM"` | Return `used_idr`, `limit_idr`, `percent`, `status: ok\|warning(≥80%)\|exceeded` | no | no |
| 10 | `save_memory` | `content`, `kind` | Embed (`passage:`), dedupe, store (see `memory.md`) | DB | no |
| 11 | `recall_memory` | `query`, `k ≤ 5 = 3` | Embed (`query:`), cosine search, threshold filter, update `last_recalled_at` | light | no |
| 12 | `forget_memory` | `memory_id` | Delete a memory owned by the user | DB | no |

## `parse_idr_amount(text) -> int` (`domain/money.py`, pure, table-tested)

- Multipliers: `rb`, `ribu`, `k` = ×1,000 · `jt`, `juta` = ×1,000,000 · `miliar`/`milyar` = ×1,000,000,000. Bare `m` is **unsupported** (ambiguous) → `AMBIGUOUS_AMOUNT`.
- With a multiplier, `.` or `,` is a **decimal**: `"1,5 juta"` → 1,500,000; `"2.5jt"` → 2,500,000.
- Without a multiplier, `.`/`,` followed by **exactly 3 digits** is a thousands separator: `"1.250.000"` → 1,250,000; `"Rp 35.000"` → 35,000. Anything else → `AMBIGUOUS_AMOUNT`. Never guess.
- Ignore `Rp`/`IDR` prefix and spaces. Value ≤ 0 or > 10,000,000,000 → error with a hint to confirm.

## Notes

- Unclear category: model may use `lainnya` or ask. User category rules (e.g. "Gojek = transport") arrive via `recalled_memories` in the prompt.
- `query_expenses` is the ONLY read path for expense data. No free-form SQL tool, ever.
