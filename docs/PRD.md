# PRD — Duitku

Personal-finance AI agent on Telegram, Indonesian language. Status: draft v1. This file = WHAT/WHY. HOW = `docs/spec/*`. Order of work = `TASKS.md`.

## Purpose

1. **Primary (v1): learning.** Build a real agent covering tool calling, ReAct, short- and long-term memory, an agent framework (LangGraph core), custom tools.
2. **Secondary (post-v1): product** for Indonesian individuals/freelancers. Build for yourself first, use 4+ weeks, then decide.

## Problem

People log expenses manually or not at all; form-based apps are tedious and don't understand everyday Indonesian ("makan siang 35rb di warteg"). Result: incomplete data and unanswered questions like "where am I overspending?".

## Users

- Primary: Keina (developer, Telegram user).
- Secondary (validate later): young professionals / freelancers; mixed personal+work spend, e-wallets, minimarket receipts.

## Goals

| # | Goal | Measure |
|---|---|---|
| G1 | Logging is as easy as sending a chat | p50 message→confirmation < 6s (text) |
| G2 | Agent picks/fills tools correctly | ≥ 90% on eval suite (`EVALS.md`) |
| G3 | Reliable receipt extraction | ≥ 85% correct (total + merchant) on 20 real receipts |
| G4 | Real usage | Keina logs ≥ 5 days/week for 4 weeks |
| G5 | Controlled cost | Cost/message tracked; target ≤ US$0.50 per active user/month at ~100 msgs (hypothesis, validate in Phase 4) |

## Non-goals (v1)

Income/transfers/balances (scope) · bank/e-wallet integration (legal/security risk) · WhatsApp (paid API, slow onboarding) · web dashboard · multi-agent inside the core app (explored only in `experiments/`) · investment advice.

## Learning scope → feature

| Skill | Delivered by |
|---|---|
| Tool calling | all tools in `spec/tools.md` |
| ReAct loop | multi-step questions (R-07); agent↔tools graph |
| Short-term memory | Postgres checkpointer per `thread_id` + trimming |
| Long-term memory | pgvector `memories` + save/recall/forget tools |
| Framework | LangGraph (core); CrewAI/AutoGen as separate experiments |
| Custom tools | 12 tools in `spec/tools.md` |

## Requirements

### P0 (v1)

| ID | Requirement | Acceptance |
|---|---|---|
| R-01 | Log expense from Indonesian text | "makan siang 35rb di warteg" → 1 expense: 35000 IDR, makan-minum, date = today (Asia/Jakarta), one-line confirmation |
| R-02 | Indonesian amount parsing | `35rb`, `35k`, `35 ribu`, `1,5 juta`, `Rp 1.250.000` parse correctly; ambiguous/≤0 amounts → clarification question, no write |
| R-03 | Query + aggregation | Totals/by-category/by-period match reference SQL |
| R-04 | Deterministic periods | "this month", "yesterday", "last week" resolved by `resolve_period`, not LLM guesswork |
| R-05 | Short-term memory | "change the last one to 40rb" updates the right expense |
| R-06 | Long-term memory | A stated preference/rule is used in a new thread without being asked |
| R-07 | Multi-step ReAct | Period comparison triggers ≥2 sequential tool calls; answer consistent with data |
| R-08 | Human-in-the-loop | `delete_expense` and receipt-sourced saves require explicit inline-button approval; rejection = no side effects |
| R-09 | Receipt extraction | Photo → total, merchant, date, items; shown for confirmation before saving |
| R-10 | User isolation | No path where LLM-supplied args reach another user's data |
| R-11 | Step limit, graceful errors | Loop capped by `MAX_AGENT_STEPS`; user sees a friendly message, never a stack trace |
| R-12 | Developer CLI | Full agent runs without Telegram |
| R-13 | Observability | Every LLM/tool call emits an OpenTelemetry span (tool name, duration, status, tokens) |
| R-14 | Allowlist | Only `TELEGRAM_ALLOWED_USER_IDS` are served |

### P1

R-20 budgets with 80%/100% warnings · R-21 foreign currency via `get_exchange_rate` (stored with rate) · R-22 scheduled weekly/monthly report · R-23 bulk correction with approval · R-24 CSV export.

### P2 (don't block architecture)

Full multi-user + billing (schema is already `user_id`-scoped) · income/transfers · WhatsApp (interfaces are modular) · web dashboard · recurring bill reminders.

## Metrics

Leading: msgs/day, eval tool-call accuracy, agent error rate, HITL rejection rate, p50/p95 latency.
Lagging (4+ weeks): active days/week, corrections per 100 logs, cost per active user, memories actually reused.

## Open questions

| Question | Owner | Blocking |
|---|---|---|
| `multilingual-e5-small` vs `bge-m3` for Indonesian on a small VPS | Eng, experiment T-303 | No |
| Is the configured vision model (OpenAI-compatible `image_url` input) enough for faded thermal receipts or is preprocessing needed | Eng, test on 20 receipts in T-404 | No |
| Column encryption, retention, UU PDP compliance before external users | Legal/Eng | **Yes, for product (not for personal v1)** |
| Subscription price vs token cost | Business, after Phase 4 data | No |

## Phasing

Phases 1–4 = v1 (sequential), Phase 5 optional. Rough estimate 6–8 weeks while working full-time. No hard deadline.
