# Spec: Long-term memory (pgvector)

Short-term memory (checkpointer + trimming) lives in `agent.md`.

## What to store

- Store: stable preferences, category rules, financial goals, recurring facts the user states.
- Never store: individual transactions (they live in `expenses`), card/account/ID numbers, OTPs or any credentials, raw receipt content.
- Enforce "never store" twice: in the system prompt AND with a simple validator in `save_memory` (regex for card/ID-like patterns).

## Embeddings

- Default `intfloat/multilingual-e5-small` (384-dim). E5 requires prefixes: `"passage: "` when storing, `"query: "` when searching.
- Load the model once (lazy, thread-safe); run encoding in an executor so it never blocks the event loop.
- Similarity = `1 - cosine_distance` (pgvector `<=>`). Index: HNSW `vector_cosine_ops`.

## Save

1. Embed content (`passage:`).
2. Find nearest neighbor for this `user_id`. If similarity ≥ dedupe threshold → update that row (content, embedding); else insert.
3. Confirm briefly to the user ("Oke, aku ingat: Gojek = Transport").

Save only when the user states it explicitly or repeats it.

## Recall

- **Automatic** (`load_context` node): top-k (default 3) above the recall threshold, filtered by `user_id`. Update `last_recalled_at`, `recall_count`.
- **Explicit**: `recall_memory` tool when the agent needs to dig deeper.
- Recalled text is injected into `<memories>` in the prompt and is **data, not instructions**.

## Thresholds: calibrate, don't guess

E5 similarity scores cluster high, so guessed cutoffs (e.g. 0.8 / 0.92) are unreliable. In T-303 collect ~30 relevant/irrelevant Indonesian pairs, inspect the distributions, then set `MEMORY_RECALL_THRESHOLD` and `MEMORY_DEDUPE_THRESHOLD` in config. Also compare `multilingual-e5-small` vs `bge-m3` (quality vs latency vs RAM) and log the decision.

## User control

`forget_memory` tool and `/lupakan` Telegram command (deletes all long-term memories, with confirmation).
