# Console Trajectory View — Design

Date: 2026-08-14
Status: Implemented (2026-08-14)
Inspirational reference: deepseek-harness `packages/client/ui-trajectory` ("Trajectory" tab; server-side `traceSession`/`traceEvent` query APIs)

## Goal

Give the Console a trajectory (trace) view over a conversation: an event ledger
grouped by turn with tool-call nesting, a per-record inspector showing token
usage and timing, search, and live tail-follow during streaming. A brushable
duration timeline is planned as a follow-up task and is deliberately out of
scope here, but the data model must support it.

Scope: **all Console conversations** (character conversations included — they
are Console conversations with tool-calling capability).

Non-goals (v1):

- Brushable/zoomable timeline strip (follow-up task; data model must enable it).
- dsh-style append-only parallel event log. Chatbook's `messages` table +
  variant sets + compaction repository already form the normalized event
  history; we do not duplicate it.
- Synced trajectory data. All new storage is local-only, like `usage_json`
  and `metadata_json`.
- Editing or replaying from the trajectory view; it is read-only.

## What exists today (verified)

- Schema v37 (`ChaChaNotes_DB.py`). `messages` has `timestamp`,
  `last_modified`, local-only `usage_json` (v30) and `metadata_json` (v31).
- `ProviderUsage` (`Chat/provider_usage.py`) already normalizes
  uncached_input / cache_read / cache_write / output (+audio, partial) per
  provider payload, persisted per message via `update_message_usage`.
- Turn identity exists **in memory only**: `ConsoleChatMessage.turn_id`,
  assigned at persist time in `console_chat_store.py`.
- Forking = per-turn regenerated variants (`ConsoleVariantSet`), persisted as
  message rows. Compaction = transactional, branch-safe
  (`Chat/console_context_compaction.py` + `console_context_repository.py`).
- Timing (step start / first token / completion) is captured **nowhere**.
- Console screen: `UI/Screens/chat_screen.py` with `UI/Console_Modules/*`;
  DataTable + footer-shortcut patterns in `evals_screen.py`; keybinding rules
  in ADR-031.

## Data model — schema v38 sidecar table

New local-only table `message_trajectory_metadata`, following the
`message_generation_metadata` (v24→25) precedent:

```sql
CREATE TABLE IF NOT EXISTS message_trajectory_metadata(
  message_id      TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  turn_id         TEXT NOT NULL,
  seq             INTEGER NOT NULL,           -- per-conversation monotonic order
  event_kind      TEXT NOT NULL,              -- user | assistant | tool_call | tool_result
  step_started_at REAL,                       -- unix seconds; NULL when unknown
  first_token_at  REAL,
  completed_at    REAL,
  model           TEXT,
  provider        TEXT,
  payload_json    TEXT,                       -- tool records ONLY: name/args/result
                                              -- (incl. full untruncated output);
                                              -- NULL for user/assistant rows
  PRIMARY KEY (message_id, event_kind, seq)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trajmeta_conv_seq
  ON message_trajectory_metadata(conversation_id, seq);
CREATE INDEX IF NOT EXISTS idx_trajmeta_msg ON message_trajectory_metadata(message_id);
```

Design points:

- **Sidecar, not ALTER on `messages`**: trajectory facts evolve independently
  of the synced message core. Future columns (timeline filter ranges, lineage
  seqs, cost roll-ups) are added here via new migrations without touching sync
  triggers. This is the long-term-stability/extendability trade: stable synced
  core, evolvable local edge.
- **The sidecar is the sole persisted home for tool records.** Verified:
  TOOL-role messages are deliberately session-only display markers in
  `console_chat_store.py` (never persisted, never tree nodes — the
  TOOL-marker invariant), and `tool_output_full` is likewise session-only.
  Writing tool records into `messages` would violate that invariant, so
  `tool_call`/`tool_result` rows live entirely in this table, with
  `payload_json` carrying the tool result, capped at 256 KiB with a
  `{"truncated": true}` marker beyond that (full output stays available live
  in-session). The trajectory view becomes the only place historical tool
  output is reviewable — a feature, not a workaround. Both kinds key on the
  *parent assistant message's* `message_id`; `seq` orders multiple tool calls
  within one assistant step.
- **PK is `(message_id, event_kind, seq)`**: one assistant message may emit
  several tool calls, so `(message_id, event_kind)` alone would collide. The
  unique `(conversation_id, seq)` index enforces ledger ordering; writes are
  upsert-by-seq.
- **NULLable timing**: older messages and non-streamed records simply have
  NULL timing; the view renders blanks (mirrors dsh: never fabricate
  durations for in-flight rows).
- **`seq`** is assigned at write time (max+1 per conversation) *inside the
  same transaction as the insert*, so concurrent writers (hands-free,
  compaction auxiliary turns) cannot collide.
- **Turn boundaries are derived, not stored as rows**: a turn starts at each
  user record. Compaction markers are likewise NOT sidecar rows — compaction
  produces no persisted message row; the projection reads compaction
  transactions from `console_context_repository` and renders them as
  between-turn markers.
- **Branch semantics**: the ledger renders the **active path** by default —
  derived by walking `parent_message_id` from the persisted local-only
  `conversations.active_leaf_message_id` (v23→24) to the root.
  Superseded variants (tree siblings off the active path) are not top-level
  rows; the inspector of a record with variants lists them (contents +
  selection state), keeping forking history visible without cluttering the
  ledger.
- **Soft-deleted messages are excluded**: the projection joins `messages`
  and filters `deleted = 0`; orphaned sidecar rows are ignored (`ON DELETE
  CASCADE` only fires on hard deletes).
- Migration: `chachanotes_v37_to_v38_message_trajectory_metadata.sql` +
  PRAGMA-checked Python runner + per-migration test (existing pattern).

## Timing capture

In `console_chat_controller.py` streaming path (`_stream_assistant_response_inner`,
`_attach_stream_usage` vicinity): record step start (before provider call),
first-token time (first chunk arrival), completion (stream end), then persist
through the same seam that writes `usage_json`. Tool-call records get
start/completion from tool executor boundaries; user/compaction records get
`step_started_at` only. No provider-layer changes required.

## Projection module

New `tldw_chatbook/Chat/trajectory.py` (pure, no Textual imports):

```
derive_trajectory(messages, usage_by_id, traj_rows, variant_sets, compaction_records)
    -> TrajectorySnapshot
```

- Folds into `turns -> records`, nesting `tool_call`/`tool_result` under the
  assistant step within the turn (dsh `deriveTrajectoryLayout()` is the
  reference shape).
- Each record carries: kind, content preview, token facts
  (input/cache_read/cache_write/output), timing facts, model/provider,
  variant annotation (superseded variants shown but marked), compaction
  markers between turns.
- Fully unit-testable with plain dataclasses/factories.

## UI — Trajectory screen

Launched from the Console (keybinding on `chat_screen.py`, single-letter per
ADR-031; footer hint registered via `register_footer_shortcuts`):

- `DataTable` ledger: one row per record; turn-header rows with collapse
  (per-turn collapse state), nested tool rows indented under assistant steps.
- Selection inspector pane: tokens (input / cache read / cache write /
  output), timing (step start → first token → completion, derived duration),
  model/provider, full tool payload access (from the sidecar's `payload_json`
  for history; the live in-memory `tool_output_full` while the session is
  open).
- Search box filtering rows (turn headers match if any child matches).
- Live tail-follow: the screen polls a public payload-revision getter on the
  Console store via `set_interval` (there is no observer bus; the trajectory
  write path bumps the existing per-session revision counter so polling sees
  changes) and refreshes in a worker; follows the tail unless the user has
  scrolled up (follow suspends, resumes via a footer action).
- Keybindings/footer per ADR-031; modal opens with safe Escape.

## Error handling & performance

- Projection runs in a worker (`run_worker`) for large conversations; ledger
  renders incrementally (DataTable is virtualized; load newest page first,
  "load earlier" control at top mirrors dsh pagination).
- Conversations predating v38 render fine — timing columns are blank; turn
  grouping falls back to `turn_id` derivation from in-memory restore or
  timestamp adjacency heuristic (defined once, in the projection module).
- Missing sidecar rows never block rendering.

## Testing

- Migration test (`Tests/DB/test_chachanotes_trajectory_metadata_migration.py`).
- Unit tests: timing capture seam, `derive_trajectory` (turn grouping,
  nesting, variants, compaction, NULL timing, tie-ordering by `seq`).
- UI tests: screen mounting, collapse, inspector contents, search filter,
  tail-follow suspend/resume; footer-hint governance test per ADR-031 suite.

## ADR

Required: schema + cross-module decision.
Path: `backlog/decisions/066-console-trajectory-view-and-trace-metadata.md`
(created before implementation; linked from the backlog tasks).

## Follow-up (separate tasks, out of scope here)

1. Brushable/zoomable timeline strip widget (uses `seq` + timestamps).
2. Optional: lineage view over variants/compaction (dsh `traceEvent` analog).
