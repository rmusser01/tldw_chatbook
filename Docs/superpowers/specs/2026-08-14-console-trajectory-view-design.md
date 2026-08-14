# Console Trajectory View — Design

Date: 2026-08-14
Status: Approved direction (this doc pending user review)
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
  event_kind      TEXT NOT NULL,              -- assistant | tool_call | tool_result |
                                              -- user | compaction
  step_started_at REAL,                       -- unix seconds; NULL when unknown
  first_token_at  REAL,
  completed_at    REAL,
  model           TEXT,
  provider        TEXT,
  PRIMARY KEY (message_id, event_kind)
  first_token_at  REAL,
  completed_at    REAL,
  model           TEXT,
  provider        TEXT
);
CREATE INDEX IF NOT EXISTS idx_trajmeta_conv_seq ON message_trajectory_metadata(conversation_id, seq);
```

Design points:

- **Sidecar, not ALTER on `messages`**: trajectory facts evolve independently
  of the synced message core. Future columns (timeline filter ranges, lineage
  seqs, cost roll-ups) are added here via new migrations without touching sync
  triggers. This is the long-term-stability/extendability trade: stable synced
  core, evolvable local edge.
- **NULLable timing**: older messages and non-streamed records simply have
  NULL timing; the view renders blanks (mirrors dsh: never fabricate
  durations for in-flight rows).
- **`seq`** is assigned at write time (max+1 per conversation) so the ledger
  has a stable total order even when wall-clock timestamps tie.
- **Turn boundaries are derived, not stored as rows**: a turn starts at each
  user (or compaction) record; no `turn_marker` rows exist.
- **Tool calls are not message rows** (they are embedded in assistant
  content). A sidecar row with `event_kind='tool_call'` reuses the *parent
  assistant message's* `message_id` — `(message_id, event_kind)` is the
  effective identity — and `seq` orders multiple tool calls within one
  assistant step. `tool_result` rows key on the tool message's own id.
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
  model/provider, full tool payload access (`tool_output_full`).
- Search box filtering rows (turn headers match if any child matches).
- Live tail-follow: subscribes to the Console store/stream completion events;
  follows the tail unless the user has scrolled up (follow suspends, resumes
  via a footer action).
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
