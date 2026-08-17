# ADR-066: Console Trajectory View and Trace Metadata Sidecar

- Status: Accepted
- Date: 2026-08-14
- Spec: `Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md`
- Reference: deepseek-harness `packages/client/ui-trajectory` (Trajectory screen) and
  `packages/session-query` (`traceSession`/`traceEvent`)

## Context

The Console needs a trajectory (trace) view over conversations: an event
ledger grouped by turn with tool-call nesting, per-record token usage and
timing inspector, search, and live tail-follow. Verified substrate facts:

- Schema v37 already persists per-message token usage (`usage_json`) and
  provenance (`metadata_json`), both local-only.
- Timing (step start / first token / completion) is captured nowhere.
- `turn_id` exists only in memory (`ConsoleChatMessage.turn_id`).
- TOOL-role messages are **deliberately session-only display markers**
  (TOOL-marker invariant in `console_chat_store.py`); tool activity and
  `tool_output_full` are never persisted.
- Forking = variant tree siblings; compaction = transactional records in
  `console_context_repository` (no persisted message row).

## Decision

1. **New local-only sidecar table `message_trajectory_metadata` (schema
   v38)**, PK `(message_id, event_kind, seq)`, unique `(conversation_id,
   seq)`. It holds: turn identity, per-record timing, model/provider, and —
   for `tool_call`/`tool_result` records — the full payload including
   untruncated output in `payload_json`. Never synced.
2. **Tool records live only in the sidecar** — writing them to `messages`
   would violate the TOOL-marker invariant. The trajectory view is the only
   surface where historical tool output is reviewable.
3. **Timing captured in the Console controller streaming path** (step start,
   first chunk, completion) and persisted through the existing usage/metadata
   persistence seam. No provider-layer changes.
4. **Pure projection module `Chat/trajectory.py`**: folds messages +
   usage_json + sidecar rows + variant sets + compaction records into a
   `TrajectorySnapshot`. No Textual dependency. Turn boundaries derived; the
   ledger renders the active path with variants surfaced in the inspector;
   soft-deleted messages excluded.
5. **Trajectory screen launched from the Console** (ADR-031 keybindings):
   DataTable ledger with turn collapse, inspector, search, live tail-follow.
   Brushable timeline widget is a deliberate follow-up, enabled by the
   sidecar's `seq` + timestamps.

## Alternatives considered

- **JSON keys in existing `metadata_json`**: no migration, but weakest
  long-term extendability — lineage/timing facts deserve typed, evolvable,
  queryable storage (forking/compaction already exist and will grow).
- **ALTER `messages` with new columns**: couples trajectory facts to the
  synced core and its sync triggers; migrations on the sidecar are cheaper
  and independently evolvable.
- **dsh-faithful append-only event log table**: duplicates the normalized
  history `messages` + variants + compaction repo already record; doubles
  write paths for no v1-visible gain. Lineage/tracing views can be built
  later on the sidecar + variants if needed.

## Consequences

- One migration (v37→v38) with the standard runner + per-migration test.
- Local DB grows for tool-heavy conversations (full tool outputs). Accepted:
  local-only, and it restores reviewability of past tool activity.
- Conversations predating v38 render with blank timing and timestamp-derived
  grouping; no backfill.
