---
id: TASK-17169
title: Persist and track console selection feedback
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-16 13:57'
updated_date: '2026-08-17 06:29'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User request (2026-08-16): feedback sent via the selection menu (Request changes / LGTM / Comment) is currently ephemeral — dispatched as the next user message and forgotten. Track it durably, ideally in the session trace. Two candidate homes: (a) the trajectory trace sidecar (schema v38 message_trajectory_metadata, ADR-066 on dev / PR stack) as a new event_kind=user_feedback record carrying action, quoted selection, optional comment, and the anchor (message id / diff row key / turn id) — feedback becomes part of the run's auditable event ledger, queryable per turn; (b) the spec phase-4 transcript_annotations table (spec 2026-08-14 console-selection §3: (session_id, row_key)-anchored, quote + comment, soft-delete, inline badge UI) — feedback persists as row-anchored annotations with a visible marker. These serve different needs: trace events = chronological audit/history; annotations = inline review markers on rows. Recommendation to validate at planning: BOTH may be right — write the feedback event to the trajectory sidecar (audit) and upsert an annotation for Comment actions (inline marker); phase 4's row_key derivation spike (message:<id> / diff:<tool_call_id>) is the shared prerequisite. Decide the split in this task's plan; may merge with the phase-4 annotations task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Feedback events recorded durably (not ephemeral), surviving restart
- [x] #2 Option A evaluated: trajectory sidecar event_kind with anchor fields
- [x] #3 Option B evaluated: spec phase-4 annotations table
- [x] #4 Decision recorded (A, B, or both) with rationale + ADR if storage changes
- [x] #5 Feedback history viewable (trace viewer and/or inline badges per decision)
- [x] #6 Tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Decision (AC#4, maintainer, 2026-08-17): BOTH homes. Every feedback action
(Request changes / LGTM / Comment) writes a user_feedback event to the
ADR-066 trajectory sidecar (chronological audit; no migration -- event_kind
is unconstrained TEXT on the local-only v38 table). Comment actions
ADDITIONALLY persist a transcript_annotations row (spec 2026-08-14 §3)
rendered as an inline marker. Corrected input to that decision: sync
triggers are opt-in per table, so the annotations table is local-only
unless triggers are deliberately added -- sync was NOT a differentiator.

Slice 1 -- sidecar (all actions):
1. Anchor plumbing: ConsoleSelectionFeedbackRequested gains anchor_message_id from the origin row.
2. Store seam: ConsoleChatStore.record_feedback_event -> TrajectoryRowWrite(event_kind='user_feedback', seq=None auto-assign); skips (returns False) on unknown/unpersisted anchor or ephemeral session; never raises.
3. Screen hook: write after the comment modal resolves, before prompt_queue.dispatch; failures logged, never block the dispatch.
4. Rendering: projection nests feedback at depth 1 under its anchor (_NESTED_KINDS -- a message_rows bucket entry would displace the anchor's own row); timeline style; inspector feedback branch (payload is action/quote/comment, not tool-shaped).
5. Tests: DB round-trip incl. restart survival; projection nesting/preview/accumulation; screen-hook e2e incl. an UNMOCKED screen->store->SQLite test (harness store has no persistence -- inject a real one via _console_chat_store AND the controller's cached reference).

Slice 2 -- annotations (Comment only):
6. row_key spike (spec-mandated): inventory selectable row kinds for durable keys (message:<db_message_id>; diff rows via their tool message id); any kind without one is excluded with a hint.
7. Migration v39->v40: transcript_annotations table per the spec sketch (annotation_id PK, session_id, row_key, nullable message_id, quote_text, comment, timestamps, soft-delete), migration test.
8. DB + store seam: upsert/list/soft-delete; Comment dispatch path writes the annotation alongside the sidecar event.
9. Badge UI: inline marker on the anchored row + viewer popover.
10. Tests per slice; docs (user guide page + ADR-068 amendment already records the decision).
<!-- SECTION:PLAN:END -->

## Implementation Notes

Both slices complete.

**Slice 2 (annotations, Comment only).** Migration v39->v40 adds the
local-only `transcript_annotations` table -- (conversation_id, row_key)
anchor per the spike (the spec sketch's "session_id" is per-process and
would orphan every annotation on reload; the durable identity is the
persisted conversation). Spike result: plain/markdown rows of persisted
messages key as `message:<persisted_message_id>`; TOOL markers and diff
rows have no durable key today (marker invariant + session-only tool_diff)
and are excluded exactly as the spec anticipated -- consistently, the
sidecar write already skips them. Upsert is BY annotation id (repeated
review accumulates); soft-delete per conventions.
`ConsoleChatStore.record_feedback_annotation` mirrors the sidecar seam's
skip/never-raise contract; the screen hook writes it (Comment with a
non-empty note only) inside the same guard as the sidecar event.

**Inline markers.** The citation-sources pattern reused verbatim: a
screen-owned previews map keyed by NATIVE message id, pushed at the sync
tick, derived into a "Review note(s)" sub-row with the notes riding the
signature. Live Comment writes update the map immediately; a conversation
switch reloads it off-thread (exit_on_error=False) and re-keys persisted
message ids to current native ids, discarding stale in-flight results.

**Test evidence** (scoped per maintainer to touched code): selection e2e 56,
annotation markers 9, citation sources 55, trajectory capture/projection/
screen/timeline 91, migration 4, annotation store 5, DB suites 1226 -- all
green on the shipped tree; Tests/Chat full run at byte-identical baseline
parity with clean dev (same 14 failures, same params). Post-push Qodo
finding (duplicate-dispatch hazard) closed with the in-flight guard.

**Ripple fixes carried by this task** (all pre-existing red): the v39
landing's forgotten test pins (now tracking _CURRENT_SCHEMA_VERSION) and
sql_validation allowlist; the citation-sources SimpleNamespace fake missing
set_change_review_provider_factory -- the crashed session's un-named fourth
UI failure.

- `record_feedback_event` writes one `user_feedback` TrajectoryRowWrite per
  dispatched feedback; `seq=None` auto-assign keeps repeated feedback on one
  message from colliding on the `(message_id, event_kind, seq)` PK.
- Anchor = the origin row's native message id; the stored row keys off its
  persisted id. Unpersisted/unknown anchors and ephemeral sessions skip
  silently (False), and the screen hook never lets an audit failure block
  the actual dispatch.
- Two silent traps caught by tests (details in ADR-068 amendment 4): a
  feedback row bucketed as its anchor's own sidecar row displaces that row's
  timing/turn attribution (fixed via _NESTED_KINDS + depth-1 nesting), and
  the inspector's tool-shaped payload branch rendered a phantom `tool --`
  line for feedback payloads (fixed with a dedicated branch).
- Evidence includes an unmocked screen->store->SQLite e2e; the console test
  harness ships a persistence-less store, so the test injects a real DB via
  the screen's `_console_chat_store` setter AND the controller's cached
  store reference (the runtime swap alone leaves the controller writing to
  the old store).
