---
id: TASK-34958
title: Gate-the-0.2s-Console-transcript-sync-tick-on-a-store-change-counter
status: Done
assignee: []
created_date: 2026-10-13 14:48
labels:
- performance
- ui-perf
- wave-2-streaming
dependencies: []
priority: high
updated_date: 2026-10-13 17:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While a run is active, every 0.2s transcript-sync tick shallow-copies ALL session messages (_snapshot, ~50 fields each), re-groups the full transcript (set_messages, chat_screen.py:18111), rebuilds an N-tuple fingerprint, and walks all messages for image/generation/video specs and citations — all BEFORE the refresh_key gate at chat_screen.py:18249 which only skips the DOM reconcile. Cost scales linearly with transcript history at 5Hz for the whole stream, so long conversations burn CPU and can drop frames while streaming even though only the tail row changed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] A cheap store change indicator short-circuits the tick when nothing changed since the last processed version
- [x] Per-tick work no longer scales with the unchanged transcript prefix while streaming
- [x] Transcript rendering behavior unchanged (existing transcript tests pass)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Survey ConsoleChatStore mutation surface for a version/observer seam
2. Add a cheap store version counter bumped at every mutation
3. Short-circuit the 0.2s transcript tick on (session, version)
4. Tests + targeted runs + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- The store already had the right seam: `display_projection_revision(session_id)` ("constant-time invalidation token for screen projections"), already consumed by two chat_screen cache keys. Coverage was partial — stream chunks and content folds bump it via the speech-revision fence, and `set_message_usage` bumped directly.
- Widened coverage (`_bump_display_revision_for_message` helper; extra bumps only cost one re-sync, never staleness): `append_message`, `_mark_message_complete` / `_mark_message_stopped` / `_mark_message_failed` (terminal status flips WITHOUT new content were the dangerous gap — the final status row would never have repainted), `_update_message_content`, `_delete_message`. `reset_stream_content` was already covered via its speech bump.
- Screen gate at the top of `_sync_native_console_transcript` (chat_screen.py), before snapshot materialization, projections, grouping, and spec walks: tuple of (active session id, display token, change-review projection key, send-ack present) — the projection's inputs can flip rendering without a store change, so `ConsoleChangeReviewProjection.revision_key()` (new, mirrors its internal cache key) joins the gate. Session switches reopen via the session-id component; token-read failures fall through ungated.
- Safety net: a forced full pass every 15th tick (~3s at the 0.2s cadence) bounds staleness from any mutation the token does not cover, and the existing refresh-key reconcile inside the pipeline remains the final arbiter of DOM writes.
- Tests: `Tests/Chat/test_console_store_display_revision.py` (4 tests: append/stream bump, terminal marks bump even without new content, edit/delete bump, reads alone never bump) and `Tests/UI/test_console_transcript_sync_gate.py` (4 tests: skip-on-unchanged, reopen on token bump, forced 15th tick, session switch) — 8/8 pass.
- Regression: `Tests/Chat/test_console_chat_store.py` 422/423 — the single failure is the pre-existing `test_hidden_db_legacy_cas_miss_with_unavailable_reader_marks_live_stale`; `Tests/UI/test_console_raw_cli_transcript.py` + `test_console_transcript_units_cache.py` 20/20.
- ADR required: no — gate over an existing invalidation token; consumers only become more conservative.
