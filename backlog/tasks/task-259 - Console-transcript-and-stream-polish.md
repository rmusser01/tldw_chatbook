---
id: TASK-259
title: Console transcript/stream polish (signature cache, buffer collapse, targeted RAG-launch card, inspector rows)
status: In Progress
assignee: ['@claude']
created_date: '2026-07-16 14:30'
labels: [performance, console]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four bounded improvements from the audit: _transcript_rows re-derives every row per changed tick (measured 0.86ms@200 msgs → 5.15ms@1000; cache per-message signatures keyed on id/content-len/status); _materialize_stream_buffer re-joins the whole chunk list per tick (collapse after join); _stage_console_library_rag_launch recomposes the ENTIRE ChatScreen for one pending card (5729-5731 — make it a targeted widget; several _build_console_*_state builders read _pending_console_launch_context, so verify); ConsoleRunInspector per-row updates instead of wholesale recompose. Depends on task-251 landing first (shared code region). Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P3 D1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transcript row derivation is O(changed messages) per tick with correctness preserved for delete/reorder/variant-switch
- [x] #2 Library-RAG launch no longer recomposes the screen
- [x] #3 Inspector updates changed rows only
- [ ] #4 Streaming behavior verified live (rig)
<!-- AC:END -->

## Implementation Plan

1. `ConsoleTranscript` (Widgets/Console/console_transcript.py): add a per-message
   signature cache keyed by message id, guarded by a cheap token
   (status, selected, active-content str ref, variant index/ids, attachment
   metadata, image byte length). `_transcript_rows` consults the cache so the
   expensive `_message_render_text` derivation runs only for changed messages.
   Prune the cache in `set_messages` (delete); reorder needs no invalidation
   (id-keyed); variant-switch changes the token (selected_index + content ref).
   Add `message_signature_compute_counts()` / cache-ids accessors for tests.
2. `ConsoleChatStore._materialize_stream_buffer` (Chat/console_chat_store.py):
   after joining, collapse the chunk list to the single joined string
   (`buffer[:] = [joined]`, materialized count = 1) so each tick joins only
   chunks that arrived since the last materialize.
3. `_stage_console_library_rag_launch` (UI/Screens/chat_screen.py): replace
   `refresh(recompose=True)` with a targeted sync: swap the
   `#console-pending-launch-card` / `#console-live-work-source-readiness`
   container in the inspector rail body (await remove → mount, scheduled via
   `call_later`), sync the staged-context tray (new `sync_state`), then reuse
   the existing targeted sync suite (`_sync_console_control_bar` +
   `_sync_console_workspace_context` + `_sync_console_settings_summary` +
   `_sync_console_mode_bar`). Fall back to full recompose if the Console shell
   isn't mounted. Move the blocked-outcome auto-open-inspector flag set BEFORE
   the stage call so rail auto-open still applies without recompose.
   Audit every `_pending_console_launch_context` reader (list in notes).
4. `ConsoleRunInspector.sync_state`: compute a structural key (ordered rendered
   row ids + full action tuples + dictionary rows/actions); when unchanged,
   update only changed row Statics in place (text + status class + summary);
   otherwise keep the widget-level recompose. Track `recompose_count` for tests.
5. Tests: transcript O(changed)/delete/reorder/variant-switch/equal-length-edit
   pins; store buffer-collapse pins; launch staging no-screen-recompose pins;
   inspector row-level-update pins. Run Console/chat-screen suites + Tests/Chat.

## Implementation Notes

**Status: AC #4 (live rig streaming verification) is PENDING controller QA —
task intentionally left In Progress.** ACs #1-#3 are implemented and pinned by
tests.

1. **Transcript signature cache** (`Widgets/Console/console_transcript.py`):
   `_cached_message_row_signature` keyed by message id, guarded by a cheap
   token of every render input (role, status, selected, active-content str
   ref, variant selected_index + id tuple, attachment metadata incl. byte
   LENGTHS not bytes, attachment_label/image mime). Unchanged messages hit
   Python's identity fast-path on the shared str refs; content is compared by
   VALUE (an equal-length edit still misses — pinned by test). Cache pruned in
   `set_messages` (delete); reorder needs no invalidation (id-keyed, position-
   free signatures — reconciler move_child handles order); variant switch
   changes the token. Bench (audit methodology, 1000 msgs, changed tick):
   5.15 ms -> 1.54 ms; residual is cheap `_TranscriptRow` tuple construction,
   the expensive Content assembly is O(changed). Test seams:
   `message_signature_compute_counts()` / `message_signature_cache_ids()`.
2. **Stream-buffer collapse** (`Chat/console_chat_store.py`):
   `_materialize_stream_buffer` now does `buffer[:] = [joined]` +
   materialized count 1 after each join, so each tick joins only new chunks.
   In-place list mutation preserves outstanding refs; invariant
   `"".join(buffer) == full content` pinned across complete/stop/retry-seed/
   variant-stream flows. Counts and buffers are always popped together at
   every terminal/reset site (audited), so the count-semantics change is safe.
3. **Targeted Library-RAG staging** (`UI/Screens/chat_screen.py`):
   `_stage_console_library_rag_launch` no longer calls
   `refresh(recompose=True)` (that was the ONLY screen-recompose call in the
   file); it now runs `_sync_console_pending_launch_surfaces` — full
   `_pending_console_launch_context` reader audit lives in that method's
   docstring. Readers and their freshness paths: `_build_console_control_state`
   (control bar + workbench strips + hidden mode bar via
   `_sync_console_control_bar`/`_sync_console_mode_bar`),
   `_build_console_inspector_state` (inspector + composer Chatbook action,
   inside `_sync_console_control_bar`), `_build_console_staged_context_state`
   (NEW `ConsoleStagedContextTray.sync_state` + rail badges/auto-open via rail
   state + settings context estimate via `_sync_console_settings_summary`),
   `_current_console_workspace_context` (workspace/details trays via
   `_sync_console_workspace_context`), the pending-launch/source-readiness
   card (async `_apply_console_live_work_card_swap`: await remove -> mount
   after `#console-run-inspector`, scheduled via `call_later` with a dedupe
   flag; card builders refactored out of the compose renderers), and
   on-demand event readers (help panel, draft sync, send-blocked reason,
   card buttons) which cannot go stale. The blocked-outcome
   `_pending_console_launch_auto_open_inspector = True` moved BEFORE the
   stage call (the old deferred recompose read it late; the sync path reads
   it during staging). Fallback to full recompose only when the Console shell
   isn't mounted.
4. **Inspector per-row updates** (`Widgets/Console/console_run_inspector.py`):
   `sync_state` computes a structural key (rendered row ids in compose order +
   full action tuples + dictionary shape); equal-structure changes update row
   Statics in place (text + status class + summary, changed rows only),
   anything structural recomposes the widget (widget-level only, never the
   screen). Gotcha found while fixing a real regression
   (`test_console_native_missing_key_blocks_before_clearing_generic_draft`):
   the rail-collapse cascade stamps `display=False` +
   `_console_rail_prior_display` on descendants; a wholesale recompose
   implicitly drops that state (fresh children), so the in-place path must
   restore it explicitly — and DEFERRED (`call_after_refresh`) to match the
   recompose path's timing relative to same-tick rail cascades.
   `recompose_count` added as a test seam.

**Tests added**: 6 transcript-cache pins (test_console_native_transcript.py),
5 store buffer-collapse pins (test_console_chat_store.py), 3 staging pins
incl. screen-recompose spy + widget-identity + auto-open
(test_console_live_work_handoffs.py), 6 inspector row-update pins (NEW
Tests/UI/test_console_run_inspector.py). Suites run: Tests/Chat (957 passed /
69 skipped), Console/chat-screen UI suites (all green; two KNOWN flaky
conversation-browser-search tests — `..._ignores_stale_results`,
`..._blank_query_clears_error_cache` — each failed once under full-suite load
and pass in isolation and on re-run; full chat-flow file passes clean on
repeat with this diff and at base).
