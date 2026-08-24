---
id: TASK-21121
title: >-
  The 0.2s run tick gained a full per-tick message-snapshot pass for
  changed-files scope
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-24 00:25'
labels:
  - performance
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21121).

`_console_changed_files_scope()` (chat_screen.py:11467-11497) runs on every 0.2 s run tick and
calls `messages_for_session` - a full shallow-snapshot of every session message
(console_chat_store.py:2858-2865) - then reverse-scans; its own docstring concedes worst-case
O(messages) per tick when the session has no change-review marker, which is the common case.
Combined with the cost path this makes >=2 full snapshot passes per tick during a run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The newest run-id (or marker presence) is memoized on the store and bumped on marker append (pattern: the token-estimate cache), so the no-marker common case is O(1) per tick
- [x] #2 A counter probe during a streamed reply in a large session shows the snapshot-pass reduction; run-tick behavior unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Census every write to `_messages_by_session` and every site that can introduce a `change_review_run_id`-carrying message; record it in the notes.
2. Add a verified memo on ConsoleChatStore (`newest_change_review_run_id(session_id)`) in the TokenEstimateCache shape: correctness verified per hit against a cheap signature (the live view list object identity + its length), never an invalidation protocol. Reverse-scan the INTERNAL list on a miss -- no `messages_for_session` snapshot, no `_snapshot`/`replace` copy per message.
3. Repoint `_console_changed_files_scope()` at the store accessor; keep the guard tuple shape and semantics identical.
4. Red-first counter probe: count `messages_for_session`/`_snapshot` calls across simulated run ticks on a several-hundred-message session, before vs after.
5. Control arm: prove the scope is still CORRECT with a marker present (newest of several, off-path marker dropped, resume overlay, marker appended mid-run) -- a memo that always returned None must fail.
6. Verify teardown: close_session / restore_state / rollback paths do not resurrect or leak a memo entry, and the 0.2s tick's self-stopping behavior is unchanged.
7. Run the Console changed-files / store / run-tick suites; A/B any red against base fb0a9601e; preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the changed-files guard's per-tick transcript copy with a verified memo on the store.

## What changed

`ChatScreen._console_changed_files_scope()` no longer calls `messages_for_session()`. It asks the store for one string via a new `ConsoleChatStore.newest_change_review_run_id(session_id)`. The guard tuple's shape, values, KeyError contract and dispatch behaviour are unchanged; only the derivation moved.

The old path was `for m in reversed(store.messages_for_session(sid))` with an early break. The early break bought nothing: `messages_for_session` `dataclasses.replace`-copies EVERY message before the loop starts, so the "worst case is O(messages)" the old docstring conceded was in fact the cost of every tick, marker or no marker.

## The memo (pattern: `TokenEstimateCache`, task-15451)

Single slot on the store: `(session_id, view_list, len(view_list), newest_run_id)`. A hit is served only after re-checking that full signature — **verified, not invalidated**, exactly the property that makes `TokenEstimateCache` locally reasonable ("there is no invalidation protocol to get wrong"). A miss reverse-scans the store's INTERNAL view list, reading only `change_review_run_id`; nothing is copied on a miss either.

The signature is exact because of two store invariants, both re-verified for this task:

1. `change_review_run_id` is write-once — set in the `ConsoleChatMessage` constructor and never reassigned (grep: no `.change_review_run_id =`, no `setattr`, no `replace(...)` touching it, anywhere in `tldw_chatbook` or `Tests`). Streaming mutates `.content` in place, which the answer does not depend on.
2. Every write to `_messages_by_session[sid]` either installs a NEW list object or appends to the existing one. Identity catches the first, length catches the second.

The list OBJECT is held in the memo (not its `id()`), so a memoized list cannot be freed and have its address recycled — the `is` check is sound.

## Invalidation census (complete; every path is covered by verification, none by a hook)

Marker introduction — the only two ways a `change_review_run_id` reaches a view:
- `append_message(role=TOOL, change_review_run_id=...)` (console_chat_store.py TOOL branch) — live agent markers from `ConsoleAgentBridge._append_change_markers`. In-place `.append`; the ONLY in-place mutation of a view list in the codebase → caught by LENGTH.
- `apply_resume_marker_overlay(...)` — resume markers rebuilt from `AgentRunsDB` by `ConsoleAgentBridge.resume_marker_messages`, written straight into the view, bypassing `append_message` entirely → new list → caught by IDENTITY. (An invalidate-on-append design would have missed this one; there is a test for it.)

Every write to `_messages_by_session[session_id]` (all inside `console_chat_store.py`; grepped `tldw_chatbook`, `Tests`, `Helper_Scripts` — no external writer):
- `create_session` (new `[]`) → IDENTITY
- `apply_resume_marker_overlay` (new list) → IDENTITY
- `restore_state` (clear + new `[]` + `_ingest_linear_messages`) → IDENTITY
- `append_message` TOOL branch (in-place append) → LENGTH
- `_recompute_active_path` — the documented SINGLE writer, always a fresh list → IDENTITY. Its 11 callers are the whole mutation surface: send, `create_sibling` (regenerate / edit-and-resend), generation/video/variant appends, `_purge_descendants_invalidated_by_edit` (message edit), `delete_message`, `discard_provider_continuation`, `set_active_leaf` (branch switch / rewind / swipe), `_ingest_linear_messages` and `_ingest_full_tree` (conversation load, import, restore).
- `close_session` / `rollback_created_pristine_session` (dict pop) → a later query for that session raises `KeyError`, same contract as `messages_for_session`; other sessions are unaffected because the slot is session-keyed.

Not changed, and deliberately not composed with: the screen-side guard resets (`handle_console_turn_file_card_notes_changed`, `_on_console_change_review_dismissed`) still force a recompute on note mutations, and `_console_derivation_scope` (task-15452) is a PER-PASS memo — routing through it would still pay one full copy pass per tick, which is the defect.

## Teardown / tick behaviour

No new timer, worker, or deferred callback. The memo is one attribute on the store and dies with it; it retains at most one view list, replaced by the next query for another session. Nothing in the 0.2s poll's arm/disarm path (`_start_console_transcript_sync_timer`, `CONSOLE_ACTIVE_RUN_STATUSES`) is touched, and the guard tuple is byte-identical, so when the tick stops is unchanged.

Materialization parity was measured, not assumed: `messages_for_session` also folds buffered stream chunks (`_materialize_stream_buffer` → `_persist_pending_message_if_ready`). Instrumented on a mounted Console over ONE full `_sync_native_console_chat_ui()` tick with a live stream, base made 6,258 materialize calls over 21 distinct messages, after made 6,237 over the same 21 — exactly one full pass fewer, the streaming row still materialized, content identical. The transcript render and cost chip still materialize every tick.

## Evidence

Counter probe, 25 simulated run ticks with a reply streaming, measuring the guard alone (base `fb0a9601e` → this branch):

| session | marker | `messages_for_session` | message copies | guard wall time |
|---|---|---|---|---|
| 400 msgs | none | 25 → 0 | 10,025 → 0 | 32.08 ms → 0.02 ms |
| 400 msgs | present | 25 → 0 | 10,050 → 0 | 29.88 ms → 0.01 ms |
| 40 msgs | none | 25 → 0 | 1,025 → 0 | 2.87 ms → 0.01 ms |
| 40 msgs | present | 25 → 0 | 1,050 → 0 | 2.88 ms → 0.01 ms |

The reported scope tuple is identical before and after in all four arms. Numbers are recorded in `Docs/Design/2026-08-22-holistic-perf-review.md` ("Landed after close-out").

`Tests/UI/test_console_changed_files_scope_memo.py` (14 tests, new): 3 cost tests parametrized over two session sizes so the assertion states the SHAPE (flat in session size); 6 control-arm correctness tests; 5 store lifetime/teardown tests. Red-first at base: 7 fail there (2 on the per-tick count, 4 on the missing store API, 1 both). Mutation-tested three ways — `return None` kills 8 tests; dropping the length check kills 1; dropping the identity check killed 0 until `test_scope_re_derives_when_a_same_length_branch_replaces_the_view` was added for it.

## Discovered, not fixed

`restore_state` clears `_messages_by_session` but not `_tool_markers_by_session`, so an anchor-`None` TOOL marker from the replaced state is re-spliced into the head of every restored session's view — including change-summary markers. Confirmed at base with the pre-21121 reverse scan, so pre-existing and unchanged here. Filed as TASK-21311.

## Files

- `tldw_chatbook/Chat/console_chat_store.py` — memo slot + `newest_change_review_run_id`
- `tldw_chatbook/UI/Screens/chat_screen.py` — `_console_changed_files_scope` repointed
- `Tests/UI/test_console_changed_files_scope_memo.py` — new
- `Docs/Design/2026-08-22-holistic-perf-review.md` — measurements
- `backlog/docs/lessons-testing-evidence.md` — the early-break-over-a-copy trap + the fixture-billed-counter trap
<!-- SECTION:NOTES:END -->
