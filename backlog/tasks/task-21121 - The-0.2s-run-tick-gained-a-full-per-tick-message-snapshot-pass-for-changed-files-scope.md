---
id: TASK-21121
title: >-
  The 0.2s run tick gained a full per-tick message-snapshot pass for
  changed-files scope
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-24 00:36'
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

`Tests/UI/test_console_changed_files_scope_memo.py` (new; 15 test functions = 18 node ids after parametrization): 2 cost functions (5 node ids) parametrized over session size AND marker presence so the assertion states the SHAPE (flat in session size) and "always return `None`" fails the cost half too; 6 control-arm correctness functions; 1 scan-race function; 6 store lifetime/teardown functions. Red-first at base `fb0a9601e`: 11 node ids across 9 functions fail there. Mutation-tested four ways — `return None` kills 13 node ids across 12 functions; dropping the length check kills 1; dropping the identity check killed 0 until `test_scope_re_derives_when_a_same_length_branch_replaces_the_view` was added for it; reverting the length hoist kills exactly the scan-race test.

## Review fix round

A reviewer independently reproduced the census, the probe (all eight arms, byte-identical scope tuple), the write-once property and the console red set, fuzzed 138,565 probes with 0 violations (non-vacuous: identity dropped → 465 violations, length dropped → 4,158) — and found one real hole, fixed here.

**MAJOR — the signature was sampled after the work it describes.** `len(view)` was evaluated *after* the reverse scan, so an append landing in between recorded the post-append length beside the pre-append answer. Every later tick then passed the full signature check and served that stale value until the list object was replaced — for a settled run, potentially never, so the rail's `✎ N` badge would stop refreshing for good. This is genuinely reachable: `ConsoleAgentBridge._append_change_markers` appends through the `append_todo_marker` seam, which its own docstring says fires on the agent worker thread with **no `call_from_thread` marshalling**, while `run_reply` runs under `asyncio.to_thread` — so the append races the event-loop tick running the guard. Note the failure MODE is a regression even though the memo is new: base recomputed every tick and self-corrected on the next one; the memo made it durable.

Fix: hoist `length = len(view)` above the scan. `reversed()` snapshots the size at iterator creation, so the scan is already consistent with the pre-append length, and recording a length that is short can only cost an extra miss — never a stale hit.

Regression test `test_a_marker_appended_during_the_scan_is_not_memoized_away_forever` forces the interleaving deterministically with a `list` subclass whose `__reversed__` performs the append after creating the iterator. Its red-first evidence is the **mutation** (revert the hoist → it, and only it, fails), not the base run: at base the fixture cannot fire at all (base reverses a *snapshot copy*, never the store's own list), so its base red is `assert racing.fired`, i.e. fixture inapplicability rather than the bug. Base does not have this bug.

Also in this round:

- **Retention.** `close_session` and `rollback_created_pristine_session` now drop the slot via `_drop_newest_change_review_memo`. The "next query for another session evicts it" argument fails exactly when the closed session was the ACTIVE one: `_console_changed_files_scope` then short-circuits on a falsy `active_session_id` and never queries again, so the slot pinned that session's whole view — every `ConsoleChatMessage` in it — for the life of the store. Dropping a memo can only cost a recompute, so this is hygiene, not an invalidation protocol.
- **Docstrings corrected.** Both "can never serve a stale run id" (store) and "can only change how long this takes, never what it returns" (screen) were true only *given* the sampling order; both now say so explicitly, so the next reader knows what the guarantee rests on.
- **`session_id` reworded.** Dropping that component kills zero tests and zero of the reviewer's 138,565 fuzz probes, because identity already covers it (a view list is uniquely owned by one session and the memo pins it alive). Kept as defence-in-depth, no longer described as load-bearing.
- **Counts corrected** above; every drift was conservative (the real evidence was stronger than claimed).

## Discovered, not fixed

`restore_state` clears `_messages_by_session` but not `_tool_markers_by_session`, so an anchor-`None` TOOL marker from the replaced state is re-spliced into the head of a restored session that reuses the SAME session id (the registry is keyed by session id, so unrelated sessions are unaffected) — change-summary markers included. Confirmed at base with the pre-21121 reverse scan, so pre-existing and unchanged here. Filed as TASK-21311.

## Files

- `tldw_chatbook/Chat/console_chat_store.py` — memo slot, `newest_change_review_run_id`, `_drop_newest_change_review_memo` + its two teardown call sites
- `tldw_chatbook/UI/Screens/chat_screen.py` — `_console_changed_files_scope` repointed
- `Tests/UI/test_console_changed_files_scope_memo.py` — new
- `Docs/Design/2026-08-22-holistic-perf-review.md` — measurements
- `backlog/docs/lessons-testing-evidence.md` — the early-break-over-a-copy trap, the fixture-billed-counter trap, and the sample-the-signature-first trap
<!-- SECTION:NOTES:END -->
