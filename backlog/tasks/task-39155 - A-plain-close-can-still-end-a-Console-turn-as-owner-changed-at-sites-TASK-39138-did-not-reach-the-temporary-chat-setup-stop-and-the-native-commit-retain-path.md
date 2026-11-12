---
id: TASK-39155
title: "A plain close can still end a Console turn as owner changed at sites TASK-39138 did not reach: the temporary-chat setup stop and the native-commit retain path"
status: Done
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 18:03
labels:
- follow-up
- console
- lifecycle
priority: low
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_native_commit.py
- tldw_chatbook/Chat/console_pre_provider_stop.py
dependencies:
- TASK-39138
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39138 and its review, round-12 branch fu/r12-ctlr). Read on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08; line numbers re-read on dev 3a6498ae14, 2026-11-09. Found by reading; not reproduced, and no test covers it. Line numbers are dev 3a6498ae14's unless marked.

**The fixed site.** TASK-39138 fixed one site where a session the user closed in place read as an ownership change. It added `AcceptedStopAuthority.closed_in_place` (round 12, `console_pre_provider_stop.py`), and the stopped-stream settlement now returns the closed-session result ("Session closed.", run state `STOPPED`). Before, it returned `BLOCKED` / `OWNER_CHANGED` and logged "Console submit requires attention".

**What a close does.** `begin_session_close` (`console_chat_controller.py:14458`) calls `_signal_stop(session_id=..., reason="session_close")` (`:14569`, `:14689`). `_signal_stop` (`:14960`) sets the run's own cancel event and marks the session's ordinary native commits `explicit_stop` (`:15013`). A close in place also removes the session from the store.

**Sites with the same shape.**
- **Temporary chat, agent pre-provider setup** (`console_chat_controller.py:25463`; round-12 stack `:25484`). This branch handles a cancelled reply in `_replies_in_pre_provider_setup` whose recovery is `EPHEMERAL_ACCEPTED`. It checks `setup_authority.is_current(self)` and, when that fails, returns `ConsoleSubmitResult(True, False, terminal_status=BLOCKED, blocker=OWNER_CHANGED)` with no copy. A closed temporary chat fails `is_current`, because its session is gone from the store.
- **Native-commit retain.** `_controller_retain_original_native_commit_recovery` (`console_native_commit.py:387-427`) returns `BLOCKED` / `OWNER_CHANGED` with "Saved turn retained for recovery." or "Prepared turn owner changed; draft kept.". It is called right after `_controller_ordinary_native_commit_current` (`:374-384`) comes back False, and that check also fails when the session was closed. The call sites:
  - the controller (`console_chat_controller.py:12452`, `:12922`, `:13051`, `:13098`);
  - `console_native_commit.py:517`, `:802`, `:836`, `:848` and `:1020-1021`.

Whether a plain close reaches each window in practice was not traced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 For each site, a test closes the chat inside that window (a temporary chat during agent pre-provider setup; an ordinary native commit in flight) and shows whether a plain close reaches it; a site no close can reach is named in the notes with the reason
- [x] #2 Where a plain close reaches a site, the turn ends with the closed-session result TASK-39138 gives ("Session closed.", run state `STOPPED`, no "Console submit requires attention" log); the test is red before the change
- [x] #3 A real ownership change (a replaced store, persistence or database, or a successor session with the same id) still gets the owner-changed result at each site
- [x] #4 The tests run on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. For each site, close the chat the way production does (`begin_session_close`, drain, `finalize_session_close`; the runtime's `close_session` with its grace) inside that window, and record the turn's result.
2. Where a plain close reaches an owner-changed or other wrong end, fix it at the shared function all callers route through; where it cannot, name the reason in the notes.
3. Keep a real ownership change (successor session, replaced store/persistence/database) owner-changed.
4. Put the tests in a whole-file target of a non-fast PR lane; three timings; size ratchet row.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What a plain close does.** `begin_session_close` fences the chat, signals stop and cancels its submit task at once; the chat is deleted only by `finalize_session_close`, which the runtime's `close_session` calls after its bounded drain (2 s grace) and after `_drain_ordinary_native_commits`. Tests: `Tests/Chat/test_console_durable_commit_window.py` (bootstrap_profile).

**Native-commit retain (`_controller_retain_original_native_commit_recovery`, all nine call sites): no plain close reaches it.** `finalize_session_close` refuses while an ordinary native commit owns the session ("An ordinary save still owns this session."), and the runtime drains each commit owner's retirement, which every path signals only after its ownership check, before it finalizes. So `_controller_ordinary_native_commit_current` always still finds the chat, and the close ends the turn as a stop. `test_a_chat_closed_during_its_native_commit_never_reads_as_an_owner_change[close]`: the commit is held on its thread, the close begins, the commit is released, the turn ends "Response stopped." (STOPPED, no blocker, no "Console submit requires attention"), then finalize deletes the chat. Green on base and after; no change at these sites.

**Temporary chat, agent pre-provider setup.** The `setup_authority.is_current` branch (`console_chat_controller.py`, the `EPHEMERAL_ACCEPTED` cancelled-reply handler) is not reached by a plain close either: the close cancels the task while the chat still exists. What a close in this window did reach was `_submit_draft_body`'s accepted-cancellation handler. During setup no run cancel event is registered yet, so `_accepted_cancellation_was_requested` read the close as a failure, called `mark_message_failed` on the row `begin_session_close` had already stopped, and `ValueError('Cannot mark a stopped message terminal.')` escaped `submit_draft` (diagnostics `error_category=internal`). **Fix:** `_accepted_cancellation_was_requested` also counts `session_id in self._session_close_generations` (a close in progress; `read_hook_review_shutdown` already reads it the same way), so the turn ends with TASK-39138's closed-session result: "Session closed.", run state STOPPED, no attention log. `test_a_temporary_chat_closed_during_agent_setup_ends_the_turn_closed[within_grace|outlasts_grace]` (chain creation held; `ConsoleRuntime.close_session` with the setup finishing inside the grace, or outlasting a 0.2 s grace) was red on base ae9f9e8d21 for both cases with that `ValueError`, green after.

**AC #3.** `[replaced_session]` (a successor session object under the same id while the commit runs) still ends BLOCKED / `owner_changed`, before and after. The temporary-chat site's owner changes stay pinned by `test_temporary_chain_stop_refuses_successor_publication[store|persistence|database|session]` (`Tests/Chat/test_console_accepted_setup_native.py`), which passed in the covering run.

**AC #4.** `Tests/Chat/test_console_durable_commit_window.py` is a whole-file target of `pr-fast-lane-admission-2` ("Run admission-sensitive suites 2"), not the "Run fast PR contract" step. Three timings (`-n 4`, load 38-44): close 8.98 / 9.99 / 8.81 s, replaced_session 8.63 / 9.85 / 8.59 s, within_grace 2.90 / 2.79 / 2.84 s, outlasts_grace 2.90 / 2.82 / 2.79 s.

**Covering run.** 12 files (accepted setup, close fence, run state per session, durable turn acceptance and fix round 1, run markers, dispatch recovery fix rounds 1 and 4, runtime shutdown, close-session fleet, first-send atomicity, chat controller), `-n 4`, load ~43: 569 passed, 23 failed; the same 23 fail with this change reverted (`RecoveryRequired: raw_source_selection_changed` and other environment failures), so none is this change's.

**Size.** `Tests/Architecture/test_module_size_ratchet.py`: controller row 29183 -> 29184 (+1, exact; owner waiver).
<!-- SECTION:NOTES:END -->
