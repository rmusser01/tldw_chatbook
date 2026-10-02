---
id: TASK-33661
title: Resend a broken Console turn in place from the user message
status: Done
assignee:
  - '@claude'
created_date: '2026-10-01 17:30'
updated_date: '2026-10-02 06:17'
labels:
  - console
  - transcript
  - owner-request
dependencies: []
references:
  - tldw_chatbook/Chat/console_message_actions.py
  - tldw_chatbook/Widgets/Console/console_transcript.py
  - tldw_chatbook/UI/Console_Modules/message.py
  - tldw_chatbook/Chat/console_chat_controller.py
  - tldw_chatbook/Chat/console_chat_store.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: when a Console send fails or gets stuck, the user sees an error and has no direct way to try again. Today the user has to choose Edit, then Edit & resend. That forks a new sibling branch, and for some failure shapes it is the only path left. The owner asked (2026-10-01) for a one-click **Resend** on the user's own message that resumes the broken chat.

Owner decisions:
- Resend appears only when that turn failed or got no reply.
- It re-runs the same turn in place: no fork, no sibling branch, no second copy of the user message.

Lead defaults (the owner can override): Resend applies to the last turn of the active path only, since a failed reply higher up keeps its own Retry. An empty reply that the user stopped counts as "no reply", which makes stuck → Stop → Resend work. A partial reply does not count, because Continue covers it.

How the code models a broken turn today (research against dev 84247cb843, 2026-10-01):
- Provider or HTTP error: the user message and the assistant row are both persisted. The assistant is marked failed, and a transcript-only SYSTEM "Provider stream failed" row follows. The assistant row offers Retry, but nothing on the user row does.
- Empty stream: the assistant is marked failed with no SYSTEM row.
- Stop on a hung or stuck run: the assistant is marked stopped and a SYSTEM "Response stopped by user." row follows. It offers ♻ (which forks) and Continue, but no Retry.
- Readiness or other refusal before the send is accepted: the user echo is unpersisted and marked failed, a SYSTEM block row follows, and there is no assistant row. The draft and staged attachments stay in the composer. The user row shows Continue, which today parents the new reply under the SYSTEM row and leaves the echo failed (an untested hazard).
- After a restart: hydration rebuilds nodes as complete, keeping only assistant_generation_state. A failed reply then reads "Response failed." and loses Retry.
- Dispatch recovery after a restart (a reply left in flight) has its own card with Retry response and Discard. Discard settles the reply as discarded.

Existing machinery that resend can reuse without forking:
- The assistant in-place retry (retry_message / prepare_message_retry) for a failed reply.
- continue_from_message, which re-runs from a persisted user message and appends a reply at the active leaf.
- delete_message (subtree tombstone).
- The normal send path, for re-dispatching a refused echo.

Related open tasks, not absorbed here: TASK-370 (resume or retry for interrupted replies), TASK-33620.3 (a failed durable commit leaves a user message with no assistant row, stuck on Running), TASK-33621.22 (Stop during "Connecting tools…" wedges the next send).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user message that is the last turn on the active path shows a Resend action only when its turn is broken. Broken means: its send was refused before acceptance; or it has no assistant reply; or its reply failed; or its reply is empty and stopped, discarded, or restored as "Response failed." after a restart. Resend never shows on a healthy turn, on a partial (non-empty) stopped reply, on a turn above the last one, or on an assistant row.
- [x] #2 Resend re-runs that turn in place, with no new branch, no sibling and no second copy of the user message. The broken or empty reply and its trailing failure or stop SYSTEM rows are cleared, and the new reply appears directly under the same user message. A refused echo is re-sent with the same text and attachments as exactly one user message, and the composer is not left holding a duplicate of that text.
- [x] #3 Resend is not offered while a run is live in that tab (Stop comes first) or while a dispatch-recovery card is unresolved (the card owns that case). Every normal send gate applies, including readiness, vision and skill refusal, and a refused Resend shows the same refusal copy as a normal send.
- [x] #4 Resend works after an app restart for a persisted broken last turn: a restored failed or discarded reply, or a last user message with no reply.
- [x] #5 Keyboard: with a broken user row selected, `r` runs Resend. With a failed assistant row selected, `r` runs Retry; today it does nothing there. The action's tooltip and the help or footer text describe Resend, and no hint names a key that does not work (ADR-031).
- [x] #6 The selected action row still fits the reference terminals at 211x44 and 235x52. Tests that pin the user-row and assistant-row action sets are updated on purpose and named in the notes.
- [x] #7 Tests cover each broken shape with a real controller and store: provider error, empty stream, refused echo, empty stopped reply, restored after restart, and discarded. Each proves the re-run is in place by message count, ids and parentage, with no sibling. There are negative tests for a healthy turn, a mid-path turn, a live run and an unresolved dispatch recovery. A pilot test clicks Resend in the real Console.
- [x] #8 The Console User Guide documents Resend: when it appears and what it does.
- [x] #9 No size ratchet rises (ADR-097). The resend logic lives in its own module. console_chat_controller.py, console_transcript.py and console_chat_store.py each net 0 or fewer lines, and preflight passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. New module Chat/console_turn_resend.py: pure broken-last-turn detection, controller-side in-place re-run (retry a failed reply; else clear the empty reply + stale rows and continue from the user with resend semantics), refused-echo re-dispatch through the normal send path.
2. Action model: resend_available kwarg swaps the disabled regenerate for Resend (drops Continue) on the broken last user row; dispatch + guide + tooltip; r presses Retry/Resend when that is the row's action.
3. Routing in UI/Console_Modules/message.py gated like Retry (send_refusal_copy, console-run worker); wiring for the echo path.
4. continue_from_message gains resend=True (thinking preflight + pinned prefill); offset by trimming proven-dead code so controller/transcript/store net <= 0.
5. Tests: real controller+store (SQLite) per broken shape incl. restart and discard, negatives, pilot click; update pinned tests; User Guide; live tmux verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resend re-runs a broken LAST user turn in place from the user message. It never forks, never creates a sibling and never calls edit_and_resend_message. The logic lives in the new module `tldw_chatbook/Chat/console_turn_resend.py`:
- `resend_target_id(messages)`: pure broken-turn detection over the active path.
- `resend_turn(controller, user_id, resend_echo=...)`: the gates, the clearing and the in-place re-run.
- `resend_refused_echo(...)`: sends a refused, never-persisted echo through the normal send path.

**Approach**
- Broken means:
  - a refused echo (unpersisted failed USER row, no reply after it);
  - a persisted user message with no reply;
  - a failed reply (live status failed, or restored generation state failed);
  - an empty reply that was stopped, discarded or restored. "Empty" ignores the "Response discarded." copy, and every assistant row of the turn must be empty.
- In the transcript, `_action_groups` passes `resend_available` into `available_actions`. Its value is `resend_target_id(self._messages) == message.id and not self._selection_run_active()`, following the `fork_eligibility` precedent. On that user row the disabled ♻ becomes `("resend", "Resend")` and Continue is dropped, mirroring the failed-assistant Retry swap.
- Routing in `UI/Console_Modules/message.py` shares the Retry branch: `send_refusal_copy` first, then `run_worker(..., exclusive=True, group=f"console-run-{sid}")`.
- What a run does, by shape:
  - A live failed reply: tombstone the rows after it, then `retry_message` on the same assistant row.
  - Every other persisted shape: tombstone the subtree of the first row after the user message. That removes the empty reply and the stale failure or stop rows, persisted or not. Then `continue_from_message(user_id, resend=True)`.
  - A refused echo: the echo is deleted, then its text and attachments go through `chat_screen._dispatch_console_draft_send`, wired as `_resend_refused_console_echo` in `wiring.py`.
- `continue_from_message` gained `resend: bool = False`. With it, the run applies the thinking-persistence preflight and the pinned prefill. Plain Continue is unchanged.
- The `r` key: the binding stays `invoke_selected_action('regenerate')`. When the row has no ♻ button, `_press_selected_action_button` presses the row's Retry or Resend instead. The binding's help text reads "Regenerate/Retry/Resend". The guide names "r Retry" and "r Resend". The Inspector's message-actions row reads "Regenerate/Retry/Resend". No Ctrl binding was added (ADR-031).

**Rulings**
- **No-reply needs a persisted user row.** An unpersisted user row with no reply is an in-flight send (validating, or paused for library preparation), so it is never broken. Cost if wrong: in a temporary chat, a user row with no reply and no live run gets no Resend; Edit & resend still works.
- **Failed reply: retry in place.** A live failed reply is retried on the same row. Restored, stopped and discarded replies are cleared, and the turn re-runs from the user message. Reason: `retry_message` requires status "failed", and a restored row reads "complete". Cost: the cleared empty reply is a soft-deleted tombstone in the database.
- **Thinking preflight and pinned prefill: `resend` flag on `continue_from_message`.** A normal send applies both. Continue skips them, and `test_continue_never_gets_prefill` pins that Continue never gets a prefill, so the flag gates them. Cost: plain Continue still has no thinking preflight (pre-existing).
- **Prefill: pinned only, never the one-shot.** This matches `retry_message` (`test_retry_applies_pinned_but_not_one_shot`). The one-shot belongs to the user's next send.
- **Agent or tool-calling mode: no change needed.** `_stream_assistant_response` picks agent mode for every route: agent runtime enabled, a bridge present, no prefill and not a character session. A resend therefore runs the session's mode exactly as a send does.
- **Attachments and vision gate.**
  - Persisted turns: the turn's own attachments are re-checked with the same `vision_block_reason` call that the send and edit-resend paths use, before anything is cleared. A refusal goes through `controller._block`, the same copy and SYSTEM row a send shows.
  - Refused echoes: the normal send path's own gate applies.
  - Readiness and skill refusal reuse the `_block` calls that `retry_message` and `continue_from_message` already make.
- **Hide gate.** A live run hides Resend through `_selection_run_active`; VALIDATING counts as active. An unresolved dispatch recovery hides it by shape: the owner row is accepted or dispatch_started, never failed, stopped or discarded. `resend_turn` also re-checks `send_refusal_copy` before it clears anything. No chat_screen seam was added. Cost: if a future recovery shape carries a failed owner row, Resend would show but be refused with the recovery copy.
- **Refused echo: the unsent-turn recovery is the authoritative copy.**
  - The matching recovery holds the exact draft and attachment objects, and it is consumed, so the shelf never offers a duplicate.
  - A composer holding the same text is passed to the send path as its stash, which commits (clears) it.
  - A composer holding different text gets a refusal ("Send or clear the composer draft before resending this message.") and is never overwritten.
  - If the send path refuses before custody, the draft goes back into the composer, as for a refused normal send.
  - The echo is deleted before dispatch. Otherwise the new echo would attach under the old one at the active leaf.
  - Cost: attachments staged for a new draft at that moment are sent along, as they would be with Send.
- **Sync timer.** For a refused echo, the worker does not start the transcript sync timer itself. The live check caught this: starting it first let a tick stop it while the session still read blocked, and the resent turn then ran unpolled. Recorded in `lessons-console-wiring.md`.
- **Line budgets.** Proven-dead code was trimmed to keep the controller and transcript at net ≤ 0:
  - `ConsoleChatController._revoke_question_rounds`, unused since 5dfd9a4d49 (TASK-31384).
  - `console_transcript._message_role_label`.
  - `ConsoleTranscript.select_previous_variant`.
  - Neither transcript helper has any reference in tldw_chatbook/ or Tests/.
- **UI PR-gate census not extended.** `Tests/UI/test_console_turn_resend_ui.py` was not added to `scripts/ui_pr_gate_census.txt`, because it could not be verified green under the fast-lane dependency set locally.

**Pinned tests changed on purpose**
- `Tests/Chat/test_console_message_actions.py::test_failed_user_row_offers_no_retry_action` was extended: the broken last row offers resend and still never retry.
- New beside the existing pins:
  - `test_broken_last_user_row_swaps_regenerate_for_resend_and_drops_continue`
  - `test_resend_is_never_offered_on_an_assistant_row`
  - `test_resend_dispatch_targets_the_user_message`
  - `test_action_row_guide_names_r_for_resend_and_retry_swaps`
- `test_user_and_stopped_assistant_action_groups_are_exact` and `test_continue_action_remains_available_for_failed_user_message` are unchanged. The kwarg defaults to False, and Continue is still dispatchable.
- `Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_updates_inspector_action_guidance` was updated for the "Regenerate/Retry/Resend" Inspector copy.
- The width-budget, stable-id and user-regenerate-disabled tests in `test_console_native_transcript.py` are unchanged: they use healthy or unpersisted rows. The broken-row width is pinned in the new UI test (≤ 48 cells).

**Tests**
- `Tests/Chat/test_console_turn_resend.py`: real controller and SQLite store for every broken shape, each asserting the user id, the parentage, sibling_count 1 and the live database rows. Shapes covered:
  - provider error and empty stream (retried in place, same assistant id);
  - empty stopped reply;
  - restored failed reply after a relaunch through the production hydration;
  - a user message with no reply after a relaunch;
  - discarded reply, both live and relaunched;
  - refused echo, with and without a recovery, composer copy and attachment;
  - the gates: readiness, vision, skill refusal, the thinking preflight, the pinned prefill and a refused clear;
  - negatives: a healthy turn, a mid-path turn, a live run and an unresolved dispatch recovery.
- `Tests/UI/test_console_turn_resend_ui.py`:
  - transcript action row, tooltip and guide at 211x44;
  - hidden while a run is live;
  - `r` presses Resend on a broken user row and Retry on a failed assistant row;
  - the sync-timer regression test;
  - pilot tests in the real ChatScreen: a click on Resend re-runs a failed turn in place, and `r` re-sends a refused echo as one message with the composer left empty.

**Verification**
- Live tmux runs at 211x44 and 235x52 on a scratch profile (HOME, XDG and TLDW_CONFIG_PATH all in scratch). Captures and the procedure are in `qa/task-33661-resend/`.
- Branch feat/task-33661-console-resend, 2026-10-01.

**Files**
- New: `tldw_chatbook/Chat/console_turn_resend.py`, `Tests/Chat/test_console_turn_resend.py`, `Tests/UI/test_console_turn_resend_ui.py`, `qa/task-33661-resend/`.
- Modified:
  - `Chat/console_message_actions.py`
  - `Chat/console_chat_controller.py`
  - `Widgets/Console/console_transcript.py`
  - `UI/Console_Modules/message.py`
  - `UI/Console_Modules/wiring.py`
  - `UI/Screens/chat_screen.py` (copy only, 0 lines)
  - `Docs/User_Guide/console/chat-basics.md`
  - `Docs/User_Guide/console/branching-and-rewind.md`
  - `backlog/docs/lessons-console-wiring.md`
  - the two pinned test files above.

**Evidence (2026-10-01)**
- New tests, run with the plain command (no extra -p plugins): `Tests/Chat/test_console_turn_resend.py` and `Tests/UI/test_console_turn_resend_ui.py` together give 45 passed. `Tests/Chat/test_console_message_actions.py`: 120 passed. Its 10 failures (canvas and tool-marker cases) fail identically on base 142b4b8407.
- Covering suite, 24 files, compared against base 142b4b8407 with a scratch bootstrap_profile plugin (comparison only): HEAD 125 failed / 1603 passed; base 125 failed / 1599 passed. The failing test-node sets are identical; the 4 extra passes are the new tests.
- Size ratchets: the 12 failures are identical at base and HEAD (dev is already over these budgets). Line counts, base → HEAD:
  - console_chat_controller.py: 30660 → 30656
  - console_transcript.py: 8457 → 8448
  - console_chat_store.py: 22534 → 22534
  - chat_screen.py: 25311 → 25311
- `./scripts/preflight.sh`: rc 0.

**Follow-ups found, not fixed**
- Pre-existing, outside this diff: after a real relaunch, the dispatch-recovery card cannot settle. Discard (and Retry) refuse with "That response recovery action is unavailable." The recovery state keys the persisted assistant id, but hydration gives each restored node a fresh native id, so `claim_dispatch_recovery_action` gets a KeyError on `_message_or_raise`. Reproduced at the controller level and live (`qa/task-33661-resend/e2`/`e3` captures). The dispatch-recovery suite's own restore keeps native ids equal to persisted ids, which hides it. Until this is fixed, the User Guide's "after Discard, Resend" path is reachable only within one process.
- Plain Continue still has no thinking-persistence preflight (pre-existing; Resend adds it).
- `Tests/UI/test_console_turn_resend_ui.py` could join `scripts/ui_pr_gate_census.txt` once it is verified green under the fast-lane dependency set.

**Review fixes (fix round on 3a59ed8255)**
- **C1, data loss: a turn whose earlier reply holds text is never broken.** `resend_target_id` returns None when any reply before the last has text. That case is a Continue chain whose last reply failed. Before the fix, Resend after a relaunch anchored its clear at the user row and tombstoned the healthy reply's subtree.
- **I1: tool output is a partial reply.** Any TOOL row with content after the user message means no Resend for the no-reply and empty-stopped/discarded shapes. Ruling: failed replies keep Resend even when a tool or error marker is present — that is the owner's scope for I1. A restored failed agent turn always carries the re-derived "Error" warning marker, and it should stay resendable.
- **I2: a refused echo's text survives a cancelled or failing send.** `resend_refused_echo` wraps the send-path await in `try/except BaseException`, puts the draft back, and re-raises.
- **I2: per-session in-flight guard.** The Resend worker is now named `console-resend`. The handler refuses a second Resend while an unfinished `console-resend` worker exists in that session's `console-run-{sid}` group ("Resend is already in progress.").
  - Ruling: guard on the worker's own state rather than a set kept by hand, or hiding the button. Reason: a hand-kept set is never cleared if its worker is cancelled before its first step, which would block Resend for that session for good. Hiding the button would need transcript state. This guard is about 10 lines in message.py, with nothing to clean up.
- **M2: text typed during the send always wins.** The draft is put back only when the composer is empty after the await. Ruling: if the user typed different text during a failing send, the resend text is not put back. There is no ambiguous merge, matching the runtime's own `restore_turn_recovery` refusal.
- **I3:** the Discard→Resend sentence was dropped from the User Guide's restart-recovery section, because after a reopen Discard fails (TASK-33662). The Resend section now names the Continue-chain and tool-output exclusions.
- **M1:** the `resend_turn` docstring now says which gates run before the clear and which run after it (readiness, skill refusal, the thinking preflight, the maintenance pause). It also notes the temporary-chat cost: an empty reply cleared there is simply gone.
- **M3:** `continue_from_message` uses the file's `thinking_block` idiom: `if resend and (thinking_block := ...) is not None: return thinking_block`.
- **M4, Ruling:** Resend runs are traced as route CONTINUE (or RETRY on the in-place path). Left as is: the route tracks the controller entry point; a distinct RESEND route would need a trace-provenance vocabulary change.
- **M5:** `select_next_variant` left alone (it has a test caller).
- **M6, Ruling:** a persisted user leaf with no reply can still have an inactive reply on another branch, for example after deleting the active one of two sibling replies. Resend then adds the new reply beside it. Accepted: nothing is lost, `<`/`>` shows both, and the store exposes no child-count API to the transcript. Cost: AC#2's no-sibling property does not hold in that edge.
- **M7:** `Tests/UI/test_console_turn_resend_ui.py` added to `scripts/ui_pr_gate_census.txt`, and `MINIMUM_FILES` raised from 119 to 120. Verified 10/10 green in a scratch venv with only the fast-lane dependencies (`pip install -e . pytest pytest-asyncio pytest-timeout packaging`, no xdist, `--timeout=180`).
- **A regression in the first commit, found while fixing.** Routing Retry through `run = ...; run_worker(run(...))` hid `_retry_console_message`'s dispatch site from `test_console_run_and_sync_workers_use_disjoint_groups`. The name-set comparison missed it because that test already failed at base for another missing site. Fixed by keeping direct calls. `_resend_console_turn` added to the test's `RUN_COROUTINES`. Lesson added to `lessons-testing-evidence.md`.

**Qodo fixes (PR #2956)**
1. **Docs rule.** `is_refused_echo` and `continue_from_message` now have Google-style Args/Returns sections. The controller stays at origin/dev's exact line count: one blank line in that method was dropped.
   - Test: `test_resend_entry_points_document_args_and_returns` checks all five entry points.
2. **Tool output.** Any TOOL row with content after the user message makes the turn partial, whatever the reply's state (owner ruling, extending I1). The live failed reply keeps its own Retry.
   - Test: `test_resend_never_offers_a_failed_turn_with_tool_output[live|relaunch]`, plus detection cases.
3. **Backup pause.** `resend_turn` now holds the controller's `_maintenance_boundary("turn")` admission for the whole resend. A paused admission therefore refuses with "Console generation is paused for backup maintenance." before anything is cleared. The nested retry and continue calls run under the same depth, and `maintenance_drain` waits for the resend.
   - Test: `test_a_backup_pause_refuses_resend_before_anything_is_cleared[failed|stopped]` checks ids, content and the deleted flag.
4. **Restored failed reply with text.** A failed reply restored with text is partial, so it gets no Resend (Continue covers it). Only a LIVE failed reply (`status == "failed"`) stays resendable with partial text, through the in-place `retry_message`.
   - Tests: `test_resend_never_offers_a_restored_failed_reply_that_has_text`; `test_resend_retries_a_failed_reply_in_place[partial-error]`.
5. **Staged attachments.** `resend_refused_echo` refuses with "Send or remove the staged attachments before resending this message." before deleting anything, when files are staged. The one exception: with no live recovery, the staged files may be exactly the echo's own, as after a shelf Restore.
   - Tests: `test_refused_echo_resend_refuses_newer_staged_attachments[recovery|no recovery]`; the positive case `test_refused_echo_resend_sends_its_own_restaged_attachments_once`.

**AC#1 as implemented after these fixes**
- **Broken:**
  - a refused echo;
  - a persisted last user message with no reply;
  - a reply that failed in this session (live status failed), even with partial text;
  - an empty reply that was stopped, discarded, or restored as "Response failed.".
- **Never broken:**
  - any tool output in the turn;
  - text from an earlier reply of the turn;
  - a restored failed reply with text;
  - a partial stopped reply.

**New Rulings**
- **Backup pause mid-resend.** Holding the admission covers everything in the resend's own task. For a refused echo, the send path hands the turn to a runtime custody task, and that task is admitted separately by `submit_draft`. Remaining race: a backup that pauses admission between the resend's admission and the custody task's own refuses that custody submit. The text is not lost: the runtime records it as an unsent turn on the shelf.
- **Staged-attachment check compares bytes.** It compares the staged files' bytes with the echo's attachment bytes. Cost: a staged inline (text) attachment always refuses the resend, which is conservative.

**UI fast-lane failure (PR #2956, run 36975977069): a product race, not a test defect**
- **Symptom.** Both real-send pilots timed out on the Linux runner with "No messages yet." on screen. The controller logs showed each turn reached its expected terminal state (stream failed or refused). Locally they passed alone, with neighbours, and in a full-census run.
- **Root cause.** The 0.2 s transcript poll stops when `_console_transcript_poll_needed()` sees no live work. That check ignored runtime custody. A turn the runtime has accepted, but whose controller run has not started yet, counted as idle. On a slow runner the first poll tick landed in that gap (RAG capture and provider resolution). The poll stopped, and nothing synced the turn afterwards. Evidence:
  - Under `taskpolicy -b` (macOS background QoS), the pilots took about 15 s, like CI's 16 s, and failed the same way. A 600-attempt wait did not help either.
  - An instrumented run logged the stop with run state IDLE, `in_flight_run_count() == 0`, one custodied turn and zero store rows. The store later held all three rows; the screen never rendered them.
- **Fix.** `ConsoleRuntime.has_custodied_turns()` is now one more live-poll reason in `_console_transcript_poll_needed()`. This also re-arms polling when a view reattaches mid-custody. `chat_screen.py` stays at net 0 lines.
- **Tests.**
  - `test_console_poll_outlives_a_turn_the_controller_has_not_started` holds `submit_draft` across several poll ticks. It fails without the fix at normal speed.
  - `test_reconciled_view_keeps_each_live_poll_reason_and_one_timer[custody]` extends the existing poll-reason pin; its runtime double gains `has_custodied_turns`.
  - The original assertions are unchanged, and the file stays in the census.
- **Ruling.** No TASK-33663 was filed. The defect is a real product race, and fixing it is in this PR's path, so this is not a CI-environment defect.
<!-- SECTION:NOTES:END -->

### Post-merge fix (2026-10-02)

Dev's Perf Guard failed after #2956 merged: the `_ui_ready` census read 1034 against its limit of 1033. The cause was that `console_turn_resend` had become resident at boot through module-level imports in `message.py`, `wiring.py` and `console_transcript.py`. All three are now lazy, function-level imports. The census measures 1033 when run with `PYTHONPATH=<worktree>`, and the 75 Resend tests pass. `test_resend_worker_leaves_a_refused_echo_sync_timer_to_the_send_path` now patches `console_turn_resend.resend_turn`.
