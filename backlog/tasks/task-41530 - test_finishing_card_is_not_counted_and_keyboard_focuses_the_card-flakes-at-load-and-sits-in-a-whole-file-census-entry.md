---
id: TASK-41530
title: "test_finishing_card_is_not_counted_and_keyboard_focuses_the_card flakes at load and sits in a whole-file census entry"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-18 08:36'
created_date: 2026-11-17 05:39
labels:
- follow-up
- tests
- flaky
- console
priority: medium
references:
- Tests/UI/test_console_mcp_approval.py
dependencies:
- TASK-41138
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by testinfra (round 15). `test_finishing_card_is_not_counted_and_keyboard_focuses_the_card` (`Tests/UI/test_console_mcp_approval.py:2167`) fails at `-n 4`, load about 90, because `screen._console_attach_reconciled` is still False when the test reads it (asserts at about `:1640`/`:1655`). The file is a whole-file entry in `scripts/ui_pr_gate_census.txt`, so the flake can redden the required check. TASK-41138 only listed it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test waits on `_console_attach_reconciled` (a condition wait with a bounded deadline) and passes 10 of 10 at `-n 4`, load recorded
- [x] #2 The same pattern is grepped across the file and each other fixed read-after-event is listed or fixed
<!-- AC:END -->

## Implementation Plan

1. Trace what sets `_console_attach_reconciled` and why the readiness helper's wait can end with it False under load (slow work against a shared deadline, or a stall).
2. Make the helper wait on it (and the other readiness flags) with a bounded deadline of its own and a failure message that names the stuck state.
3. Grep the file for fixed read-after-event pauses (a `set_task_resume_state`, press or focus call followed by one `pilot.pause()` and a read) and fix or list each.
4. Red before under a forced delay in the reconciliation (reverted); green after; 10 of 10 at `-n 4` with load recorded.

## Implementation Notes

Test-only (`Tests/UI/test_console_mcp_approval.py`). The production-Console readiness wait now has its own bounded deadline and names what is still pending. The finishing test waits for the card's decisions before it reads them.

**Cause: timing, not a stall.**
- `_wait_for_production_console_ready` set one 10 s deadline before its 10 s projection-drain wait. It then spent what was left polling `_console_attach_reconciled` and the sync flags, and asserted a bare `assert screen._console_attach_reconciled`.
- The attach chain can outlast that remainder. The chain is the first `_sync_native_console_chat_ui`, its coalesced re-run, the `console-attach-reconciliation` worker, and the runtime's `finish_view_reconciliation`.
- Control (temporary, reverted): `await asyncio.sleep(12.0)` at the top of `_reconcile_console_after_attach`. The old helper failed at `:1655`, `assert False` where `False = ..._console_attach_reconciled`, the reported shape. The new helper passed: the flag turned True once the sleep ended, so this path re-arms and does not stall.

**Changes.**
- **Readiness helper.** After the projection drains, it waits up to 30 s of its own. It asserts `not unsettled()`, naming each pending item: attach not reconciled, sync in progress or requested, control-bar replay or sync scheduled, a running `console-sync` worker. The message also gives the seconds waited and the reconcile worker's running flag, retry count and exhaustion. The helper is also imported by `test_console_approval_compact_layout.py` (4 callers: lines 168, 301, 355, 486) and `test_approval_batch_geometry.py`.
- **Finishing test, Select check.** After `set_task_resume_state`, the test waits (bounded, `_await_condition`) for `card._information_ready`, then asserts at least one Select and that all are disabled.
  - The old check read `card.query(Select)` one pause after the event. The Select mounts after its row, and `_defer_decisions` holds every control disabled until the round's facts lay out. A probe saw 0 Selects right after the row mounted, so `all(...)` could pass on nothing, or on the deferral's own disabling.
  - Negative control (temporary, reverted): the finishing card no longer disables its Select. The new test failed 4 of 4 at `:2223`. The old test failed 4 of 4 in that run, but passed 1 of 3 in an earlier one: it read before the Select existed.
- **Finishing test, focus check.** `focus_first_decision` is followed by a bounded wait for `app.focused is card` instead of one pause.

**AC#2: the same shape elsewhere in the file** (an event, one `pilot.pause()`, then a read). Listed, not changed: none was reported flaky, and the census entry is whole-file, so a follow-up should take them together.
- **Production Console, two tests.** `test_alt_a_focuses_the_pending_approval_decision_select` (`set_task_resume_state`, pause, `alt+a` at :2313, pause, focus read) and `test_alt_a_reaches_the_card_at_80_columns_with_inspector_closed` (`alt+a` at :2366) press Alt+A one pause after the batch is set. Before `_information_ready`, that is the same race as the finishing card's. `test_alt_a_notifies_when_nothing_is_pending` reads a notification one pause after `alt+a` (:2336).
- **Widget harness (one card, no Console).** 65 sites in 36 tests: lines 111, 283, 348, 384, 426, 480, 503, 555, 570, 574, 615, 656, 747, 760, 798, 808, 815, 852, 871, 900, 910, 939, 948, 955, 960, 980, 987, 992, 1011, 1033, 1044, 1048, 1065, 1097, 1110, 1119, 1128, 1156, 1180, 1211, 1215, 1227, 1252, 1290, 1306, 1336, 1372, 1387, 1400, 1420, 1435, 1459, 1481, 1492, 1517, 1527, 1544, 4035, 4060, 4068, 4095, 4111, 4117, 4122, 4129. They are mostly `card.set_batch(...)` or a button press, then a row query or an assertion.

**Runs.**
- The node: 10 of 10 alone at `-n 4` (load 10.5 before, 11.6 after). Also 30 of 30 with another node (load 13.9-15.8), and 30 of 30 across three 10x batches with seven other nodes (load 10.7-15.0).
- One more such batch, the first (load 15.7-11.2), failed this node 2 of 10; see fix round 1 for the cause. (The `ChaChaNotesDB (CharactersRAGDB) instance not found/assigned in app.__init__` line in their logs is routine: it appears in passing runs too.)
- Whole files at `-n 4`: `test_console_mcp_approval.py`, `test_console_approval_compact_layout.py`, `test_approval_batch_geometry.py` and the Personal Context and ingest files: 417 passed, 1 skipped (load 28-30). Preflight passed.

**Fix round 1 (review r16wt-rev1): focus theft after readiness.**
- **Cause.** The review's loaded batches failed this node 4 of 70 times with "focus_first_decision left focus on ConsoleComposerBar(id='console-native-composer', ...)".
  - `ChatScreen` schedules `_restore_console_workbench_focus` at mount through `call_after_refresh` and through `set_timer(0.2, ...)`, and again through `call_after_refresh` on the attach's resume.
  - The restore keeps focus that is inside a Console pane, and otherwise moves focus to the composer. The approval card is in no pane (`_console_workbench_focus_id_for_widget(card)` is None).
  - A probe (temporary) saw the three restores run 0.15-0.4 s before the helper returned, at load ~12.
  - When the helper returned before a restore had run, the restore took focus from the card to the composer for good, as the reviewer saw.
- **Fix: the readiness helper's `unsettled()` gains two items.**
  - "focus restore timer pending": a screen timer whose target is `_restore_console_workbench_focus`. `set_timer` wraps its callback in `partial(call_next, callback)`; `_timer_target` reads the partial's argument.
  - "initial focus not on a Console pane", which names `app.focused`.
- **Settle loop.** After the flags settle, the helper drains `call_after_refresh`, which covers the resume's restore, since that one has no timer. It repeats until a drain leaves nothing pending.
  - The first version asserted right after a second drain. It failed once in 60 node runs with "Console still unsettled 0.7s into its wait: ['control-bar sync scheduled']": a focus move during the drain scheduled a control-bar sync.
- **Red/green.** Temporary edits: the restore timer moved to 1.5 s, and a 2 s hold with `assert app.focused is card` added at the end of the test.
  - The old helper lost focus to the composer 3 of 3 times ("stolen: ConsoleComposerBar(...)").
  - The new helper kept it 3 of 3, and 4 of 4 after the loop change.
  - A first version of the timer check compared `timer._callback.__func__`, never matched, and failed this control too; that is how the partial was found.
- **Batches with the final helper.** Eight 8-node x 10 batches in two concurrent streams passed 640 of 640, including this node 80 of 80. Alongside them, four 20-rep rounds of this node alone passed 80 of 80. 1-minute load was 10.4-23.4.
- **Old helper, same load.** Under that load the committed (old) helper passed 60 of 60 in three side-by-side rounds (load 17.6-21.3). The theft did not reproduce there; the forced control above is the red evidence.
- **Alt+A tests.** They use the same helper, so they get the fix. `test_console_mcp_approval.py`, `test_console_approval_compact_layout.py`, `test_approval_batch_geometry.py` and the Personal Context files passed 261 of 261 (load 13.7-19.3).

**Proposed follow-up.** Convert the Alt+A and widget-harness sites above to condition waits (`_information_ready` for a fresh batch).
