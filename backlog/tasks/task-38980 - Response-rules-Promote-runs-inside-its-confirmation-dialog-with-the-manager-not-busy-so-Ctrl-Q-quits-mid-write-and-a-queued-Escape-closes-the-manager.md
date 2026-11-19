---
id: TASK-38980
title: "Response-rules Promote runs inside its confirmation dialog with the manager not busy, so Ctrl+Q quits mid-write and a queued Escape closes the manager"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 01:42
updated_date: '2026-11-07 02:04'
labels:
- follow-up
- bug
- console
- response-rules
- quit
priority: low
references:
- tldw_chatbook/Widgets/Console/response_rules_modal.py
- tldw_chatbook/Widgets/confirmation_dialog.py
- Tests/UI/test_response_rules_modal.py
- Tests/UI/test_modal_quit_in_flight_hooks.py
- Tests/Architecture/test_guarded_modal_quit_census.py
dependencies:
- TASK-38747
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b on 2026-11-05; the Ctrl+Q part was read from the TASK-38747 branch, which merges first.

**How Promote runs.**
- Promote pushes a ConfirmationDialog with `confirm_callback=commit` (`tldw_chatbook/Widgets/Console/response_rules_modal.py:556-587`).
- `ConfirmationDialog.on_button_pressed` awaits that callback before `dismiss(True)` (`tldw_chatbook/Widgets/confirmation_dialog.py:153-155`).
- So the promote write runs inside the dialog's button handler. `_busy = True` is set only for the manager's other actions (`:589`).

**Probe on dev, with the promote write held open.**
- `_busy` was False. The screen stack was `['Screen','ResponseRulesModal','ConfirmationDialog']`.
- An Escape pressed during the write was still unprocessed after 3 s, because the dialog's handler was blocked.
- Once the write was released, the stack went straight to `['Screen']`: the queued Escape closed the manager as soon as the promote landed. The binding was written (revision 1).

**Ctrl+Q after TASK-38747 merges (from reading its branch).**
- TASK-38747 adds `ResponseRulesModal.confirm_quit` (`response_rules_modal.py:712-738` on that branch). It refuses only when `_busy` is set; otherwise it asks only about unsaved edits.
- ConfirmationDialog has no `confirm_quit`.
- So Ctrl+Q during a promote write quits without asking.
- TASK-38747 covers Test, Save and the pin changes (Disable/Enable, Exclude, Delete local pin), but not Promote.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 While a confirmed Promote write is in flight, Ctrl+Q answers "still working" as it does for Test, Save and pin changes; the test is red against the current Promote flow
- [x] #2 While the write is in flight the manager shows that the promote is running, its action buttons are disabled, and Escape does not close the manager before the write settles
- [x] #3 After the write, the manager shows the promote's outcome (promoted, refused or failed), including when Escape was pressed during the write
- [x] #4 The guarded-modal quit census and Tests/UI/test_modal_quit_in_flight_hooks.py stay green
<!-- AC:END -->

## Implementation Plan

1. Red first, with the promote write held in `runtime.run_db_call` (no Pilot while it is held: on dev the write runs inside the dialog's button handler, which blocks Pilot):
   - `Tests/UI/test_app_quit_in_flight_modals.py`: real `TldwCli`, real Ctrl+Q (driver key path) during a confirmed Promote; it must answer "still working" and not quit. Released in a `finally`.
   - `Tests/UI/test_response_rules_modal.py`: during the held write the manager shows the promote is running, its action buttons are disabled and Escape leaves it open; after the write it shows the outcome (promoted, refused, failed), with Escape pressed during the write.
2. Fix in `ResponseRulesModal._action`: the confirmation dialog only asks. Its confirm callback claims the manager (`_busy`, buttons, "Promoting…" status) before the dialog closes, so no key lands between; the `_action` worker waits for the answer (`push_screen_wait`, already inside a worker) and runs the write in the manager's existing busy section. Escape during a promote keeps the manager open with a promote-specific status instead of "Testing stopped". `confirm_quit` already refuses while `_busy`, through `refuse_quit_while_working` (the TASK-38747 choke point); its docstring names Promote.
3. Keep `Tests/Architecture/test_guarded_modal_quit_census.py`, `test_quit_flow_prompt_choke_point.py` and `Tests/UI/test_modal_quit_in_flight_hooks.py` green, plus the existing response-rule modal suites; update the Response rules doc copy for the new running state.

## Implementation Notes

**Approach.** Promote pushed a `ConfirmationDialog` whose `confirm_callback` was the write, and `ConfirmationDialog.on_button_pressed` awaits that callback before `dismiss(True)`. So the write ran inside the dialog's button handler with the manager not busy: the quit walk asked the dialog (no `confirm_quit`) and then the manager (`_busy` False) and quit mid-write, and an Escape queued behind the blocked handler and closed the manager as soon as the write landed. Now the dialog only asks. Its confirm callback claims the manager before the dialog closes (`_busy`, `_promoting`, buttons disabled, "Promoting revision N to <scope>…"), so no key lands between the answer and the write. `_action`, which always runs in the manager's worker, waits for the answer (`push_screen_wait`) and runs the write in the manager's existing busy section, which reports the outcome and clears the busy state in its `finally`. `confirm_quit` was already the choke point (`refuse_quit_while_working` while `_busy`, TASK-38747); only its docstring changed. Escape during a promote keeps the manager open and says "The revision is still being promoted. The manager stays open until it finishes." instead of "Testing stopped", and does not call `runtime.cancel`.

**Tests.**
- `Tests/UI/test_app_quit_in_flight_modals.py::test_ctrl_q_waits_for_a_confirmed_rule_promote_being_written`: real `TldwCli`, the promote write held in `runtime.run_db_call`, Ctrl+Q through the app's key path (`app._press_keys`, what `pilot.press` calls first) with plain-sleep polls while the write is held, per the TASK-38747 lesson (on the base the write ran in the dialog's handler, which Pilot waits on). It answers still working and stays; after release the manager reports the promote and Ctrl+Q quits through `pilot.press`.
- `Tests/UI/test_response_rules_modal.py::test_a_confirmed_promote_keeps_the_manager_busy_and_open_until_it_reports[promoted|refused|failed]`: while the write is held the manager is on top, says "Promoting revision 1", every action button is disabled, and Escape is answered and leaves it open; after the write it shows the outcome, with Escape pressed during the write.

**Red/green.** Against the base `response_rules_modal.py` (temporarily restored): the real-key test failed 2 of 2 with "Ctrl+Q quit straight past ResponseRulesModal while its promote was still being written" (`[True] == []`), and the manager test failed 3 of 3 with "timed out waiting for the manager to show the write" (the dialog stayed on top during the write). With the fix: 3 of 3 and 3 of 3 passed (the real-key test also 3 of 3 serial).

**AC#4.** `Tests/Architecture/test_guarded_modal_quit_census.py` and `test_quit_flow_prompt_choke_point.py`: 11 passed (serial). `Tests/UI/test_modal_quit_in_flight_hooks.py`, `test_response_rules_modal.py`, `test_console_response_rules.py`, `test_response_rules_end_to_end.py`, `test_console_modal_dismissal.py`, `test_app_quit_in_flight_modals.py` and `Tests/Chat/test_response_rules_write_contention.py` at `-n 4`: 269 passed, 1 failed. The failure, `test_console_response_rules.py::test_active_run_copy_takes_priority_over_an_earlier_rule_verdict` ('Waiting for reply…' == 'Streaming response.'), fails the same way on the read-only base worktree (fu-base-r11), is not in any lane, and is unrelated to this change.

**Lane.** The Console P0 regression gate job, whose admission-sensitive step runs this file's other nodes, took 16.5-19.6 min of its 20 over the same recent PR runs, so the new manager node joins the UI census tail instead (shard 23 of 32 on this base, slowest recent run 13.6 min; the node is about 3 s a case). The real-key test is in the file TASK-38978 lanes.

**Review r11m-rev1.** `test_promotion_previews_exact_pin_without_private_examples` and the `promote` case of `test_saved_edit_stays_selected_and_pinned_for_next_manager_action` counted fixed `pilot.pause()` calls after Confirm. The write now runs in the manager's worker, and the first one failed once at load 32 to 54. Both now wait, bounded (10 s), until the manager is back on top, not busy, and the binding exists. The second of them is in the Console P0 gate's argv.

**Docs.** `Docs/User_Guide/index.md` (Ctrl+Q's still-working list now names promoting a response rule) and `backlog/docs/console-response-rules.md` (the Promote running state).

**Verification after review r11m-rev1** (2026-11-07 UTC; the machine was shared, and the load is given for each run).
- **Head.** 19 modal and fleet files under the scratch boot plugin, `-n 4`, load 20 to 80: 272 passed, 2 failed. The two failures are the inspector-section summary tests, which also fail on base.
- **Base.** The same files: 234 passed and 17 failed, and then the run hung with idle workers. The watchdog killed it after 30 minutes, with about 20 tests unfinished. An earlier base attempt hung for more than 2 hours. Base still has the hangs these tasks fix.
- `Tests/CI -n 4`: 1215 passed, 1 skipped.
- Preflight passes.
- Architecture and size ratchets: 216 passed.
- Boot budget: 32 passed.
- **Stress.** The new and changed nodes are 22 selectors, 45 tests, run at `-n 6` under the plugin, 3 runs per round:
  - Round 1, load 65 to 104: 1 failed, 0 failed, 1 failed.
  - Round 2, load 53 to 77: 2 failed, 1 failed, 0 failed.
  - Each test-wait failure was fixed (bounded waits, a held-save precondition) or traced to the product re-sync above.
  - Round 3, on the final head, load 42 to 61: 1 failed, 0 failed, 0 failed. The one failure is the freshness node's known "Mix-in attempt could not be saved" race, an automatic turn's mix-in anchor, which is reported as a follow-up.

**Verification after review r11m-rev2** (2026-11-07 UTC, load about 25-50).
- **Head.** 19 modal and fleet files under the scratch boot plugin, `-n 4`: 272 passed, 2 failed. The two failures are the inspector-section summary tests, which also fail on base.
- **Receipt and read-state coverage.** `test_console_chat_controller.py`, `test_console_activity_receipts.py`, `test_console_receipt_bootstrap.py`, `test_buddy_hidden_results.py`, `test_console_refresh_read_batching.py` and `test_console_historical_read_cancellation_lifetime.py`: 36 failed and 376 passed on both head and base, with identical failure sets.
- `Tests/CI -n 4`: 1215 passed, 1 skipped.
- Preflight passes.
- Architecture and size ratchets: 216 passed.
- Boot budget: 32 passed.
- **Stress.** The new and changed nodes are 22 selectors, 45 tests, run at `-n 6`:
  - First round: 1 failed, 0 failed, 1 failed. Both failures were the view-all test at 120x35, now fixed: it waits for the toggle, tail and steps to be painted.
  - Final round: 45 passed in each of the 3 runs.
- **Tooling, all in `$FU`.** `collect_ui_shard_minutes.py` writes `ui_shard_ci_minutes.json`. `census_shards.py` shows the shard dealt to each census entry and flags heavy entries on hot shards. `census_reorder.py` permutes one block. The weights are in `modals_census_weights.json`.

**Files.** `tldw_chatbook/Widgets/Console/response_rules_modal.py`, `Tests/UI/test_response_rules_modal.py`, `Tests/UI/test_app_quit_in_flight_modals.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Docs/User_Guide/index.md`, `backlog/docs/console-response-rules.md`.
