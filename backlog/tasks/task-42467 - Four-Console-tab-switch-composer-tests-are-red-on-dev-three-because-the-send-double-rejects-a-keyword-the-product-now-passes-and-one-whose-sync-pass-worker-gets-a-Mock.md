---
id: TASK-42467
title: "Four Console tab-switch composer tests are red on dev: three because the send double rejects a keyword the product now passes, and one whose sync-pass worker gets a Mock"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 00:57
labels:
- follow-up
- console
- tabs-sessions
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_console_tab_switch_binds_composer.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/composer_run_controls.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23): TASK-42414's covering runs recorded 9 local reds in `Tests/UI/test_console_tab_switch_binds_composer.py` at plain dev 2f7790873a. Re-run on the round-22 stack before its rebase (142e821116, `-n 4`, load about 30): the same 9. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** Five of the nine are TASK-35830's (open): the setup assertion no longer produces the bound-elsewhere state. The other four are not tracked anywhere:
- `test_enter_after_a_tab_click_during_a_sync_pass_sends_to_the_clicked_chat`, `test_a_switch_during_a_sync_pass_binds_the_composer_at_once[browser]` and `test_keys_typed_after_a_second_quick_tab_click_go_to_that_tab` each time out in the send-acknowledgement wait (`Tests/UI/test_console_send_acknowledgement.py:278`, 22-23 s each). In each, the Send raises `TypeError: _record_dispatch.<locals>.dispatch() got an unexpected keyword argument '_captured_inputs'`, and the Console logs `console_send_stage ... error_category=validation exception_type=TypeError phase=ui_action`. The file's `_record_dispatch` double (lines 173-182) replaces `console._dispatch_console_draft_send` with `dispatch(draft, stash=None, *, session_id=None)`. The call that reaches it, and the frame the traceback names, is `UI/Screens/chat_screen.py:18423-18426`: `self._dispatch_console_draft_send(draft, stash=stash, session_id=session_id, **({"_captured_inputs": received_inputs} if received_inputs is not None else {}))`. `git log -S` first shows that keyword argument there in 077e5bfa09 (2026-10-31). This looks like test-double drift.
- `test_an_activation_that_loses_authority_after_switching_asks_for_a_repaint[False]` fails at line 834: the recorded `'pass started'` call to `screen.run_worker` has `(<Mock ...>,)` as its arguments where the test expects `('sync pass',)`. Whether the double or the product's call changed is not established.

Only `test_running_harness_retires_actual_borrowed_database_connections` from this file is in `scripts/ui_pr_gate_census.txt`, so these reds report nowhere, and no CI scan has seen them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the four nodes has a recorded cause: test-double drift after a named product commit, or a product defect
- [x] #2 Drift is repaired in the tests (the doubles accept what the product passes), with each test's assertions kept; a product defect is fixed in product code with a test shown red first
- [x] #3 The four nodes pass, alone and in a `-n 4` run of the file
- [x] #4 The file's green nodes run in a PR lane, with their serial time under the lane's flags recorded in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. The three TypeError nodes: confirm the Send reaches `_record_dispatch`'s double with `_captured_inputs` (added to the call by 077e5bfa09) and let the double accept that keyword, by name, recording what it did before.
2. The `[False]` node: confirm `request_console_repaint` hands `run_worker` the sync method itself since d043e22438, and make the test expect that object; keep every other assertion.
3. Run the four nodes alone and in a `-n 4` run of the file.
4. Lane the file's green nodes (all but TASK-35830's five) as node-id entries at the UI census tail, timed serially with the UI lane's flags.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Causes (AC#1): all four are test-double drift; no product defect.**
- `test_enter_after_a_tab_click_during_a_sync_pass_sends_to_the_clicked_chat`, `test_a_switch_during_a_sync_pass_binds_the_composer_at_once[browser]` and `test_keys_typed_after_a_second_quick_tab_click_go_to_that_tab`: drift after 077e5bfa09 (2026-10-31, "Reconcile Console Send custody and feedback with current owners"). That commit made `chat_screen.py`'s send pass `_captured_inputs=received_inputs` to `_dispatch_console_draft_send` whenever the Send captured its inputs (`chat_screen.py:18430-18433` at base). The product dispatcher (`composer_run_controls._dispatch_console_draft_send`) declares that keyword; the file's `_record_dispatch` double did not, so the Send raised `TypeError` inside the UI action and the tests waited out the send-acknowledgement wait.
- `test_an_activation_that_loses_authority_after_switching_asks_for_a_repaint[False]`: drift after d043e22438 (2026-11-05, "Fix native Console sync coroutine worker admission"). `request_console_repaint` (`UI/Console_Modules/session.py`) now hands `run_worker` the sync method itself, which the worker calls, instead of a coroutine made by calling it. The test still expected `run_worker("sync pass", ...)`, the Mock's return value; the recorded call carried the Mock.

**Fix (AC#2, tests only).** `_record_dispatch`'s double accepts `_captured_inputs=None` by name (it records what it did before, the session and the draft). The repaint test expects `((screen._sync_native_console_chat_ui,), {"exclusive": True, "group": "console-sync"})`. Every assertion is unchanged.

**Red/green (AC#3).** At base the four nodes failed in a `-n 4` run of the file (9 failed, 18 passed; the other five are TASK-35830's). After the fix, each node alone passed (7.4, 7.7, 7.7 and 3.4 s), and the file at `-n 4` gave 22 passed, 5 failed: only TASK-35830's five setup-assertion reds (the four `test_send_refuses_and_keeps_the_draft_when_the_composer_is_bound_elsewhere` cases and `test_send_refuses_when_the_composer_chat_is_off_screen_after_two_switches`), which this task leaves to TASK-35830.

**Lane (AC#4).** Only `test_running_harness_retires_actual_borrowed_database_connections` was censused. The other 13 green test functions join the UI census tail as node ids (all their parameter cases; the two slowest were later split by parameter, see "Review fixes" below); TASK-35830's two red functions stay out. Serially with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`) the 13 entries (21 tests) took 233.7 s at load 58-77 (setup+call+teardown per entry: workspace switch paths 52.9 s, strip buttons 30.1 s, sync-pass binds 26.2 s, lost click 22.6 s, second quick click 17.4 s, two waiting clicks 13.5 s, session tab keyboard 12.9 s, Enter after click 12.6 s, held activation keys 12.3 s, Redirect refusal 10.9 s, tab click highlight 10.6 s, and under 0.01 s each for the two stub tests). The census floor is the measured entry count (shared with TASK-42469's two entries).

**Load flake in two laned nodes, found in the final check (fixed).** A final `-n 4` run of every touched file (load about 52) failed `test_pressing_tab_strip_buttons_leaves_the_keyboard_in_the_composer[close-a]`, and a repro loop of the file's green tests (6 runs, load 45-51) failed `test_pressing_a_session_tab_leaves_the_keyboard_in_the_composer` once: `{'composer': 'Reply with the single word ECHO.'} != {'composer': 'Reply with the single word ECHO.y'}`. That is 2 failures in 8 loaded `-n 4` runs. Both send the "y" through the driver and then read the composer after one `pilot.pause()`. That pause returns once the process looks idle (`wait_for_idle` compares process CPU with wall time), and a CPU-starved worker looks idle before the Key reaches the composer: test synchronisation, not a product race. With the key delivered 0.5 s late (temporary edit, reverted), the old waits failed 4 of 4 cases at the composer check; the fixed waits passed 4 of 4. The fix, `_until_key_lands`, waits up to 10 s until the draft changes or Trace opens, then the unchanged assertion says where the key went. Soak of the file's 22 green tests after the fix: 20 of 20 runs at `-n 4` passed (two concurrent loops of 10, load 38-52).

**Files.** `Tests/UI/test_console_tab_switch_binds_composer.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

**Review fixes (2026-11-23).**
- Census: the two composer functions over the ~30 s entry budget are now listed by parameter, as TASK-42416 did. Serial seconds per entry (setup+call+teardown, UI lane flags, load 87-106):
  - `test_workspace_switch_paths_bind_the_composer_during_a_sync_pass`: `[workspace_switch]` 15.0, `[workspace_new_session]` 14.5, `[failed_open_restore]` 13.7, `[resume]` 12.8.
  - `test_pressing_tab_strip_buttons_leaves_the_keyboard_in_the_composer`: `[close-a]` 14.8, `[console-new-chat-tab]` 13.4, `[console-new-temporary-tab]` 12.8.
  The other 11 composer entries keep the times recorded above. The file's green tests are now 18 entries.
- The key-landing wait no longer has its own helper. Both tests use the file's imported `until` inside `contextlib.suppress(AssertionError)` (10 s), so the unchanged assertion still says where the key went. With the key delivered 0.5 s late (temporary edit, reverted), the two tests' 4 cases passed 4 of 4, as with the old helper; the single-pause version had failed 4 of 4. Soak of the file's 22 green tests after the change: 20 of 20 runs at `-n 4` (two loops of 10, load 63-136).
<!-- SECTION:NOTES:END -->
