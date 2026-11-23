---
id: TASK-42467
title: "Four Console tab-switch composer tests are red on dev: three because the send double rejects a keyword the product now passes, and one whose sync-pass worker gets a Mock"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
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
- [ ] #1 Each of the four nodes has a recorded cause: test-double drift after a named product commit, or a product defect
- [ ] #2 Drift is repaired in the tests (the doubles accept what the product passes), with each test's assertions kept; a product defect is fixed in product code with a test shown red first
- [ ] #3 The four nodes pass, alone and in a `-n 4` run of the file
- [ ] #4 The file's green nodes run in a PR lane, with their serial time under the lane's flags recorded in the notes
<!-- AC:END -->
