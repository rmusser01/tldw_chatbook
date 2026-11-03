---
id: TASK-38690
title: "Three Console cost-chip screen tests are red on every run on dev"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_cost_chip_screen.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies:
- TASK-35831
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down, where `test_active_edit_cancels_an_already_armed_idle_cost_timer` was reported red on dev and on a PR tree. Checked on dev 9bbd5c2515, 2026-11-03 (load 14-29).

Three tests in `Tests/UI/test_console_cost_chip_screen.py` fail on every run, alone or in the file:
- **`test_active_edit_cancels_an_already_armed_idle_cost_timer`**: 3 of 3 alone, 1 of 3 whole-file. It fails with `Expected 'mock' to not have been called. Called 1 times.`. The test arms the idle cost timer, starts a run, edits the draft, waits 0.3 s, and asserts that neither `_sync_console_settings_summary` nor `_sync_console_cost_chip` ran. The report traced the call to the 0.25 s `_poll_console_credential_readiness` timer, which re-syncs inside that window. If so, the test cannot tell the timer it pins from an unrelated periodic sync.
- **`test_keyboard_send_cancels_idle_refresh_before_run_starts`**: 2 of 2 alone, 3 of 3 whole-file. It fails with `assert 'send before debounce fires' == ''`: 0.3 s after Enter, while the gateway is still resolving, the composer still holds the draft.
- **`test_unaccepted_predispatch_echo_is_visible_but_excluded_from_context_and_current`**: 2 of 2 alone, 3 of 3 whole-file. It fails with `assert 11 == 0` on the context control state's `request_tokens`.

TASK-35831 already covers this file's intermittent "On next send —" failures and its missing PR lane (its AC#3). Those three reds would fail any lane the file joins. The file last changed in de693b2e61 (2026-10-20).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_active_edit_cancels_an_already_armed_idle_cost_timer, test_keyboard_send_cancels_idle_refresh_before_run_starts and test_unaccepted_predispatch_echo_is_visible_but_excluded_from_context_and_current pass 10 of 10 runs alone on dev
- [ ] #2 Each still fails when the behaviour it pins is removed (the idle cost timer survives an active edit, the idle refresh survives a keyboard send, an unaccepted echo counts toward context), shown under temporary source edits
- [ ] #3 The cause of each red is recorded, and a test that a periodic poll can fail is rewritten so the poll cannot satisfy or break it
<!-- AC:END -->
