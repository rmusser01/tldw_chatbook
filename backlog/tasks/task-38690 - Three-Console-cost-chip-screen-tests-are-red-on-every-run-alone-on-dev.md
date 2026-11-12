---
id: TASK-38690
title: "Three Console cost-chip screen tests are red on every run alone on dev"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-03 11:40
updated_date: 2026-11-05 21:51
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_cost_chip_screen.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down, where `test_active_edit_cancels_an_already_armed_idle_cost_timer` was reported red on dev and on a PR tree. Checked on dev 9bbd5c2515, 2026-11-03 (load 14-29).

Three tests in `Tests/UI/test_console_cost_chip_screen.py` fail on every run alone; two of them also fail every whole-file run:
- **`test_active_edit_cancels_an_already_armed_idle_cost_timer`**: 3 of 3 alone, 1 of 3 whole-file. It fails with `Expected 'mock' to not have been called. Called 1 times.`. The test arms the idle cost timer, starts a run, edits the draft, waits 0.3 s, and asserts that neither `_sync_console_settings_summary` nor `_sync_console_cost_chip` ran. The report traced the call to the 0.25 s `_poll_console_credential_readiness` timer, which re-syncs inside that window. If so, the test cannot tell the timer it pins from an unrelated periodic sync.
- **`test_keyboard_send_cancels_idle_refresh_before_run_starts`**: 2 of 2 alone, 3 of 3 whole-file. It fails with `assert 'send before debounce fires' == ''`: 0.3 s after Enter, while the gateway is still resolving, the composer still holds the draft.
- **`test_unaccepted_predispatch_echo_is_visible_but_excluded_from_context_and_current`**: 2 of 2 alone, 3 of 3 whole-file. It fails with `assert 11 == 0` on the context control state's `request_tokens`.

TASK-35831 already covers this file's intermittent "On next send —" failures and its missing PR lane (its AC#3). Those three reds would fail any lane the file joins. The file last changed in de693b2e61 (2026-10-20).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_active_edit_cancels_an_already_armed_idle_cost_timer, test_keyboard_send_cancels_idle_refresh_before_run_starts and test_unaccepted_predispatch_echo_is_visible_but_excluded_from_context_and_current pass 10 of 10 runs alone on dev
- [x] #2 Each still fails when the behaviour it pins is removed (the idle cost timer survives an active edit, the idle refresh survives a keyboard send, an unaccepted echo counts toward context), shown under temporary source edits
- [x] #3 The cause of each red is recorded, and a test that a periodic poll can fail is rewritten so the poll cannot satisfy or break it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-run the three nodes alone on base 29592ecc9b and probe each red (call stacks for the mocked syncs, draft/run state over time during the held resolve).
2. Active edit: watch the draft-spend timer's own callback instead of the summary/cost syncs every Console sync shares; give the timer a delay it cannot reach inside the test so only the edit can clear it.
3. Keyboard send: drop the composer-cleared assertion that ADR-230 made stale, with the same timer delay.
4. Unaccepted echo: compare the request tokens with the pre-press value (the pressed draft still sits in the composer) instead of 0.
5. Negative controls under temporary source edits; 10 of 10 alone runs per node.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Re-check on base 29592ecc9b (2026-11-05, load ~8):** all three nodes were still red alone, with the filed messages (`Called 3 times`, `'send before debounce fires' == ''`, `11 == 0`). Two different causes. Only the first test could be failed by a periodic poll.

**Causes (AC#3).**
- `test_active_edit_cancels_an_already_armed_idle_cost_timer`: a probe recorded the stack of every call to the two mocked syncs inside the 0.3 s window. In 5 runs (load 15-40), the draft-spend timer was always already cancelled. The calls came from three other places: the 0.25 s `_poll_console_credential_readiness` tick (`chat_screen.py:10804/10805`), the native UI sync's `project_console_tick` -> `_sync_console_rail_and_controls`, and, once, the attach reconcile's `on_screen_resume`. Mocks on `_sync_console_settings_summary` and `_sync_console_cost_chip` cannot tell the timer from any of these, so the filed guess (the credential poll) was right, and was not the only source.
- `test_keyboard_send_cancels_idle_refresh_before_run_starts` and `test_unaccepted_predispatch_echo_is_visible_but_excluded_from_context_and_current`: both assumed that the composer clears when Send is pressed. Since PR3050's current-owner reconciliation (077e5bfa09, 2026-10-31; ADR-230, "only that snapshot can retire its original draft after durable acceptance"), the pressed draft stays in the composer until the turn is durably accepted, which comes after the held `resolve_for_send`. A probe sampled the state for 3 s while the resolve was held: the draft and the store draft stayed, with the status VALIDATING and the echo row present. They cleared only after the release. In the echo test, the 11 tokens are that draft, still counted as the draft. The echo was excluded as intended: a counted echo gives 20.

**Changes** (`Tests/UI/test_console_cost_chip_screen.py` only):
- Active edit: mocks `ConsoleDraftSpendRefresh.refresh` (the timer's own callback) instead of the shared syncs, and waits until the timer is cleared, not a fixed 0.3 s. The polls and ticks call the syncs, never this callback, so they can no longer satisfy or break the test.
- Keyboard send: drops the composer-cleared assertion, with a comment citing ADR-230.
- Echo: asserts that the draft is still in the composer and that `request_tokens == before_tokens`, instead of 0.
- Both timer tests set the instance's `delay_seconds` to 60 s, so only the code under test can clear the timer. The send's dispatch stopped the timer 0.05-0.13 s after Enter at load ~37, against a production delay of 0.2 s, so under heavier load the timer could have fired first.
- The reply waits after the release use the file's 10 s settle timeout. With the old 4 s default they timed out at load ~50-60 (3 of 3 probe batches, 48 copies) while every asserted value was correct.

**Negative controls (AC#2)**, each a temporary source edit, reverted (load 21-28). Each was red on the pinned assertion:
- `route_edit` keeps the timer during an active run: `assert Timer(60.0, ...) is None`.
- `_dispatch_console_draft_send` without its `stop()`: the same assertion.
- `build_console_context_messages` counting every row: `assert 20 == 11`.

**Runs (AC#1)**, each node alone with `-n 4`:
- active edit: 10 of 10 (load 47-66);
- keyboard send: 10 of 10 (load 48-82), then 10 of 10 after the wait change (load 19-21);
- echo: 8 of 10 before the wait change (load 44-62; the two reds were 59-61 s runs, consistent with the 4 s reply wait, though their output was not kept), then 12 of 12 (load 25-36) and 10 of 10 (load 19-29) after it.

Whole file: 37 passed (load ~28).

**Files.** `Tests/UI/test_console_cost_chip_screen.py`; lesson in `backlog/docs/lessons-testing-evidence.md` (a pressed draft stays until durable acceptance).
<!-- SECTION:NOTES:END -->
