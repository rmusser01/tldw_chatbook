---
id: TASK-40311
title: "Three plugin_review tests assert worker-set state after one pause and fail at -n 4 under load"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 03:52
labels:
- follow-up
- tests
- plugins
- flaky
priority: low
references:
- Tests/UI/test_plugin_review.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-40132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (plugins stream, TASK-40132's covering runs: 2-4 of 8 cases failed per run at `-n 4`, load ~35; all pass serially). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** These tests press a control that starts a modal worker, take one `pilot.pause()`, and assert state the worker sets. Covering runs report false failures, and the tests cannot be laned as written. Only `test_unlock_mismatch_shows_restore_copy_not_passphrase_advice` is on the UI census (`:2908`).

**The tests** (`Tests/UI/test_plugin_review.py`): `test_refresh_growth_keeps_focused_action_and_cleaned_status_painted` (`:79`, assert at `:115`); `test_action_inputs_hide_without_losing_drafts_or_keyboard_focus` (`:1435`, `assert modal.draft.review is not None` at `:1480`); `test_confirmed_profile_save_survives_close_during_child_prune[save-success]` (`:2172`, one pause before `:2216`). The file has 81 bare `await pilot.pause()` calls and already uses `modal.workers.wait_for_complete()` elsewhere. TASK-32695 (In Progress) touches this file for a different test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each press that starts a modal worker in these tests waits for that worker, or for the observed state, before asserting
- [x] #2 The three tests pass 5 of 5 at `-n 4` under load, and run on a PR lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Checked on base 26b2e677dd (2026-11-13). The premise holds: the three tests take one `pilot.pause()` after opening the modal and after a press, then assert.

1. Show the failure at `-n 4` under load before changing anything.
2. In `Tests/UI/test_plugin_review.py`, add a bounded wait for an observed condition. Wait for the modal load to finish (`_loading` false, which the load's `call_after_refresh` focus restore clears) before the first interaction. After each press that starts an awaited handler, wait for the state it sets (`draft.review`, or the refreshed body) with `_busy` clear.
3. Append the three tests at the UI census tail, raise `MINIMUM_FILES`, and measure each entry's serial time with the lane flags.
4. Run the three tests at `-n 4` under load, 5 times.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** The three tests now wait for observed state instead of one `pilot.pause()`. They are on the UI PR lane. Only test code and the census changed.

**Causes found.**
- **Load race.** The dialog's load worker ends with `call_after_refresh(self._loaded)`. `_loaded` clears `_loading` and restores focus to the draft's `focus_id`, which defaults to `plugin-review-close`. One pause after `workers.wait_for_complete()` did not always cover it. Under load the restore landed after the test had focused its own control. The red refresh run showed `app.focused` as `plugin-review-close` instead of `plugin-review-refresh`. While `_loading` is set, `_review_input_changed` also ignores input changes.
- **Long handler.** Refresh and Review action run inside the awaited `@on(Button.Pressed)` handler. `pilot.press` and `pilot.pause` wait for every pump on the screen and raise `WaitForScreenTimeout` after 30 s. The refresh reads 29 operation statuses one at a time. It took 30.8 s and more at load ~70, and 73–75 s at load ~57 (four concurrent cases, measured in a temporary copy).

**Changes** (`Tests/UI/test_plugin_review.py`):
- `loaded(pilot, modal)` waits for the load worker, then polls until `_loading` clears, then takes one pause so the queued `focus()` applies.
- Each press that starts a handler or worker is sent with `Tests/UI/pump_probe.key`, which skips Pilot's idle wait. The test then polls the state that handler sets with `pump_probe.until`, bounded by `SETTLE_SECONDS = 240`:
  - Review action: `draft.review` is set and `_busy` is clear.
  - Refresh: the body shows the cleaned roots and `_busy` is clear. The test then waits until the modal has no pending layout, scroll or after-refresh callback, so the keep-visible scroll has run on fresh geometry. An earlier region-based wait passed on stale geometry; a late-load control run caught it.
  - Confirm (profile save): the existing waits for the gate and the worker are kept, with the bound raised from 3 s and 10 s to `SETTLE_SECONDS`.
- Failure messages name the dialog status, focus and body.
- Bare pauses after synchronous `Select`/`Input` changes are unchanged.

**Red → green.**
- Base at `-n 4` under load:
  - Load 44: 2 of 8 failed, both `refresh_growth` cases (focus on Close).
  - Load ~60–70: the same 2 failed with `WaitForScreenTimeout`, in two runs.
  - The other two tests passed in those runs. Round 13 saw them fail at load ~35.
- Negative control, a temporary autouse fixture that delays `_loaded` by 1 s (removed afterwards):
  - Base: 4 of 8 failed (`refresh_growth` ×2, `action_inputs_hide` ×2).
  - Final code: 8 of 8 passed.
- Final code, the three tests at `-n 4`, 5 runs in a row: 8 passed in each, at load 44–63.

**PR lane.**
- The three tests are appended at the tail of `scripts/ui_pr_gate_census.txt`, and `MINIMUM_FILES` is now 2031.
- Serial times with the lane flags (`-p no:xdist --timeout=180 --durations=0`), run one entry at a time:

| Entry | Time | Load |
|---|---|---|
| `refresh_growth` (2 cases) | 254 s; calls 124 s and 116 s | 47→43 |
| `action_inputs_hide` (2 cases) | 43 s | 43→38 |
| `confirmed_profile_save` (4 cases) | 87 s | 38→39 |

- The refresh cases are the costly ones: 12 data-root creations plus a refresh that reads 29 statuses in sequence. Measured loaded, they sit well inside the 180 s per-test limit, but not by a wide margin.

**Checks.**
- `Tests/CI` census, replay and gate tests: 442 passed.
- `./scripts/preflight.sh` passes.

**Files.** `Tests/UI/test_plugin_review.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:NOTES:END -->

