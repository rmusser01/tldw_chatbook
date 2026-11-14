---
id: TASK-40311
title: "Three plugin_review tests assert worker-set state after one pause and fail at -n 4 under load"
status: To Do
created_date: 2026-11-12 19:36
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
- [ ] #1 Each press that starts a modal worker in these tests waits for that worker, or for the observed state, before asserting
- [ ] #2 The three tests pass 5 of 5 at `-n 4` under load, and run on a PR lane
<!-- AC:END -->
