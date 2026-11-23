---
id: TASK-42462
title: "The Console Inspector-navigation tests run on no PR lane and are red on dev: the staged-owner-sync test times out every time and four others fail intermittently"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- console
- inspector
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_console_inspector_navigation.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-24705
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), in TASK-42415's covering runs. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths start at the repository root.

**Why.** `Tests/UI/test_console_inspector_navigation.py` (35 test functions, 39 cases) is in no PR lane: it is not in `scripts/ui_pr_gate_census.txt` and no `derived-artifacts.yml` step names it. So its reds report nowhere. Locally every case fails on the known `RecoveryRequired: raw_source_selection_changed` trap, which is out of scope here. Under a temporary `bootstrap_profile` mark the file is red the same way at plain dev 2f7790873a and on TASK-42415's branch:
- two tests time out every time: `test_staged_owner_sync_drives_ten_eleven_ten_cue_and_clamp` and `test_responsive_hide_and_reveal_hands_off_focus_without_losing_offset`;
- one or two of four others fail in each run, rotating between runs: `test_n_and_p_are_rail_local_no_wrap_and_editable_input_keeps_printable_key` (line 486), `test_navigation_focuses_overflow_viewport_contains_header_and_preserves_state` (line 640), `test_empty_owner_is_settled_absence_with_one_outer_fallback` (line 1407) and `test_structural_control_and_boundary_removal_recovers_without_stale_reentry` (line 1577);
- whole-file runs failed 4 and 3 times at base, and 4 and 3 times on the branch.

**Earlier records.** TASK-32470 (open, medium, 2026-09-11, "Fix dev-tip test harness drift in console suites") first listed this file's staged owner sync cue and responsive hide/reveal handoff tests as red on dev a0b8f96416. TASK-24705 (open, high) owns `test_responsive_hide_and_reveal_hands_off_focus_without_losing_offset`, bisected there to d5e1a26ab3, and its notes list the staged-owner-sync test as a separate red to triage. This task covers the staged-owner-sync test, the four intermittent ones and the lane; it does not re-fix TASK-24705's node. TASK-42416's notes record that this file's harness configures a ready provider, so the setup-overlay race it fixed does not apply here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the five nodes (the staged-owner-sync test and the four intermittent ones) has a recorded cause in the notes: test drift after a named product change, a test race, or a product defect
- [ ] #2 Drift and races are repaired in the tests, with their assertions kept; a product defect is fixed in product code with a test shown red first
- [ ] #3 `test_staged_owner_sync_drives_ten_eleven_ten_cue_and_clamp` passes, and the four intermittent nodes pass in repeated runs of the file under parallel load (at least 20 runs, or a soak recorded in the notes)
- [ ] #4 Every green node of the file runs in a PR lane, with its serial time under the lane's flags recorded in the notes
<!-- AC:END -->
