---
id: TASK-42425
title: "Three Console projection-cadence tests fail UI Fast Lane 36 under CI load"
status: To Do
assignee: []
created_date: 2026-11-22 22:10
updated_date: 2026-11-22 22:10
labels:
- follow-up
- ci
- console
- tests
- flaky
priority: high
references:
- Tests/UI/test_console_projection_reads_off_loop.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22), in the CI runs the round's filing read. Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), the required "UI Fast Lane (36)" failed in 11 jobs, from 2026-11-20 09:54Z to 2026-11-22 20:26Z, each on one node of `Tests/UI/test_console_projection_reads_off_loop.py`, a whole-file census entry:
- `test_explicit_syncs_of_a_run_start_share_the_next_ticks_projection` (line 410), 7 jobs (including runs 44954347282, 44974749607, 45031398878, 45048643145 attempt 2, 45050972963): `assert calls[-1] is True` at line 491 (`False is True`), or `assert late_requests and ...` at line 522 (`assert ([])`).
- `test_a_full_sync_slower_than_the_display_life_does_not_replay_itself` (line 227), 1 job (45050945775): `a FULL sync whose awaits outlived the display replayed itself` at line 266.
- `test_a_full_sync_asked_for_during_a_pause_runs_once_after_resume` (line 1827, `@private_profile_test`), 3 jobs (including 44976516789): `assert started_at < armed_at + delay` at line 1914 fails in the child.

The file passes locally at the base (`-n 4`, load 6-8).

All three pin the Console's sync and projection cadence (coalescing a run start's syncs into the next tick, not replaying a slow FULL sync, running a paused request once after resume). They drive a fake clock but still depend on real task ordering, so a loaded runner can reorder the awaits they count. Whether any failure is a real cadence bug (a replayed FULL sync, a request owed to the wrong tick) is not known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each of the three nodes, the notes record whether the CI failure is test ordering or a product cadence defect, with the evidence
- [ ] #2 A defect is fixed in product code with a test shown red first; an ordering dependence is removed from the test, shown needed by a temporary delay (reverted) that fails the old test
- [ ] #3 The file passes in repeated runs under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->
