---
id: TASK-2560
title: Schedules destination visual-parity tests fail on a missing media DB
status: To Do
assignee: []
created_date: '2026-08-06'
labels:
  - tests
  - scheduling
dependencies: []
priority: medium
---

## Description (the why)

`Tests/UI/test_destination_visual_parity_correction.py` has four failing
cases on the **schedules** destination, each raising `Local media DB is
required...`. Confirmed pre-existing on dev and unrelated to the Watchlists
UAT work: found during batch-4 verification only because that file entered
the verification set for the first time there, and reproduced in isolation.

A test that fails on every run teaches the suite's readers to ignore
failures, which is how the `_delete_item` case (task-2330) survived as long
as it did.

## Acceptance Criteria (the what)

- [ ] The four schedules cases either provide the media DB the destination
      genuinely needs, or are skipped with an explicit reason naming the
      dependency.
- [ ] The whole file passes on dev.
- [ ] If the failure reflects a real product requirement (a destination that
      cannot render without a media DB), that requirement is stated in the
      test rather than implied by a crash.

## Notes

Delta re-confirmed during task-3200's review-fix round (2026-08-08, branch
`fix/library-polish-batch`): `Tests/UI/test_destination_visual_parity_correction.py`
currently fails **5** cases on a clean run, not 4. Four are this task's own
schedules/media-DB cases (unchanged, still `Local media DB is required for
local media operations.`):
`test_operational_destinations_use_timing_or_procedure_workbench[schedules-...]`,
`test_operational_empty_or_blocked_states_preserve_workbench_geometry[schedules-...]`,
`test_operational_loading_states_preserve_workbench_geometry[schedules-...]`,
`test_top_level_destinations_keep_primary_workbench_visible_at_compact_size[schedules-contract6]`.
The fifth, `test_mcp_forced_loading_state_stays_inside_workbench`, is a
**different, unrelated** failure (`AssertionError: MCP inspector; assert (6 + 45) <=
42` -- an MCP workbench-geometry overflow, nothing to do with a media DB) that a
task-3200 fix-round report had folded into "this task's territory" by mistake. It
does not belong to this task's AC and is not fixed by anything this task's scope
covers. Correction (final review round of `fix/library-polish-batch`, 2026-08-09):
this test is NOT untracked -- task-2960 ("MCP forced-loading parity regression from
the PR-1385 tall-section change", To Do) names this exact test and already covers
it. Left untouched here regardless -- out of task-3200's scope to fix, and out of
this task's stated scope (schedules/media-DB only) to adopt without widening the AC.
