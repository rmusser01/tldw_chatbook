---
id: TASK-41723
title: "Two Library shell nodes fail often in dev's push-to-dev run"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: '2026-11-22 16:12'
labels:
- follow-up
- tests
- library
priority: medium
references:
- Tests/UI/test_library_shell.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41532's review (round 16), reading 19 push-to-dev runs: `test_library_shell_ingest_canvas_submit_clears_path_and_title_keeps_metadata` failed 8 times (also red in run 44190585155) and `test_library_media_initial_error_is_unknown_and_retry_is_unique` failed 4 times. TASK-41532 named only four other nodes, so the job will likely stay red without these. TASK-39152 already re-pinned the ingest-canvas node to wait for the cleared form and it still fails in CI; the CI failure state shows `already_in_library=1`, a lead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each node's failure shape across those runs is recorded, with its cause
- [ ] #2 Each passes 10 of 10 at `-n 4` with the load recorded, or the product defect is fixed with a red-before test
<!-- AC:END -->

## Implementation Notes

**2026-11-22 (round 20, branch fu/r20-libraces): the `test_library_media_initial_error_is_unknown_and_retry_is_unique` half is covered by 1ae18ce611**, the partial port of another contributor's PR #3535 (recorded in TASK-39152). Failure shape: `'Media' == 'Media (2)'` on the title assertion after the Retry wait, in dev push runs 14e13291aa, c1e9d12c2d and d46f753d82 (the three found in the runs read for TASK-39152, 2026-11-13..22). Cause: the Retry's loading sync already removes Retry and moves focus to the type filter, and `_apply` sets `applied_result` before its canvas sync repaints the title, so the old wait could pass on the stale "Media". The wait now also requires the "Media (2)" title, and its timeout message names the title (808c7ab7d2). With the page and facet syncs delayed 0.5 s (a temporary source edit, reverted), the old wait failed 4 of 4 with that message and the new one passed 3 of 3. The ingest-canvas node is not touched, so neither AC is fully met and nothing is ticked: AC#1 still needs that node's shape and cause, and AC#2 still needs 10 of 10 runs for both nodes.
