---
id: TASK-42627
title: Wait for the rail restore in the create-workspace scroll test
status: Done
assignee:
  - '@claude'
created_date: '2026-11-27 05:50'
updated_date: '2026-11-27 17:00'
labels:
  - tests
  - flaky
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_post_release_workspaces_library_depth.py::test_create_workspace_preserves_rail_scroll` failed on Linux UI Fast Lane shard 35 with "rail scroll reset to top after the create-workspace recompose" (`scroll_y` 0.0): twice on #3769, once on #3472 and once on #3777. A full-tier PR that fails a shard twice is evicted from the merge queue after hours of CI, so this flake cost whole queue cycles.

The test should fail only when the Library rail really loses its scroll offset after a workspace is created, not when a slow runner checks between the rebuild and the restore.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test waits for the recompose's new rail and its restored offset instead of checking at a fixed 0.6 s
- [x] #2 The test passes when the screen's recompose is slowed, and fails when the rail-scroll restore is removed
- [x] #3 The test no longer passes on the old rail when the recompose has not run yet
- [x] #4 `Tests/UI/test_library_details_hanging_indent.py`, which imports this file and so runs in its light job, passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the order of the restore and the recompose with a probe.
2. Reproduce the CI failure by slowing the recompose.
3. Replace the fixed pause with a bounded wait on the new rail and its offset.
4. Check the fix under slowed recomposes and with the restore removed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** The product restore is correct. `LibraryScreen._preserve_library_rail_scroll` re-applies the offset one layout after the recompose. A probe on dev 9926834bc8 traced the order: the recompose ran from 0.009 s to 0.271 s after the confirm, the restore landed at 0.381 s, and the test checked at 0.6 s. A 0.3 s delay injected into `LibraryScreen.recompose` moved the recompose end to 0.563 s. The check at 0.650 s then saw the new rail at `scroll_y` 0.0, and the restore landed 2 ms later. That is the CI failure. A slower runner puts the recompose end near 0.6 s. When the recompose ended after 0.6 s, the old test passed on the old rail without testing anything. Holding an app batch across the confirm did not lose the restore.

**Fix.** The test now polls (up to 5 s) until `#library-rail` is a new widget with a non-zero offset, tolerating the moment between the old rail's removal and the new one's mount. It then checks that the offset survives another 0.3 s.

**Evidence.**
- With 0, 0.3, 1.0 and 1.5 s delays injected into the recompose, the new test passes at every delay. Dev's version fails at 0.3 s, and in one of two runs at 0 s. At 1.0 s it passes only because the recompose has not run by its 0.6 s check.
- With `self._preserve_library_rail_scroll()` removed, the new test fails with "rail scroll reset to top after the create-workspace recompose".
- The whole file passes with `-n 4` (15 passed).

**Observed, not fixed.** These failures appeared only under an artificial load of two concurrent `-n 4` runs of this file, never in CI:
- `_wait_for_library_shell_ready`'s 2 s cap ran out;
- `_wait_for_selector`'s 2 s cap (`Tests/UI/test_destination_shells.py`) ran out;
- in `test_library_workspaces_create_local_workspace_mouse_clicks`, the click on the create button sometimes never pushed the modal.

**AC #4 (the PR's light run, 2026-11-27).** The light job also runs the files that import this one. `test_library_details_hanging_indent.py::test_a_wrapped_details_row_paints_its_continuations_indented` failed there on both sizes with `RecoveryRequired: raw_source_selection_changed`. It fails the same way on dev d2ad1e5330, and no PR lane runs the file. Its mounted test builds the app with `test_library_shell._build_test_app`, which needs the `bootstrap_profile` fixture; every test in `test_library_shell.py` has that marker. With the marker the file passes (5 passed).

Files: `Tests/UI/test_post_release_workspaces_library_depth.py`, `Tests/UI/test_library_details_hanging_indent.py`.
<!-- SECTION:NOTES:END -->
