---
id: TASK-42416
title: "The Console edge-focus restyle tests click the Inspect rail open control before it has a region and flake in CI"
status: Done
assignee: ['@claude']
created_date: 2026-11-22 21:17
updated_date: 2026-11-23 02:21
labels:
- follow-up
- ci
- console
- tests
- flaky
priority: high
references:
- Tests/UI/test_console_edge_focus_restyle_scope.py
- Tests/UI/test_console_left_rail.py
- Tests/UI/test_console_inspector_navigation.py
- Tests/UI/test_console_right_rail.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths start at the repository root.

**Why.** Both tests in `Tests/UI/test_console_edge_focus_restyle_scope.py` start with `assert await pilot.click("#console-inspector-rail-open")` (lines 83 and 158), straight after `make_console_pilot(size=(160, 45))` (from `Tests/UI/test_console_left_rail.py:42`) yields, with no wait for layout. `Pilot.click` returns False when the target has no region yet, so on a loaded runner the first assert fails before the test reaches what it pins. The file runs in "PR Fast Lane (admission-sensitive 2)" (`derived-artifacts.yml:818`). In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), that lane failed in 9 jobs, from 2026-11-22 04:05Z to 17:57Z, and every one failed on this file: runs 44974742439, 44974765316, 44974790480, 44977790794 attempts 2 and 3 (the two merge-queue attempts of #3653 it sank, jobs 137095986403 and 137178739290), 45016281720 attempt 2, 45034674868 attempt 2, 45050954944 and 45050972963. The same step logs `FAILED Tests/QA/test_project_chat_uat.py::test_project_chat_uat[fenced]` lines; those come from that step's nested negative-control pytest run and are expected, not failures of the job.

TASK-41632 fixed the same kind of fixed-wait flake in the `_open_console_inspector` helpers of two other files; this file was not one of them. `Tests/UI/test_console_inspector_navigation.py:116` (`_open_inspector`) and `Tests/UI/test_console_right_rail.py` click the same control the same way, without checking the result.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both tests wait until the Inspect rail's open control is displayed and has a region, with a load-tolerant deadline (or use an existing shared rail-open helper that does), before clicking it
- [x] #2 Each test then confirms the rail opened before it measures restyles
- [x] #3 With the control's first layout delayed by a temporary edit (reverted), the old tests fail at the click and the new ones pass
- [x] #4 The file passes in a soak of at least 20 runs under parallel load, recorded in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Instrument the click locally (region, display and rail state of `#console-inspector-rail-open` at the click, and the widget Pilot hit-tests) and read the failed CI jobs to name the race.
2. Reproduce the CI message (`assert await pilot.click(...)` -> `assert False`) with a temporary forced delay on the handle's first layout, then revert.
3. Fix where it lives: if the control simply has no region yet, a test wait (displayed + non-zero region, load-tolerant deadline) before the click, then confirm the rail opened before measuring; if the product can drop a click on a laid-out control, fix the product with a red-then-green test.
4. Sweep sibling tests that click `#console-inspector-rail-open` right after mount with no layout wait; fix those with the same cause.
5. Soak the file 20 times at -n 4, run the admission-sensitive 2 step once with its CI flags, preflight, ruff F.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The race was not a missing region.** `make_console_pilot` (`Tests/UI/test_console_shell_regions.py`) hid the first-run setup overlay once, after a fixed `pause(0.2)`. Every `ConsoleSetupModal.sync_card_state` sets the overlay's display back to "blocking" (`console_setup_modal.py:243`), and the screen then focuses its action. On a local mount a spy counted 11-14 setup syncs before the hide. They came from the 4 Hz `_poll_console_credential_readiness` timer's first full pass, `_refresh_console_hooks`, the `_sync_console_identity_surfaces` retry and the transcript ticks. The last one landed only 0.13-0.55 s before the click. When one sync lands after the hide, the overlay covers the Inspect handle. The control is still displayed with its full region (150, 6, 10, 32), but `get_widget_at` finds the overlay, so `pilot.click` returns False. That is the CI `assert False` at :83 (8 jobs) and :158 (job 137095986403). Locally the control had its region at every click: 8/8 and 48/48 probe mounts at -n 4/-n 12 (load 20-36), and 12/12 serial runs of the step's tail. The handle never recomposed after mount (label and badge unchanged).

**Fix (test synchronisation, in the shared harness and the tests).**
- `make_console_pilot(hide_setup_overlay=True)` now keeps the overlay hidden through every later sync, and stops it taking focus. It wraps that instance's `sync_card_state` and makes `focus_primary_action` a no-op.
- New `click_inspect_rail_open(pilot)` in the same module. It waits, on a 15 s deadline that only bounds a real failure, until `#console-inspector-rail-open` is displayed, has a region and is the widget at its click point. It keeps `assert await pilot.click(...)`, then waits until `#console-right-rail` is displayed with width > 0 before returning (AC#2).
- Both edge-focus tests use it, and so do the three Inspect opens in `test_console_edge_rail_geometry.py`. Every behaviour assertion is unchanged.

**Red/green (all temporary edits reverted):**
- Overlay race (CI shape). The credential poll's first pass was held until the overlay was hidden, at a 0.05 s interval. The base tests failed 6/6 with `assert False` at :83 and :158. The fixed tests passed 5/5 runs.
  - Control: the wait helper without the durable hide failed with `Inspect rail open control never became clickable: display=True region=Region(x=150, y=6, width=10, height=32)`. A wait alone cannot fix an overlay that stays up.
- AC#3. The open button was hidden for 4 s after the handle mounted. The base tests failed 4/4 with `assert False` at :83 and :158. The fixed tests passed 4/4 runs.

**Sibling with the same cause, fixed:** `Tests/UI/test_console_edge_rail_geometry.py`, the only other user of this harness that clicks (lines 134, 160, 230). Both it and `test_console_shell_regions.py` are plain tests that fail 60/60 locally with `RecoveryRequired: raw_source_selection_changed` at base too. So I verified them under a temporary module-level `pytestmark = pytest.mark.bootstrap_profile`, the marker the edge-focus file uses for the same harness, and removed it afterwards.
- At base, `test_live_resize_reasserts_edge_ownership_and_preserves_rail_intent` failed 2/2: resize syncs re-showed the overlay, and the `_assert_edge_owned_workbench` hit-test failed.
- With only the display hide it then failed on focus (`app.focused` was `console-setup-modal-action`), which is why the harness also stops the hidden overlay taking focus.
- Final run: 62 passed across the three files.

**Not the same cause:** `test_console_left_rail.py:78`, `test_console_context_rail_header.py:153`, `test_workbench_visual_snapshots.py:581`, `test_console_right_rail.py` and `test_console_inspector_navigation.py:118/1776`. Their harnesses configure a ready provider (`_configure_native_ready_console`), so no overlay is shown. A spy showed the handle laid out about 2 s before the click and never recomposed.

**Soak (AC#4):** 20 sequential runs of the file at `-n 4` passed 20/20 (40/40 tests), at 1-minute load 30.5-49.3. Runs 17-20 overlapped the lane step.

**Lane step:** `r19-step.sh ... "Run admission-sensitive suites 2" -p no:xdist --timeout=300` (94 targets, the workflow's flags) gave 1512 passed, rc 0, in 25:49 at load 30-62. The edge-focus tests took 5.64 s and 6.07 s.

**Checks:** preflight passed; `ruff --select F` passed on the three files. No product module changed and no ratchet lists these files. No census edit: the file is already a `derived-artifacts.yml` admission-2 target.

**Lesson:** a "re-shown overlay" variant was added to `backlog/docs/lessons-testing-evidence.md` (Pilot clicks section).

**Files:**
- `Tests/UI/test_console_shell_regions.py`
- `Tests/UI/test_console_edge_focus_restyle_scope.py`
- `Tests/UI/test_console_edge_rail_geometry.py`
- `backlog/docs/lessons-testing-evidence.md`
<!-- SECTION:NOTES:END -->
