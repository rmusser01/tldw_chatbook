---
id: TASK-17660
title: 'Settings: paste-collapse toggle persistence tests red on dev'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17'
labels:
  - settings
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two parameterizations of `Tests/UI/test_destination_shells.py::test_settings_console_paste_collapse_toggle_reflects_and_persists_config` (`[True-True-False]` and `[false-False-True]`) fail on clean origin/dev — verified 2026-08-17 in a detached baseline worktree at `8dc8c2a2c` with identical failures on the task-17653/17659 branch, which touches neither the Settings card's paste controls nor their persistence. Found during task-17653's footer-consumer sweep.

Needs a bisect against recent Settings/Console-Behavior merges (the status-row toggle, the selection-feedback arc, or another recent landing may have shifted the card's control order or the persistence seam the test drives).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The failing parameterizations are green on dev, either by fixing the regression they caught or by updating the test to the intended contract (decided by reproducing the toggle flow live first)
- [x] #2 The task records which merge introduced the red
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce; read the failure's visible-text dump (it showed "Save (s) — no changes": the checkbox click never staged a draft).
2. Hit-test the click coordinates; trace the miss to its mechanism; verify against a bundled harness before attributing.
3. Fix at the mechanism; record the generalizable trap in lessons-testing-evidence.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
No regression existed and no bisect was needed — the mechanism was the third instance of the bundle-less-harness trap, this time changing LAYOUT MODE: without the bundle, `.settings-secondary-card { height: auto }` never applies, the card is clamped to a container-default fraction of the pane, and when the card's content grew (Status row placement [task-17652] + Selection side chat [selection-feedback]) the paste checkbox laid out past the clamp with sibling detail rows painted over its coordinates — `pilot.click` missed silently (returned False), nothing staged, and Save truthfully reported "no changes". A bundled probe on the same build showed the card auto-growing (h=123) and the control clickable after ordinary scrolling: the real app was never broken.

Fix: the test gets a bundle-loading harness subclass, scrolls the toggle into view the way a user does, and asserts the click's return value so a future miss fails AT the click. Both parameterizations green; full-repo lesson appended to lessons-testing-evidence.md ("a bundle-less harness does not just hide styles — it changes layout mode").

Files: `Tests/UI/test_destination_shells.py`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
