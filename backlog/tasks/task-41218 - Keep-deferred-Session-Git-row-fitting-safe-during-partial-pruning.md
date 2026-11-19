---
id: TASK-41218
title: Keep deferred Session Git row fitting safe during partial pruning
status: Done
assignee:
  - '@codex'
created_date: '2026-11-16 02:17'
updated_date: '2026-11-16 02:35'
labels:
  - ci
  - notes
  - lifecycle
dependencies: []
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/actions/runs/43778537684/job/133021823420
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A deferred Session Git row fitter can run while Textual removes child widgets. The row is still mounted and retains its primary child after the secondary child has gone, so the current guard permits a NoMatches crash and prevents the latest visible postflight refresh. Retire row fitting when the row is detached or pruning while preserving normal presentation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A real partially pruned row with its primary child retained and secondary child gone completes its deferred fitter without a missing-widget exception.
- [x] #2 The original stage/postflight refresh scenario and healthy narrow/normal row fitting continue to satisfy their assertions.
- [x] #3 The new regression is included in the UI PR census without reducing its floor or weakening existing assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: routine deferred-callback lifecycle correction using existing attached/pruning rules; no UI geometry, storage, service or permission boundary changes.
1. Read design-language/component-patterns guidance, TASK-36527 and Notes architecture; retain the CI _fit_copy stack as the original failure.
2. Add a deterministic real Textual partial-prune regression that retains the primary child while the secondary exits, and confirm it fails before the production fix.
3. Apply the existing detached/pruning guard to row fitting, keep healthy fitting unchanged, register the new UI case and raise the census floor.
4. Run the new regression, original stage/postflight case and normal fitting/40-by-20 contracts. Independently review; record the disproved child-presence assumption as a testing lesson and close out with evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Session Git row fitter now returns when detached, pruning or childless. This preserves normal fitting and prevents a deferred callback from querying a child removed during real Textual teardown; no blanket missing-widget catch, worker changes, timeout changes or visual changes were added.

Added a deterministic test that holds the primary child's real exit, waits for the secondary child to finish, asserts the attached/mounted/pruning partial tree, and invokes the real fitter. RED reproduced the CI NoMatches exactly. GREEN passed the regression, original stage/postflight refresh case, healthy row intent fitting and 40-by-20 layout checks (8 cases). Census/sharding contracts passed (125 cases), the exact new node is registered and the floor is raised from 2145 to 2146. CI-pinned Ruff 0.16.6 ratchet and F-rule gates pass; backlog ID/readability checks and independent review pass.

Files: Widgets/Library/library_file_notes_git_panel.py, Tests/UI/test_library_file_notes_git.py, UI census/checker and lessons-testing-evidence.md. The lesson records why mounted plus nonempty children did not protect the prior deferred callback.

ADR required: no; routine callback lifecycle fix within existing view/service and design-token boundaries. TASK-36527 provided the prior lifecycle context.
<!-- SECTION:NOTES:END -->
