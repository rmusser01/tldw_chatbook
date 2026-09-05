---
id: TASK-31671
title: Align Scheduling visual tests with local-only and loading states
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:08'
updated_date: '2026-09-05 18:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep destination visual assertions aligned with the shipped local-only sync collapse and the visible Create action during initial loading.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-only visual coverage asserts the sync-off explanation and hidden server plumbing.
- [x] #2 Loading geometry coverage recognizes the visible enabled Create action while retaining loading worker cancellation and disabled Console follow assertions.
- [x] #3 All Scheduling parametrizations of affected destination visual tests pass without product layout changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Retain reproduced RED evidence: Scheduling-only visual7cases yielded2failures for Last pull and loading action; read-only geometry probe showed disabled Follow y50 versus enabled Create y13 at140x42. 2. Update only Scheduling hunks in Tests/UI/test_destination_visual_parity_correction.py: assert local sync-off note and explicitly hidden timestamps/owner buttons; include the Create button in the loading-state action contract and assert it enabled. 3. Preserve all existing pane, viewport, marker, disabled follow, loader-start and cancellation assertions. 4. Run all Scheduling parametrizations of the complete affected visual file; root/Library agent owns unrelated route cases. Run scoped lint/format/diff checks and request parent review. 5. Record evidence, mark done, scoped commit. ADR required: no. ADR path: N/A. Reason: test-only alignment to existing TASK-23105 local-only collapse and shipped rail Create action, no UI policy or layout change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated only Scheduling visual test expectations: local-only state requires the honest sync-off note and hidden server/timestamp controls; loading coverage verifies the rail Create button is visible and enabled while preserving disabled Follow, pane geometry, loading start and worker cancellation assertions. Reproduced original7cases with2failures; isolated geometry probe established Follow y50 below140x42 viewport versus enabledCreate y13. Final exact gate: pytest Tests/UI/test_destination_visual_parity_correction.py -k schedules -o addopts= -n2;7passed16.93s, report /private/tmp/tldw-31671-scheduling-visual-final.xml. Scoped Ruff lint and changed-range format checks plus diffcheck and parent independent review passed. No product CSS, behavior, or unrelated destination expectations changed. ADR required:no; routine test-only alignment to existing product behavior.
<!-- SECTION:NOTES:END -->
