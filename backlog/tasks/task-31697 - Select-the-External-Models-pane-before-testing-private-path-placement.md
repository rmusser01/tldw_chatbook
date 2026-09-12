---
id: TASK-31697
title: Select the External Models pane before testing private path placement
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:38'
updated_date: '2026-09-05 18:48'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the external-path placement test explicitly navigate to its lazy pane rather than assuming it is the default Models destination.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The external path assertion runs against the mounted External Models pane selected through the existing window navigation seam.
- [x] #2 Exactly one private path remains inside the dedicated edit view and the Models destination selection passes with no runtime change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the test assumes an External view although Models initially selects llama.cpp. 2. Wait for the external target pane and select it through LLMManagementWindow.active_view, then retain the exact dedicated-path placement assertions. 3. Run the Models destination case and related external-pane adoption cases; scoped static checks. ADR required: no. ADR path: N/A. Reason: test-only explicit navigation to an existing lazy edit view.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Selected the existing lazy External pane explicitly; kept the exact single-private-path and dedicated edit-view assertions unchanged. Models destination plus external-rail adoption: 2 passed in 15.02s (/private/tmp/tldw-review-models-external-paths-final-20260905.xml). Full-file Ruff, scoped formatter, and diff checks pass; reviewed no runtime changes. ADR not required: test-only navigation setup.
<!-- SECTION:NOTES:END -->
