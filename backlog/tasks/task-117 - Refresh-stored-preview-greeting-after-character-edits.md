---
id: TASK-117
title: Refresh stored preview greeting after character edits
status: Done
assignee: []
created_date: '2026-06-11 13:25'
updated_date: '2026-06-28 03:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PersonasPreviewPane keeps the original greeting for Reset; after editing first_message the pre-edit greeting is restored. Add a set-greeting-without-reseed seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reset after an edit uses the updated greeting
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: bounded Personas preview UI state bugfix; no storage/schema/sync/service boundary changes.

1. Reproduce the stale reset greeting behavior with a focused mounted regression.
2. Verify the regression fails for the expected reason before changing production code.
3. Add the smallest preview-pane seam to refresh the stored reset greeting without reseeding the visible transcript.
4. Wire the character-loaded/save path to update the stored greeting for the active selected character.
5. Run focused Personas preview tests and git diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a mounted Personas preview regression proving that a same-character reload keeps the visible transcript stable but updates the greeting used by Reset. Implemented `PersonasPreviewPane.refresh_greeting_seed()` and wired `PersonasScreen._handle_character_loaded()` to call it when the currently seeded character is reloaded after a save/edit, avoiding an unnecessary transcript reseed while ensuring Reset restores the updated first message. Verification: targeted regression passed, full `TestPreviewIntegration` passed, full `Tests/UI/test_personas_workbench.py` passed, and `git diff --check` passed. ADR required: no; this is a bounded UI state bugfix with no storage, schema, sync, or service-boundary changes.
<!-- SECTION:NOTES:END -->
