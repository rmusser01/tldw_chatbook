---
id: TASK-100
title: Wire avatar upload in the ds-native character editor
status: Done
assignee: []
created_date: '2026-06-11 18:12'
updated_date: '2026-06-28 01:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The legacy editor posted CharacterImageUploadRequested but no handler exists anywhere; the ds-native editor shows avatar status read-only. Add a file-picker upload flow on the Personas screen and re-add the button.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Upload button opens a file picker and persists the avatar
- [x] #2 Avatar status reflects the change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the `CharacterImageUploadRequested` message, upload button, and editor-side staged image API.
2. Add widget tests proving the upload button emits the message and staged bytes update avatar status/dirty state.
3. Add the `PersonasScreen` file-picker handler and path-based avatar staging helper using `validate_path_simple`.
4. Add screen tests for valid staging, invalid paths/extensions, stale edit mode, and Save-path persistence.
5. Run focused Personas tests and `git diff --check`.
6. Update acceptance criteria and implementation notes before marking Done.

Plan: `Docs/superpowers/plans/2026-06-27-personas-avatar-upload.md`

ADR required: no
ADR path: N/A
Reason: scoped UI workflow restoration using existing editor, file picker, dirty-state, and character persistence boundaries; no schema, sync, storage, provider/runtime, or application-architecture change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored character avatar upload in the ds-native Personas editor using the existing staged edit workflow. The editor now emits `CharacterImageUploadRequested`, stages raw image bytes in pending character data, updates avatar status, and marks the session dirty. `PersonasScreen` opens an image-filtered picker, validates selected paths with `validate_path_simple`, reads bytes off the UI thread, constrains upload to the active character editor, and persists the staged image through the existing Save flow.

ADR required: no
ADR path: N/A
Reason: scoped UI workflow restoration using existing editor, file picker, dirty-state, and character persistence boundaries; no schema, sync, storage, provider/runtime, or application-architecture change.
<!-- SECTION:NOTES:END -->
