---
id: TASK-39177
title: Open the import form from Import New Media in the command palette
status: Done
assignee: []
created_date: 2026-11-11 11:03
labels:
- uat
- library
- navigation
dependencies: []
priority: medium
updated_date: 2026-11-11 20:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The command palette offers **Media & Content: Import New Media** but opens the last Library view. A user following the import command can land in an unrelated editor.

### Reproduction

1. On Windows at local `origin/dev` `6bd97b112f`, open an existing Library note in a disposable profile.
2. Switch to Home with **Ctrl+1**.
3. Press **Ctrl+P**, search for **Import New Media**, and run **Media & Content: Import New Media**.
4. Observe that Library returns to the note editor and provides no import path field.
5. Run **Library: Import…** from the palette. The **Import media** form opens.

### Evidence

- [Selected palette command](../../qa/uat-core-workflows-2026-11-11/palette-import-choice.txt)
- [Notes editor after running the command](../../qa/uat-core-workflows-2026-11-11/palette-import-destination.txt)
- [Import form reached through the working command](../../qa/uat-core-workflows-2026-11-11/palette-library-import-destination.txt)

The legacy handler dispatches `TAB_INGEST` without import context. The screen registry aliases that route to Library. The working command dispatches Library with `LIBRARY_NAV_CONTEXT_INGEST`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Running Media & Content: Import New Media from Home or from another Library view opens the Import media form.
- [x] #2 The file, folder or URL field is visible and accepts input after the command.
- [x] #3 Library: Import… continues to open the same form.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: No
ADR path: N/A
Reason: Correct the palette action to use the existing Library import navigation context.

Add command dispatch regression coverage, reuse the working Library import route, and verify the import field and both palette actions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The palette import commands use the Library import context through one helper in `app_command_providers.py`. The regression test failed on the legacy Ingest route and passed with the explicit import context. Live walkthroughs from [Home](../../qa/uat-core-workflows-2026-11-11/palette-import-fixed-from-home.txt) and [Library](../../qa/uat-core-workflows-2026-11-11/palette-import-fixed-from-library.txt) opened the import form and accepted path input. The focused palette and startup run passed 9 tests. Four broader palette failures reproduced on the unchanged base. No ADR required for the route correction.
<!-- SECTION:NOTES:END -->
