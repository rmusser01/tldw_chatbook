---
id: TASK-558
title: Wait for Library note conflict controls after state transition
status: Done
assignee: []
created_date: '2026-07-25 17:57'
updated_date: '2026-07-25 18:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the mounted Library note conflict test synchronize on the conflict DOM instead of assuming the state flag and recompose complete atomically.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The conflict test waits for both Overwrite and Reload controls to mount
- [x] #2 The test still verifies conflict copy and preservation of user text
- [x] #3 Repeated focused runs and the full Library shell module pass
- [x] #4 Static checks and task notes record RED evidence and ADR applicability
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reproduced race where the conflict state flag becomes visible before conflict controls mount.
2. Wait on both conflict action selectors after the state transition without weakening copy or text-preservation assertions.
3. Repeat the focused case, then run the full Library shell module and static checks.
4. Record the timing contract and verification.

ADR required: no
ADR path: N/A
Reason: This is a mounted-test synchronization correction and changes no production behavior or architectural boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the mounted note-conflict test to synchronize on the actual Overwrite and Reload controls after observing the conflict state. The production transition intentionally sets the conflict state before storing the conflict snapshot and scheduling recompose, so the prior immediate DOM assertion raced that recompose. RED evidence reproduced the missing Overwrite selector immediately. The existing conflict copy and kept-user-text assertions remain unchanged. Verification: the focused case passes five independent runs; the full Library shell passes 257/257; Ruff, formatter, and diff checks pass. ADR required: no; test-only synchronization. Modified: Tests/UI/test_library_shell.py and this task file.
<!-- SECTION:NOTES:END -->
