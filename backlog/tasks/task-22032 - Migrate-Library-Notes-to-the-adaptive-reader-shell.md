---
id: TASK-22032
title: Migrate Library Notes to the adaptive reader shell
status: To Do
assignee: []
created_date: '2026-08-24 23:25'
labels:
  - library
  - ui
dependencies:
  - TASK-22031
references:
  - >-
    Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md
  - backlog/decisions/086-library-adaptive-reader-shell.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move Database Notes into the shared Library adaptive reader structure while preserving the existing editor coordinator templates import sync conflict recovery utilities and destructive-action contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Notes list remains mounted beside a permanent work pane with independent list collapse and destination-specific geometry
- [ ] #2 Edit is the default mode and mounting it does not mark the note dirty
- [ ] #3 Edit Preview and Info share the current item-owned draft and Preview renders unsaved draft content
- [ ] #4 Create templates import sync conflicts recovery utilities and destructive actions remain reachable without unmounting the list
- [ ] #5 Selection loading dirty-draft navigation stale workers deletion and retry follow the approved identity and recovery contracts
- [ ] #6 No multi-item draft registry or new Notes authority is introduced
- [ ] #7 Automated list editor conflict geometry focus and capability tests pass with a representative live TUI walkthrough
<!-- AC:END -->
