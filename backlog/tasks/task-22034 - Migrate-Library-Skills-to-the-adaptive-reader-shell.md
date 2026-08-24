---
id: TASK-22034
title: Migrate Library Skills to the adaptive reader shell
status: To Do
assignee: []
created_date: '2026-08-24 23:28'
labels:
  - library
  - ui
dependencies:
  - TASK-22033
references:
  - >-
    Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md
  - backlog/decisions/085-library-adaptive-reader-shell.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move Skills into the shared Library adaptive reader structure while preserving local-store import editing trust review supporting-file and recovery boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Skills list remains mounted beside a permanent work pane with independent list collapse and destination-specific geometry
- [ ] #2 Overview is the default mode and Edit Trust and Files are explicit destination-owned modes
- [ ] #3 Trust identifies the reviewed revision or fingerprint and existing policy marks prior review stale after applicable changes
- [ ] #4 Supporting files remain read-only unless an existing capability explicitly permits editing
- [ ] #5 Create import trust review recovery and destructive actions remain reachable without unmounting the list
- [ ] #6 Selection loading draft navigation stale workers trust changes deletion and retry follow the approved identity and recovery contracts
- [ ] #7 Automated list editor trust files geometry focus and capability tests pass with a representative live TUI walkthrough
<!-- AC:END -->
