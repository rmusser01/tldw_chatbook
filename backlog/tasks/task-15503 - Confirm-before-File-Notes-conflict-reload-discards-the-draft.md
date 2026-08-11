---
id: TASK-15503
title: Confirm before File Notes conflict reload discards the draft
status: To Do
assignee: []
created_date: '2026-08-11 20:56'
labels:
  - notes
  - filesystem
  - recovery
  - ux
  - data-safety
dependencies: []
references:
  - >-
    backlog/tasks/task-399.8.2 -
    B1b2-Build-bounded-conflict-comparison-and-resolution-UX.md
  - backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md
  - >-
    .impeccable/critique/2026-08-11T20-58-28Z__ok-widgets-library-library-file-notes-workspace-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The current conflict and error recovery surface says the draft is preserved, but activating Reload immediately replaces the editor with disk bytes. Add an explicit, keyboard-safe destructive confirmation now, while the broader three-sided comparison and resolution experience remains tracked by TASK-399.8.2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In conflict or error states, the reload action explicitly says that it will discard the current draft and load disk bytes.
- [ ] #2 The first reload activation never replaces editor contents; it opens a distinct confirmation state whose safe default is cancel.
- [ ] #3 Cancel and Escape close confirmation, preserve the exact draft and conflict state, and return focus to the reload opener.
- [ ] #4 Confirm revalidates the active root, file identity, session generation, and current disk state before replacing the draft; a stale or unavailable target fails closed with actionable copy.
- [ ] #5 Mounted tests prove preservation on first activation and cancellation, intentional replacement only after confirmation, keyboard operation, and complete copy at 40x20 and a normal width.
<!-- AC:END -->
