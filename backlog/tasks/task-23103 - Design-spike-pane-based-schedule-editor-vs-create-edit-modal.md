---
id: TASK-23103
title: 'Design spike: pane-based schedule editor vs create-edit modal'
status: To Do
assignee: []
created_date: '2026-08-28 14:05'
labels:
  - ux
  - schedules
  - design
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Schedule creation is a modal while every peer workbench surface edits in the detail pane; the modal's fixed-height layout is also the structural cause of the field-clipping defect filed in this batch. Decide the durable shape: keep-and-patch the modal, or move create/edit into the detail pane (which inherits scrolling, the state banner, and footer-key consistency structurally). Decision only, no implementation - per the stability-over-quick-wins ruling, pick the durable shape rather than the expedient one. Raised as provocative question 2 of the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tbook-ui-screens-scheduling-schedules-workbench-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A written decision (task notes or ADR) compares modal vs pane-based editing against the workbench idiom, keyboard flow, and terminal-height constraints
- [ ] #2 The decision names which follow-up tasks it spawns or closes
- [ ] #3 No implementation beyond throwaway prototyping
<!-- AC:END -->
