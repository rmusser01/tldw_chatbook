---
id: TASK-19013
title: Declare WorkspaceCreateModal in the Library modal inventory
status: To Do
assignee: []
created_date: '2026-08-21 01:23'
labels:
  - library
  - testing
  - modal
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Library modal inventory gate so it resolves and declares the existing WorkspaceCreateModal launch instead of failing before bidirectional inventory assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Library modal inventory resolves `create_local_workspace` to `WorkspaceCreateModal`.
- [ ] #2 The `WorkspaceCreateModal` contract and any transitive modal edge are declared or explicitly excluded with a recorded reason.
- [ ] #3 The full Library modal-dismissal suite reaches and passes its bidirectional inventory assertions.
- [ ] #4 No production modal behavior changes.
<!-- AC:END -->
