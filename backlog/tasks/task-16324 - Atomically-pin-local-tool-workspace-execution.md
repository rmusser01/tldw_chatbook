---
id: TASK-16324
title: Atomically pin local-tool workspace execution
status: To Do
assignee: []
created_date: '2026-08-20 20:05'
updated_date: '2026-08-20 20:06'
labels:
  - security
  - console
dependencies:
  - TASK-16320
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent workspace-root rename or replacement races from retargeting local filesystem and Git tool operations after confinement checks. The current Path-based boundary predates project instructions; selected bindings now add defense-in-depth identity checks, but fully atomic confinement needs a cross-platform execution lease.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Filesystem and Git tool operations remain bound to the originally authorized workspace identity across concurrent root rename replacement symlink junction and reparse-point attempts
- [ ] #2 Mutating tools cannot write outside the originally authorized root under deterministic check/use race tests
- [ ] #3 Read-only tools cannot return content from a replacement root under deterministic check/use race tests
- [ ] #4 The solution works or fails closed on macOS Linux and Windows without unsafe preexec_fn use
- [ ] #5 An ADR records the helper-process or alternative runtime boundary and its lifecycle failure and performance trade-offs
- [ ] #6 Existing configured-workspace and selected-binding tool behavior remains compatible when no drift occurs
<!-- AC:END -->
