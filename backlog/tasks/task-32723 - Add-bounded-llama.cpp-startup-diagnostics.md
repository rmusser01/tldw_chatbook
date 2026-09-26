---
id: TASK-32723
title: Add bounded llama.cpp startup diagnostics
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 16:11'
updated_date: '2026-09-17 16:57'
labels:
  - llamacpp
  - lab
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make llama.cpp launch failures actionable while keeping private process output within the Lab ownership boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Llama.cpp process output is drained without blocking and retained only as bounded sanitized diagnostics.
- [x] #2 Credentials, paths and prompt-bearing text never enter diagnostic display, copy or unrestricted application logs.
- [x] #3 Launch exit and known failure categories offer recovery guidance without claiming readiness from output.
- [x] #4 Process cancellation, cleanup, high-volume output and existing resource leases are covered with targeted tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/165-llamacpp-tuning-profiles-and-startup-diagnostics.md
Reason: implements ADR-114 with explicit tuning persistence and bounded diagnostics under ADR-165.

Follow the corresponding slice of Docs/superpowers/plans/2026-09-17-llamacpp-management-milestone.md and its linked design. Write failing targeted behavioral tests, implement the minimal owned interfaces, integrate the mounted flow, and verify the affected lifecycle, source, snapshot and settings boundaries. Record review and qualification limits before closeout.

## Renumbering provenance

The Backlog CLI assigned 32714 below the live all-ref/all-worktree maximum 32720. Renumbered immediately before implementation to 32723 to avoid existing branch reservations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented optional llama.cpp-only nonblocking stdout/stderr draining and a 128-entry fixed-category diagnostic buffer attached to the exact launch claim. Unknown/private raw output is suppressed; diagnostics never establish readiness. High-volume, no-newline, split-marker, real cancellation and existing resource-lease tests pass; real runtime smoke confirms clean Stop. ADR: backlog/decisions/165-llamacpp-tuning-profiles-and-startup-diagnostics.md. Files: llamacpp_diagnostics.py, server_lifecycle.py, launcher and pane integration with focused real-child tests. Evidence and platform qualification limits: Docs/superpowers/reviews/2026-09-17-llamacpp-milestone-verification.md. User guide: Docs/User_Guide/lab.md.
<!-- SECTION:NOTES:END -->
