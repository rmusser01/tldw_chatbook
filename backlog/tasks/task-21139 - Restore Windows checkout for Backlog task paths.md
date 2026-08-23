---
id: TASK-21139
title: 'Restore Windows checkout for Backlog task paths'
status: In Progress
assignee: []
created_date: '2026-08-23'
updated_date: '2026-08-23'
labels:
  - backlog
  - ci
  - windows
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore Windows CI checkout after TASK-21130 introduced a tracked Backlog
filename containing `>`, which Windows rejects before any project test can run.
Prevent another Backlog task path from making the repository uncheckoutable on
a supported CI platform while preserving task IDs and task content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] TASK-21130 retains its ID and content under a Windows-compatible filename, and every repository reference to its previous path is updated
- [ ] The existing stdlib-only Backlog guard rejects Windows-incompatible filenames across live, completed, and archived task buckets, with focused tests for invalid characters, control characters, reserved device names, trailing dots/spaces, and valid names
- [ ] The affected Windows GitHub Actions jobs progress past checkout, and the guard produces no false positive for the tracked Backlog inventory
<!-- AC:END -->

## References

- `Docs/superpowers/specs/2026-08-23-task-21139-windows-safe-backlog-paths-design.md`
