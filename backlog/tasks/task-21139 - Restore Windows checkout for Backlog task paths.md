---
id: TASK-21139
title: 'Restore Windows checkout for Backlog task paths'
status: To Do
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
Prevent another valid task record from making the repository uncheckoutable on
a supported CI platform while preserving task IDs and task content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] TASK-21130 retains its ID and content under a Windows-compatible filename, and every repository reference to its previous path is updated
- [ ] An automated repository guard rejects Backlog task paths that Windows cannot create, with focused tests covering invalid path characters and valid existing task names
- [ ] The affected Windows GitHub Actions jobs progress past checkout, and the guard produces no false positive for the tracked Backlog task inventory
<!-- AC:END -->
