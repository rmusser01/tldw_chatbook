---
id: TASK-33403
title: SSH session stderr capture keeps the most recent output
status: To Do
assignee: []
created_date: '2026-09-28 20:30'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33202
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A long-lived session keeps only the first 64 KiB of ssh stderr, so the reason printed when it dies (a mux marker, ssh's final error) can be lost, and a status-preserving MUX_ERROR is then misread as UNREACHABLE, which flips a healthy binding to BLOCKED. Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A session's death is classified from the last 64 KiB of its stderr
- [ ] #2 A mux marker printed after more than 64 KiB of earlier stderr still classifies the death as MUX_ERROR, not UNREACHABLE
<!-- AC:END -->
