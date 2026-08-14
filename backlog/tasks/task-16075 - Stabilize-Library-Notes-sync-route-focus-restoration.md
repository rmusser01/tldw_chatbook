---
id: TASK-16075
title: Stabilize Library Notes sync-route focus restoration
status: To Do
assignee:
  - '@codex'
created_date: '2026-08-14 04:31'
labels:
  - testing
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Notes sync open/back focus test wait for the intended post-recompose filter focus without weakening production focus ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sync-route focus test reproduces and then passes deterministically,No production behavior changes,Focused static and diff checks pass
<!-- AC:END -->
