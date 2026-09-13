---
id: TASK-32524
title: Persist and exchange conversation reasoning prefill pins
status: To Do
assignee: []
created_date: '2026-09-13 03:19'
labels: []
dependencies:
  - TASK-32521
  - TASK-32523
documentation:
  - Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md
  - Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md
  - backlog/decisions/159-console-native-reasoning-prefill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep enabled and disabled reasoning prefill pins with their owning conversation across supported persistence and exchange boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pin text and enabled state round-trip through SQLite resume, promotion, lossless conversation and Chatbook exchange, and Sync v2 with content protection.
- [ ] #2 Concurrent sibling metadata edits survive; stale or failed pin saves remain recoverable; malformed and unknown versions cannot silently enable sending.
- [ ] #3 Next-send state never reaches disk snapshots, defaults, fork inheritance, or exports; unsupported persistent backends refuse lossy saves.
<!-- AC:END -->
