---
id: TASK-2210
title: Console staged evidence concurrent-send double-spend window
status: To Do
assignee: []
labels:
  - console
  - rag
dependencies: []
priority: medium
---

## Description

PR-4 made Console staging consume-on-send (release gated on the controller's prepend predicate). The launch context is read-not-taken before the capture's async work, so two sessions sending concurrently within one capture window can both prepend the same staged bundle; the identity guard prevents a double CLEAR but not a double SPEND. Strictly narrower than the pre-PR-4 behavior (staged evidence rode every send in every session forever), and single-UI-loop overlap is rare — but a take-then-restore-on-block design would close it. Restore-on-block has its own hazard class (a blocked send must put evidence back without racing a new staging); design before coding.

## Acceptance Criteria

- [ ] Two concurrent sends can no longer both consume the same staged bundle
- [ ] A blocked/failed send still retains (or restores) staging — the discard-safety property from PR-4 Task 1 holds
- [ ] The identity-guarded release semantics stay pinned
