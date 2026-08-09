---
id: TASK-3311
title: >-
  Ingest Clear-button focus race can silently swallow a typed path
status: To Do
assignee: []
created_date: '2026-08-08 00:30'
labels:
  - library
  - ingest
  - ux
  - bug
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the live verification of the 3300-3305 arc (2026-08-08, worktree branch feat/media-ingest-ux-parity). Intermittent — 2 of 4 Clear clicks did NOT return focus to the path field; subsequent typing was hijacked: once the path's tail landed in the rail search box, once the typed path vanished entirely (a leading `/` from an unfocused state likely triggered the global "/ focus search" binding). Two controlled retests refocused correctly, so it is a race, not deterministic — plausibly the ⚠ tooling-warning block's relayout racing the post-Clear refocus. Consequence is silent loss of a typed path plus keystrokes running a Library search.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 After Clear, focus deterministically lands on the path field even when the preflight/warning region relayouts concurrently (looped live or harness reproduction, not a single pass)
- [ ] #2 A typed leading "/" immediately after Clear edits the path, never triggers the focus-search binding
<!-- AC:END -->
