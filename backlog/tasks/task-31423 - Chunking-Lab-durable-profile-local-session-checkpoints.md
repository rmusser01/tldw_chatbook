---
id: TASK-31423
title: Chunking Lab - durable profile-local session checkpoints
status: To Do
assignee: []
created_date: '2026-09-04 23:12'
labels:
  - chunking
  - chunking-lab
dependencies: [TASK-31422]
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recover the latest durable experiment after reopening or crashing, including invalid drafts and completed A/B outputs, without coupling scratch state to template writes. Covers spec section 8 and AC 2, 5, 9, 11-13, 16, 26. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: private storage, transactions, retention, and cross-instance conflict policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A dedicated versioned profile-local SQLite store uses existing private-path protections and atomically publishes results with their checkpoint references; current, previous, and active undo snapshots remain intact.
- [ ] #2 Reopening restores exact sample, raw drafts, pending edits, A/B results, and view state; unfinished work becomes Interrupted with no automatic execution or source re-read.
- [ ] #3 Serialized revision-aware autosaves target 300 ms debounce and a one-second maximum normal typing interval; Saved locally only reflects the latest committed revision and conflicts preserve the losing in-memory state.
- [ ] #4 Crash injection, disk failures, incompatible schemas, concurrent writers, and delayed acknowledgments cannot overwrite valid recovery data, falsely report saved state, or resurrect a cleared epoch.
<!-- AC:END -->
