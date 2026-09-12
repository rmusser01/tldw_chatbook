---
id: TASK-22501
title: >-
  add_conversation is a DEFERRED writer and dies un-retried under any concurrent chunk commit
status: To Do
assignee: []
created_date: '2026-08-26'
labels:
  - database
  - correctness
priority: high
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Found by TASK-22200's adversarial reviewer, reproduced 3/3: `add_conversation`
(`DB/ChaChaNotes_DB.py`, plain `self.transaction()`) is a read-then-write DEFERRED
transaction, so when a paced backfill chunk commits it takes the snapshot-upgrade
`database is locked` INSTANTLY — bypassing the busy handler entirely — and the UI layer does
not retry. `add_message` (IMMEDIATE) survived the same load 10/10.

This refutes one sentence of TASK-22200's own description ("every UI write is also BEGIN
IMMEDIATE"), corrected in that task file. TASK-22200's pacing shrinks the collision window
~10x but cannot close it, and TASK-22215 added a second paced backfill on a second database.

## Acceptance Criteria

- [ ] `add_conversation` uses an IMMEDIATE transaction (or an equivalent retry) and survives a concurrent chunk-commit load probe 10/10
- [ ] The repo is swept for sibling read-then-write DEFERRED writers on user-facing paths; each is fixed or explicitly recorded as safe
- [ ] A regression test reproduces the original failure shape (concurrent committing writer + a deferred read-then-write) and reds without the fix
