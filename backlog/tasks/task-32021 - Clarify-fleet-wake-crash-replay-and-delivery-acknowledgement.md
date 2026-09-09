---
id: TASK-32021
title: Clarify fleet wake crash replay and delivery acknowledgement
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 03:03'
updated_date: '2026-09-08 06:39'
labels:
  - agents
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wake delivery is stamped only after the turn returns; a crash or failed ledger stamp can replay a completion whose work already ran.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document the difference between result notification deduplication and exactly-once tool execution.
- [x] #2 Specify a crash-window recovery policy and user-visible ambiguous-delivery state before introducing persistence changes.
- [x] #3 Provide a crash/restart test matrix covering acceptance, tool execution, ledger writes, and mark clearing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md
Reason: Defines durable wake-attempt ownership and crash recovery before the budget schema is implemented.
1. Trace acceptance, dispatch, terminal return, delivery stamping, and unseen marks.
2. Reproduce a failed-stamp replay with real SQLite and controller delivery.
3. Define atomic attempt claims and acceptance fences; fail ambiguous recovery closed with saved results.
4. Document the crash matrix and correct misleading exactly-once claims without changing runtime behavior in this design task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the crash/acknowledgement contract in backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md. Unique result claims and prepared/accepted/completed/aborted/review_required attempt states separate admission, execution authority, and notification bookkeeping. Acceptance is a required FULL-synchronized database fence before any automatic model/tool work, including preparation helpers; UI accepted hooks and transcript rows do not authorize execution. Uncertain recovery retains claims and allowance, requires manual review, and discovers state independently of unseen marks. Opening a DB handle or remounting a view does not invalidate live work.

The real SQLite/controller failed-stamp probe measured one completed provider call followed by a second after durable-state claim. This is in-process reconstruction, not a killed-process test or proof of duplicate external side effects. ADR-135 lists the crash/contention matrix from pre-claim rollback through acceptance, approval/tools, terminal stamps, marks, and restart. Three design probes passed; the final combined targeted regression run passed 163 tests. Probe lint/format passes; existing edited code adds no lint findings. AST comparison proves wake module/test edits are documentation-only.

Corrected misleading exactly-once, narrow-acceptance-window, never-lost, and at-most-one-replay wording in the user guide and source/test documentation. Updated TASK-32036/32037 criteria and the linked implementation plans before persistence changes. This completes design only; durable fence and recovery enforcement remain pending. Self-review complete; no full suite or live-provider certification.
<!-- SECTION:NOTES:END -->
