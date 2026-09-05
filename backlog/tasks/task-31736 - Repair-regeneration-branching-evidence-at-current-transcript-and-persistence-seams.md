---
id: TASK-31736
title: >-
  Repair regeneration branching evidence at current transcript and persistence
  seams
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:30'
updated_date: '2026-09-05 19:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore meaningful full branching regression coverage for visible failure notices and durable regenerated siblings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failed and mid-conversation regeneration preserve the original model context and exactly one transcript-only error notice.
- [x] #2 Successful regenerated sibling is verified through real SQLite persistence with its durable content, parent and terminal status.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce existing branching failures with the agent-notice repair absent. 2. Assert the exact current transcript path including the already-shipped transcript-only system notice while retaining original context and failed sibling checks. 3. Replace the stale partial persistence fake with the existing real SQLite-backed fixture and durable assertions. 4. Run complete branching and adjacent files plus static checks. ADR required: no. ADR path: N/A. Reason: test-only corrections at existing TASK-571 recovery and current durable-generation boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Original runtime baseline independently reproduced three pre-existing branching failures (3failed/3passed): two exact paths omitted the already-shipped transcript-only SYSTEM error notice; one partial fake lacked terminal-generation projection persistence. The path assertions now include the exact notice after the restored anchor and preserve failed sibling/original context checks. Replaced the recording fake with real SQLite and assert completed durable sibling content,parent,state and unchanged original row, with explicit controller/database teardown. Six-file combined376passed48.70s; XML /private/tmp/tldw-review-agent-controller-regeneration-verified.xml. Full test lint and formatting pass; whitespace clean. ADR:no/N/A, test-only correction. Independent review pending.

Independent review caught current-thread-only database teardown leaving four worker-owned SQLite descriptors. Corrected finalization to await controller.shutdown then quiesce the exact fixture database connections; setup writes are inside try/finally. Complete branching file rerun required after this correction.

Fresh post-cleanup complete branching file:6 passed in0.73s with only the installed RequestsDependencyWarning; no resource warning. XML /private/tmp/tldw-31736-branching-quiesced.xml. Full-file Ruff lint and formatting pass; diff whitespace clean. Independent review issue addressed; no runtime persistence change.
<!-- SECTION:NOTES:END -->
