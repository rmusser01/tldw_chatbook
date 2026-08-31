---
id: TASK-26016
title: 'Compaction: bound the auxiliary call with a timeout'
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
labels:
  - console
  - context
  - reliability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The compaction model call is unbounded. Verified on origin/dev: a grep for timeout across Chat/console_context_compaction.py returns zero; the attempt is tracked with start and finish records (:1675, :2125) and pricing provenance, but nothing cuts it off. A hung summarizer therefore blocks the send that triggered it, with the composer waiting on it. Hermes bounds compaction with progress-aware inactivity budgets plus an absolute ceiling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The auxiliary compaction call is bounded by a configurable timeout with a documented default
- [ ] #2 A timed-out compaction is recorded as a distinct terminal state, not conflated with a model error or a cancellation
- [ ] #3 On timeout the existing CompactionFailureBehavior applies (stop-and-ask or omit-older-context) rather than a new failure path
- [ ] #4 A timeout never leaves a partial memory record: the prior memory state is intact and the next send is valid
- [ ] #5 The timeout is visible in the compaction provenance so a user can see why a summary is missing
<!-- AC:END -->
