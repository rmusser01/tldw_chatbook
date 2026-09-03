---
id: TASK-26016
title: 'Compaction: bound the auxiliary call with a timeout'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 16:03'
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
- [x] #1 The auxiliary compaction call is bounded by a configurable timeout with a documented default
- [x] #2 A timed-out compaction is recorded as a distinct terminal state, not conflated with a model error or a cancellation
- [x] #3 On timeout the existing CompactionFailureBehavior applies (stop-and-ask or omit-older-context) rather than a new failure path
- [x] #4 A timeout never leaves a partial memory record: the prior memory state is intact and the next send is valid
- [x] #5 The timeout is visible in the compaction provenance so a user can see why a summary is missing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED tests: hung call times out distinctly, outer cancel stays CANCELLED, coercion never unbounded\n2. AuxiliaryAttemptStatus.TIMED_OUT + terminal set\n3. asyncio.wait_for around complete_auxiliary; TimeoutError handler between CancelledError and Exception\n4. Ctor knob w/ fail-closed coercion; controller reads [console] compaction_auxiliary_timeout_seconds; config sample documents default
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
asyncio.wait_for bounds the auxiliary summarizer call (default 120s, [console] compaction_auxiliary_timeout_seconds, fail-closed coercion: None/junk/<=0/non-finite -> default so the call is never unbounded). New AuxiliaryAttemptStatus.TIMED_OUT (+terminal set) keeps timeout distinct from FAILED and CANCELLED in the attempt ledger — provenance consumers (review_selection, trajectory export) pass status through generically, so 'timed_out' surfaces with no extra wiring (AC#5). The transaction returns CompactionTerminal.FAILED reason=auxiliary_timed_out, so the existing CompactionFailureBehavior applies (AC#3); memory commit happens strictly after completion so no partial record (AC#4, pinned by test). Outer cancellation still finishes CANCELLED — wait_for keeps the two distinct (pinned). 3 new tests; compaction suite 127 passed; rewind_summarize failures verified pre-existing baseline via stash-bisect.
<!-- SECTION:NOTES:END -->
