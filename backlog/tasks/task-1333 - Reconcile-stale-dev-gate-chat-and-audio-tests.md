---
id: TASK-1333
title: Reconcile stale dev-gate chat and audio tests
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-29 08:11'
updated_date: '2026-07-29 15:04'
labels:
  - testing
  - baseline
  - cleanup
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the mandatory dev test gate by aligning three stale or nondeterministic tests with the current retired-Chat and audio-recording contracts, without changing production behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The worker-events regression test retains its non-streaming failure coverage without importing retired message classes or duplicating the existing streaming-rejection contract.
- [ ] #2 The chat-shell regression test retains live `ChatSessionData` label coverage without importing or replacing the retired `TabState` model.
- [ ] #3 The audio stream-error regression invokes one synchronous recording loop without VAD or thread races and proves the exact pre-error callback sequence, stream closure, and stopped state.
- [ ] #4 The three repaired modules and the repository-wide suite collect and run without these baseline failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-07-29-dev-gate-test-contract-repair.md

ADR required: no
ADR path: N/A
Reason: Test-only reconciliation with accepted production contracts; no architectural boundary changes.

1. Remove the retired StreamDone import and duplicate streaming assertion while preserving unique non-streaming failure coverage.
2. Remove the retired TabState half of the shell-context test while preserving live ChatSessionData labels.
3. Make the PyAudio stream-error test synchronous, VAD-independent, and exact.
4. Run affected, static, and repository-wide gates; review and close only if the full Definition of Done is satisfied.
<!-- SECTION:PLAN:END -->
