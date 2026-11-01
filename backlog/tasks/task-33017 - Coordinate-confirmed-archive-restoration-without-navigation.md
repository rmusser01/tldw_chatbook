---
id: TASK-33017
title: Coordinate confirmed archive restoration without navigation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-27 15:26'
updated_date: '2026-09-27 17:21'
labels: []
dependencies:
  - TASK-33014
  - TASK-33016
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the selected resource scope with truthful outcomes while preserving current work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both entry points use exact-target one-shot confirmation and the approved lifecycle action matrix.
- [x] #2 Stale state, name conflicts, duplicates and cross-store partial failures report actual resource outcomes.
- [x] #3 Cancellation, session closure and shutdown retain started-write ownership; restoration never activates or sends.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md
Reason: direct implementation of approved cross-store confirmation/operation custody.
1. Read Task4 brief and approved exact scope/lifecycle requirements.
2. Add failing real-store action matrix, stale and phase/cancellation tests.
3. Implement target observation and loss-aware effects with workspace CAS/chat version checks.
4. Implement one-shot confirmation/coordinator custody, expiry, native-call deduplication and retained receipts.
5. Verify targeted tests and scoped static checks, self-review, then controller-owned independent review.
No full test suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact-target one-shot coordinator with native call settlement, expiry, CAS preflight, retained storage custody and truthful partial/final observation. ADR: backlog/decisions/166-agent-assisted-archive-recovery.md. Run/presenter seams and asymmetric manual/native join follow recorded execution rulings; actual runtime/UI wiring remains Tasks5/6. Independent review found workspace-only concurrent chat-archive copy/status/action defect; fixed and re-reviewed clean. 37 new targeted tests passed, scoped Ruff/format/diff checks clean. Required broader selection81passed/4 pre-existing UI fixture failures reproduced at featurebase4aad926884; carried Task6/7 without unrelated production repair.
<!-- SECTION:NOTES:END -->

## Filing provenance

Created with Backlog CLI in a temporary Git workspace using the project configuration, after the production CLI repeatedly scanned local branch history. Copied into this project with a unique ID above the refreshed all-ref/worktree ceiling and current local task IDs; dependencies and contents were verified.
