---
id: TASK-31942
title: Resolve native SQLite crash blocking Canvas qualification
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 15:33'
updated_date: '2026-09-08 01:21'
labels:
  - database
  - canvas
  - reliability
dependencies: []
documentation:
  - backlog/decisions/029-local-private-data-boundary.md
  - backlog/decisions/125-lock-safe-private-sqlite-validation.md
  - Docs/Canvas/V2_VERIFICATION.md
  - >-
    Docs/superpowers/specs/2026-09-07-sqlite-lock-safe-private-validation-design.md
  - >-
    Docs/superpowers/plans/2026-09-07-sqlite-lock-safe-private-validation-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish and correct the cause of the owned Chatbook child SQLite SIGBUS encountered during Canvas release qualification, without weakening private storage or concurrency guarantees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A causal regression identifies the failing SQLite ownership or lifecycle invariant, with preserved native-crash evidence and no unsupported attribution of earlier failures.
- [ ] #2 The confirmed correction preserves private-file ownership, no-follow identity, WAL concurrency, transaction integrity and bounded resource cleanup.
- [ ] #3 Targeted database regressions and actual owned Canvas child workflows pass after the correction; unrelated user data, processes and shared dependencies remain untouched.
- [ ] #4 The correction has documented ADR applicability, implementation evidence and independent review before Canvas admission resumes.
- [ ] #5 If a live TTS proof helper is lost, the repository reports restart required and refuses further use without unsafe SQLite cleanup; terminal retention is explicitly bounded, repeated reopen cannot accumulate owners, healthy siblings remain usable, and an owned process-exit test proves store-lock release.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/125-lock-safe-private-sqlite-validation.md
Reason: Implements the accepted cross-process privacy/proof boundary while preserving ADR028/029/051.
User approved the revised design including restart-required terminal retention on 2026-09-07.
Detailed plan: Docs/superpowers/plans/2026-09-07-sqlite-lock-safe-private-validation-implementation.md
1. Extract shared leaf privacy checks and bounded fixed child protocol/import boundary.
2. Implement atomic capacity reservations, absolute deadlines and captured-child cleanup.
3. Repair private connections and source-pin backup/copy/restore with real lock regressions.
4. Extract shared pure TTS validation and fixed helper proof.
5. Integrate live proof, explicit restore/directory handoffs, shielded-worker ownership and bounded restart-required quarantine; qualify orderly and abrupt exit.
6. Correct exclusive descriptor/finalizer ownership and complete raw-close census.
7. Qualify installed-wheel/import isolation, targeted storage/lifetime/performance and actual Canvas workflows; independently review before TASK-31941 resumes.
Tasks 1–3 passed independent gates through 83f0be4e9/5faa04887/0e5363446. Task4 implementation2db961c4d and spec correctiond643bbc90 passed spec review; quality P1 first-pin filesystem refusal remains open. User now explicitly approves bounded staged completion: preserve/revalidate exact acquired WAL/main/directory pins, bind previously unbound SHM once, never replace/remint acquired pins, and refuse incomplete cohort use/export. Same healthy helper survives ordinary filesystem refusal; internal/transport failure remains fatal. Task5 waits for quality re-review.
Worktree restored from intact df5a48407 after disappearance; no main-checkout edits. Historical ignored evidence may be unavailable; record attribution separately from fresh runs. Prior focused root run174passes; broader repository/lifecycle360passes and11stdlib semaphore ENOSPC failures remain unqualified, along with three unchanged strict inventory gaps. No full sweep, PR/push/rebase/merge or host resource changes; V2 disabled.
<!-- SECTION:PLAN:END -->
