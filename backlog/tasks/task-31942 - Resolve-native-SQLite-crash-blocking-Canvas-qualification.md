---
id: TASK-31942
title: Resolve native SQLite crash blocking Canvas qualification
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 15:33'
updated_date: '2026-09-08 01:05'
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
User approved the revised written design, including restart-required terminal retention, on 2026-09-07.
Detailed plan: Docs/superpowers/plans/2026-09-07-sqlite-lock-safe-private-validation-implementation.md
1. Extract shared leaf privacy checks and a bounded fixed child protocol/import boundary.
2. Implement atomic helper capacity reservations, absolute operation deadlines and captured-child cleanup.
3. Repair normal private connections and source-pin backup/copy/restore, with real cross-process lock regressions.
4. Extract shared pure TTS schema/domain/metadata validation and the fixed helper proof.
5. Integrate TTS live proof, explicit restore/directory handoffs, shielded-worker ownership and bounded restart-required quarantine; qualify both orderly and abrupt process exit.
6. Correct exclusive descriptor/finalizer ownership and complete the raw-close consumer census.
7. Qualify installed-wheel/import isolation, targeted DB/TTS/privacy/lifetime/performance tests and previously failing actual Canvas child workflows; independently review the complete correction before returning to TASK-31941.
Subagent-driven execution started at 6601eba5d. Tasks1/2/3 passed independent spec and quality gates through 83f0be4e9/5faa04887/0e5363446. Task4 implemented at2db961c4d; spec correctiond643bbc90 passed re-review. Quality remains open for first-pin filesystem refusal misclassification. Implementation paused for explicit user direction on partial cohort capture: whether retry may retain the exact acquired WAL/main/parent pins while binding previously unbound SHM once. Do not infer staged completion or permanent close-only semantics. Root verification after spec fix174proof/protocol/process passes; broader repository/lifecycle360passes with11stdlib semaphore allocation failures reproduced outside app/sandbox. Those tests and three unchanged strict inventory gaps remain qualification limitations, not waivers. Task5 not started. Keep V2 disabled; no full sweep, PR, push, rebase, merge or host resource changes.
<!-- SECTION:PLAN:END -->
