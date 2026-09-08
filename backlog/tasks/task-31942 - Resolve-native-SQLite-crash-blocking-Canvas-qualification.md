---
id: TASK-31942
title: Resolve native SQLite crash blocking Canvas qualification
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 15:33'
updated_date: '2026-09-08 01:55'
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
Tasks 1–3 passed independent gates through 83f0be4e9/5faa04887/0e5363446. Task4 is reviewed complete at8b4e5c1d4 after staged-capture9fb1e3a4b and typed-parent-refusal8b4e5c1d4 corrections. Approval2a0b206dd preserves exact acquired WAL/main/directory pins; missing SHM binds once, incomplete cohort use/export refused, numeric and typed known authority refusals recover without internal/transport fallback. Fresh targeted selection415passes2existing skips; local parser2passes; root actual parent controls3passes; independent actual-child recovery/reap reproduction and scoped spec/quality pass. Task5 may start, including already-required ParentAuthority metadata adaptation and both orderly/abrupt terminal exit gates.
Worktree restored twice after runaway removal; user stopped the process. Committed history survived, exact product hash/reconstructed test ASTs verified and checks rerun. Main checkout untouched; external recovery artifacts retained. Historical ignored evidence not assumed available. Prior repository/lifecycle360passes and11stdlib semaphore ENOSPC failures remain unqualified; fresh stdlib probe still fails. Three unchanged strict inventory gaps also unqualified. No full suite, PR/push/rebase/merge, host or dependency changes. V2 disabled; whole correction not complete.
<!-- SECTION:PLAN:END -->
