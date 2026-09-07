---
id: TASK-31942
title: Resolve native SQLite crash blocking Canvas qualification
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 15:33'
updated_date: '2026-09-07 20:39'
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
Reason: Helper-owned descriptor and proof lifetimes introduce a private cross-process boundary; ADR029 privacy and ADR028/051 TTS ownership remain binding.
1. Completed diagnosis: real SQLite multiprocess tests isolate SHM descriptor-close loss of live writer exclusion; exact historical SIGBUS interleavings remain unproven.
2. Revised written design after independent review: Docs/superpowers/specs/2026-09-07-sqlite-lock-safe-private-validation-design.md. Explicit restore identity export and directory-only tombstone authority replace external descriptor consumers; helper-loss terminal retention, four retained/four transient capacity reservations and end-to-end deadline propagation await written-design approval.
3. After approval, produce concrete regression-first implementation steps for bounded helper protocol, normal connections/source-pin leases, live TTS proof, repository handoffs and exclusive descriptor close-failure ownership.
4. Implement the approved correction without changing schemas, WAL/mmap, private permissions, connection factories or shared dependencies. Preserve shielded worker ownership on async-waiter cancellation.
5. Verify targeted database/TTS/privacy/lifetime/package and actual Canvas child workflows, including saturation, restore/tombstones, proof-loss terminal behavior and exclusive access after owned process exit; obtain independent correction review before TASK-31941 qualification resumes.
<!-- SECTION:PLAN:END -->
