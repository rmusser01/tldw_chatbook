---
id: TASK-31942
title: Resolve native SQLite crash blocking Canvas qualification
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 15:33'
updated_date: '2026-09-07 16:01'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/125-lock-safe-private-sqlite-validation.md
Reason: Helper-owned descriptor and proof lifetimes introduce a private cross-process boundary; ADR029 privacy and ADR028/051 TTS ownership remain binding.
1. Completed diagnosis: real SQLite multiprocess tests isolate SHM descriptor-close loss of live writer exclusion; exact historical SIGBUS interleavings remain unproven.
2. User approved bounded helper-process approach. Review the proposed detailed design at Docs/superpowers/specs/2026-09-07-sqlite-lock-safe-private-validation-design.md, including fixed metadata-only TTS proof and exclusive artifact lifetimes, before product edits.
3. After written-design approval, produce concrete regression-first implementation steps for bounded helper protocol, normal connections/source-pin leases, live TTS proof and exclusive descriptor callers.
4. Implement the approved correction without changing schemas, WAL/mmap, private permissions, connection factories or shared dependencies.
5. Verify targeted database/TTS/privacy/lifetime/package and actual Canvas child workflows; obtain independent correction review before TASK-31941 qualification resumes.
<!-- SECTION:PLAN:END -->
