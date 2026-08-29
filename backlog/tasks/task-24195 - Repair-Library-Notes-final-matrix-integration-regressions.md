---
id: TASK-24195
title: Repair Library Notes final-matrix integration regressions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 05:59'
updated_date: '2026-08-29 12:24'
labels:
  - library
  - notes
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clear production integration regressions exposed by the final TASK-22513 Notes and Folder Files verification matrix without weakening its behavior contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The complete Notes and Folder Files final verification matrix passes with production fixes for every task-owned failure
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: this is routine correction of the approved TASK-22513 UX behavior and introduces no new boundary or policy. Characterize each failure, repair the smallest production seam, rerun the exact failure, then rerun the complete final matrix and static gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reconciled the final TASK-22513 matrix with the later TASK-23019 retained-reader contract: existing dirty Database notes park across reader routes without a hidden save, while session-created blank notes still flush through the existing garbage-collection seam.
- Repaired Library presentation/lifecycle regressions in focus-stage classification, explicit Media retry focus, unavailable Media reader truth, Database Info validation routing, Preview-to-Info origin, export emergency-guard release, and the Settings custom-width guidance branch. Updated stale geometry, capability, footer, and lifecycle assertions to the approved shared-shell behavior without removing capabilities.
- Hardened `LibraryConversationReader.sync_state()` so a mounted retained parent safely caches arrivals while any expected child is temporarily absent during recompose. Added a deterministic regression that removes a middle child; the replacement compose renders the cached state.
- Rebuilt consolidated CSS after moving `ConsoleForkChatModal` class CSS to `BUNDLED_CSS`. Added the partial-tree and load-timing regressions and recorded the aggregate-gate incident in `backlog/docs/lessons-testing-evidence.md`.
- Verification: exact focused boundary cases passed (4/4); Conversation reader passed (52/52); CSS integrity passed (48/48); Settings hub passed (388/388); Library shell passed (823/823); and the canonical 16-file Notes/Folder Files matrix passed **1,879/1,879** in 4,930.29 seconds. Scoped Ruff, baseline-clean Ruff format, compileall, and `git diff --check` passed.
- Post-rebase reconciliation resynchronizes the Files/Database source controls whenever the Notes presentation crosses its compact breakpoint, removes four byte-for-byte duplicate tests introduced by replay, and aligns stale persistence/F6 assertions with current `dev` contracts. The exact ten affected cases, Qodo path-task/status groups, Console rewind tests, scoped Ruff, compilation, and diff checks pass; the full repository suite was deliberately not rerun per user direction.
- Derived-artifact review inspected every diagnostic statement changed since the prior pin before regenerating the census: the added Console statements contain fixed operational copy plus bounded identifiers/error types, with no user content, secrets, or URLs; inherited `config_path` candidates remain explicitly classified `legacy_unreviewed`. The younger semantic-trace follow-up was renumbered from TASK-23112 to TASK-24206 under the TASK-19601 owner rule, and its ADR/spec references were updated.
- ADR required: no. ADR path: N/A. This repairs approved ADR-086/TASK-23019 behavior without changing storage, sync/conflict policy, service contracts, security, dependencies, or ownership boundaries.
<!-- SECTION:NOTES:END -->
