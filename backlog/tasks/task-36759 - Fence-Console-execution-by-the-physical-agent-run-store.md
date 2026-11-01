---
id: TASK-36759
title: Fence Console execution by the physical agent run store
status: Done
created_date: 2026-10-25 17:38
dependencies:
- TASK-36729
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
assignee:
- '@codex'
updated_date: 2026-10-25 19:33
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent native and served processes from competing for the same Console execution resource or falsely reconciling another process's live runs, while keeping non-owning inspection available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exactly one executing process owns an admitted physical agent_runs.db, including missing-store creation and different chat filenames sharing its directory; unsafe aliases and lock failures refuse execution.
- [x] #2 The executing child or native App retains ownership through physical cleanup; parent death, missing heartbeats and a second process never authorize takeover.
- [x] #3 Run-store construction, explicit reconciliation, automatic recovery and auxiliary writers require the same execution authority; non-owning inspection cannot migrate, reconcile or write the store.
- [x] #4 Existing native Console callers keep their behavior under the owner, and real separate-process contention, retirement and read-only cases prove the boundary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Direct implementation of the accepted physical run-store execution fence.

1. Read the Task 1 brief, governing architecture and all AgentRunsDB callers; establish focused baseline and RED behavior evidence.
2. Implement the fail-closed retained OS owner using the existing private-path/platform helpers; add a non-mutating read-only AgentRuns open.
3. Require execution ownership before disk construction, schema/recovery/mutations and propagate the same owner through native, auxiliary and temporary legacy callers.
4. Verify real process contention/death/alias safety and physical cleanup lifetime, update affected fixtures and living docs.
5. Run Task 1 targeted tests and scoped static checks, self-review and commit explicit scope; complete independent task review before marking Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR-244’s physical run-store fence. ConsoleExecutionOwner retains a fail-closed OS lease keyed to the secured physical parent and store leaf, refuses aliases/foreign or multiply-linked files and unsupported locks, and verifies the acquired instance, process and resource identities. Existing owned legacy files are secured through the native private-file helper only after winning the lock; contenders do not create, harden or sweep the store.

AgentRunsDB requires the same owner before schema/recovery and every mutation entry; automatic-work/Goal transactions and vacuum are covered. open_read_only uses a dedicated private-SQLite policy and matching maintenance provenance, without migrations, reconciliation or writes. Native runtime, Settings, retention, manual worktree recovery and the temporary legacy mobile engine propagate ownership. Active SQLite operations retain the lease; uncertain bounded cleanup keeps it until process exit, while proven idle cleanup permits restart.

Updated explicit file-backed test fixtures with a test-only shared-owner factory (including subclass, embedded-process and indirect-class cases); no global fixture bypass. Updated Console, database-layer and web-server living architecture docs. No schema migration was needed: this directly implements backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md.

Verification: required targeted modules plus recovery-error coverage: 571 passed, 2 skipped, 20 failures identical to the pre-change private_sqlite baseline. Final focused owner/native-retirement/mobile-restart run: 20 passed. Extended opener run: 349 passed with five change_bounds failures reproduced on the exact baseline and one mobile restart regression subsequently fixed/passing. Eleven extra receipt/fleet failures also reproduce on the baseline (raw_source_selection_changed). New-file Ruff/format checks pass; all changed Python syntax and per-file lint deltas checked. Detailed commands, failure identities, audit and remaining qualification limits are in .superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-1-report.md.

Acceptance criteria and status intentionally remain unchanged pending independent specification and quality review. Full suite, provider calls, real user data, pushes and merges were not run.
Final static follow-up: all 218 changed/new Python sources parse; default Ruff reports 1,012 inherited diagnostics and zero new per-file findings against 1bb7c507. Runtime format_ratchet snapshot at that base and worktree verify pass after formatting touched blocks only. Final owner/runtime/mobile/connection-reuse selection: 23 passed in 9.02s. Full diagnostics and immutable format baseline are saved beside the implementation report.
Review fix round 1: unpublished mobile engine initialization now rolls back real databases in reverse order and releases its execution lease only after successful retirement, enabling same-service retry after DB/bridge/controller construction failure. Runtime scratch/gateway cleanup propagates retirement outcomes; False, errors, cancellation or pending cleanup keep the fence. Clean restart/release remains covered. RED 8 failed/2 passed; final focused owner/rollback/retirement/bounded-shutdown/restart run 32 passed in 12.27s. Scoped Ruff has 39 inherited diagnostics and zero new; runtime format ratchet at reviewed commit 5b6ab0ebe4 passes. Details and exact commands appended to task-1-report.md. Task remains In Progress with AC unchecked pending re-review.
Independent task review found two lifecycle gaps; commit be5e5fe corrected failed-startup rollback and required successful physical cleanup before lease release. Scoped re-review approved both findings with no new breakage. Required run: 571 passed, 2 skipped, 20 exact pre-existing private_sqlite failures; final fix selection: 32 passed. Full default Ruff retains inherited diagnostics with zero new per-file findings; committed format ratchets pass. macOS exercised; Linux/Windows and full fixture population remain unqualified. No full sweep. ADR-244 governs. One non-blocking eager Chat package import observation is deferred to final branch review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
