---
id: TASK-19009
title: Build gated lasting-sync application runtime
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:47'
updated_date: '2026-08-21 06:28'
labels:
  - notes
  - sync
  - lifecycle
dependencies:
  - TASK-19005
  - TASK-19006
  - TASK-19007
  - TASK-19008
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the application-owned lasting-sync runtime and hint-only watcher, but keep every lease, reconciliation, watcher, and activation path inert until both the code-owned cutover admission and private cutover marker exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One application-owned runtime starts independently of the Library screen but remains inert until both the code-owned cutover admission and private cutover marker exist.
- [x] #2 The dependency-free watcher emits root IDs only; events are debounced scheduling hints and never scan, plan, execute, or mutate.
- [x] #3 After cutover authorization, manual Sync now performs a fresh reviewed check, while automatic work executes only direction-authorized one-sided operations and records durable outcomes.
- [x] #4 Paused, Offline, Passive, Needs attention, Partial, Failed, and unsupported roots cannot silently resume mutation and always expose a next action.
- [x] #5 Shutdown closes admission, stops hints, settles or journals the current stage, releases leases, and finishes before generic database/Textual teardown.
- [x] #6 Production lifecycle tests prove one runtime identity, no Library-screen lifetime ownership, and no lease/watcher/reconciliation/activation when either cutover gate is absent; no new watcher dependency is added.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write RED dependency-free watcher tests for root-ID-only coalesced hints and missing-root handling.
2. Write RED runtime tests for inert startup under either absent cutover gate, migrated-store startup ordering, leased reconciliation, manual token validation, safe automatic actions, status/next-action publication, and cancellation-resistant shutdown.
3. Implement the minimum app-owned runtime facade by composing the existing store, legacy migrator, coordinator, planner, filesystem, and executor; keep app.py cutover_admitted=False and add no scheduler or watcher dependency.
4. Wire one runtime instance after NotesScopeService, start it from app mount, and shut it down before File Notes and generic teardown without Library-screen ownership.
5. Run the prescribed task and foundation gates, Ruff/format/diff checks, two-stage independent review, then complete task/plan/docs hygiene.

ADR required: no new ADR
ADR path: backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
Reason: ADR-059/073 already define application ownership, hint-only watchers, cross-process leases, startup reconciliation, mutation fencing, and shutdown order; this task composes those accepted boundaries without introducing a new policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Added the app-owned `NotesSyncRuntimeOwner`, a dependency-free polling hint watcher, and concrete production composition over the private store, TASK-19008 migrator, root coordinator, local Notes authority, guarded filesystem, pure reconciler, and durable executor. The private code gate remains the literal `False` in `app.py`.
- Startup migrates before evaluating the exact private marker, classifies incomplete journals only under an owner lease, blocks durable Failed/Partial/Needs-attention/Unsupported states across restart, performs bounded reconciliation, and admits only direction-safe automatic actions. Manual Sync now always rebuilds fresh review authority.
- Runtime status is persisted through the existing `last_status_code` field. Watcher hints are collected off-loop, coalesced with a trailing dirty pass, fenced after watcher failure, and retain no historical content-bearing observation bundles.
- Shutdown closes admission, fences concurrent startup, joins admitted work, persists the current outcome, attempts every lease release, and supports bounded retry of failed releases before File Notes and generic app teardown.
- Atomic activate/resume/retarget/disconnect store operations do not yet exist; these facade methods intentionally return bounded `accepted=False` review outcomes rather than partially mutating authority.
- Implementation commit: `76cf0bcf5` (`feat(notes): add gated lasting-sync runtime`). Changed production/test surfaces are the runtime, watcher, device-state status setter, app lifecycle wiring, and their focused suites.
- Verification: exact task gate 70 passed; exact TASK-19004–19009 foundation gate 415 passed; focused final quality gate 109 passed; Ruff, format checks, and `git diff --check` passed. Independent spec and quality reviews both returned Ready with no remaining findings.
- ADR check: no new ADR. The implementation follows ADR-059 and ADR-073; no ownership, conflict, privacy, or cutover policy was changed.
