---
id: TASK-19011
title: Cut over atomically from legacy to lasting Notes sync
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:52'
labels:
  - notes
  - sync
  - integration
dependencies:
  - TASK-19000
  - TASK-19003
  - TASK-19006
  - TASK-19007
  - TASK-19008
  - TASK-19009
  - TASK-19010
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ship a restart-boundary cutover with no legacy admission path, migrate incomplete legacy evidence into paused candidates, swap the Notes entry points, remove legacy timers and config writes, and only then allow reviewed local root activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cutover release contains no legacy admission path; after a normal restart it migrates incomplete legacy evidence into paused candidates, swaps toolbar/navigation, records the cutover marker, enables the code-owned cutover admission, and only then permits reviewed local-root activation.
- [x] #2 No production import, timer, handler, worker group, configuration write, or construction path can activate the legacy engine or service after cutover.
- [x] #3 The application-owned lasting runtime is the only Notes filesystem mutation owner, and automated source/AST guards prove no reachable dual-owner state.
- [x] #4 Legacy configuration, note columns, sessions, and conflicts remain read-only migration/history inputs for the compatibility window and are never dual-written or presented as lasting journal state.
- [x] #5 If the replacement runtime is unavailable, `Keep a folder synced` fails closed with the nearest valid action and never falls back to legacy mutation.
- [x] #6 Production user documentation and the approved design accurately describe the new entry points, status, attention, recovery, and local-only server gate.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: the accepted ADRs explicitly require the one-way fail-closed cutover and forbid concurrent legacy and lasting owners.

## Implementation Plan

1. Write failing source, AST, and runtime admission tests that enumerate every legacy engine/service import, constructor, timer, worker group, mutating handler, and configuration-write path, plus the cutover marker and other-process activation fences.
2. Add the restart-only startup barrier: migrate legacy evidence into paused candidates, persist the private cutover marker only after migration succeeds, require the marker and sole-profile-process state for activation, and enable the existing app-owned runtime only after those checks.
3. Swap the retained Notes toolbar/navigation to the TASK-19010 `Add from files…` and conditional `Manage sync folders` entry points, preserving the reviewed import handoff and fail-closed unavailable behavior.
4. Delete the legacy writer/state modules and remove every production timer, field, handler, worker, CSS, config write, and construction seam while retaining legacy schema/config solely as read-only migration/history evidence.
5. Update production documentation, run the exact atomic cutover and broader regression gates, perform independent no-dual-owner/security review, record evidence, and close the task only if no legacy admission path remains.

ADR required: no new ADR
ADR path: backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
Reason: ADR-059 and ADR-073 already require a one-way, fail-closed cutover with one filesystem mutation owner and no legacy comparison window.

## Implementation Notes

- Replaced the legacy Library Notes sync entry points with `Add from files…` and conditional lasting-root management, removed the legacy engine/service/state modules, timers, workers, handlers, metadata writers, configuration writers, and retired CSS.
- Added the restart-only cutover barrier: legacy evidence migrates read-only into paused candidates, the canonical marker is written only after successful migration, unknown markers fail closed, and activation requires the sole-profile-process fence plus a current reviewed plan.
- Made reviewed setup and migration activation durable and restart-safe. Exact reviewed actions execute before success; provisional roots retire after clean compensation; failed compensation, partial execution, startup failure, and shutdown races retain bounded recovery ownership instead of reporting false success.
- Canonicalized local Notes ownership to `local_note`, preserved legacy schema/config only as migration history, updated first-run/settings surfaces and user/design documentation, and kept unsupported Retarget, Disconnect, and attention resolutions visibly disabled.
- Verification: the prescribed cutover gate passed **816 tests** with 15 inherited warnings in 1061.21 seconds; the compensation matrix passed 3/3; CSS integrity passed 11/11 and all five generated bundles reproduced; Ruff passed for the task diff with the repository's pre-existing E402 exception documented, `py_compile` passed, and `git diff --check` was clean. Independent review reported no Critical or Important findings.
- ADR check: no new ADR was required; the implementation follows ADR-059 and ADR-073.
