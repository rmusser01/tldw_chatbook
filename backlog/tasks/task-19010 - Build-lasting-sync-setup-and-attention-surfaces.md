---
id: TASK-19010
title: Build lasting sync setup and attention surfaces
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:51'
labels:
  - notes
  - sync
  - ux
  - accessibility
dependencies:
  - TASK-19003
  - TASK-19004
  - TASK-19005
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add feature-gated Textual flows for relationship choice, root setup, dry-run review, activation receipts, root management, manual reconciliation, and attention resolution against typed services.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Add from files…` first distinguishes `Import once` from `Keep a folder synced` before any picker or scan.
- [x] #2 Lasting-root setup captures display name, folder, local destination, direction, and capability state; server-backed setup remains visibly disabled pending its external ADR and versioned capability.
- [x] #3 Check and manual Sync now are mutation-free reviewed flows with safe actions, attention, skips, managed placements, stale-review detection, progress, and durable receipts.
- [x] #4 Root management exposes status, pause/resume, review attention, retarget, disconnect, and contextual actions without a global conflict winner or auto-sync toggle.
- [x] #5 Conflict, deletion, move, partial, offline, passive, failed, and recovery states use explicit effects and bounded next actions; disconnect never deletes either authority.
- [x] #6 The UI talks only to typed runtime projections/messages, stays feature-gated and inert, preserves legacy Sync, and is keyboard/readability tested at 60x20.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write RED pure-state tests for chooser/setup validation, bounded review and attention projections, paging, receipts, root actions, privacy, and rejection of global-winner/automatic-interval policy.
2. Implement immutable lasting-sync presentation models and a narrow controller protocol over the existing import controller and typed runtime facade; keep concrete storage, filesystem, coordinator, executor, and legacy sync out of the UI layer.
3. Write RED physical-message and production-bundle compositor tests for the chooser/setup and root-management canvases, including keyboard focus, readable disabled reasons, paging, bracket-safe copy, and 60x20 containment.
4. Integrate the canvases through the retained Library Notes shell behind an explicit inert availability gate; route Import once to the existing reviewed import controller and leave production lasting activation unavailable until TASK-19011.
5. Regenerate bundled CSS, run the exact TASK-19010 gate plus static/format checks, perform bounded Textual/Impeccable visual verification and independent review, then record evidence and close the task.

ADR required: no new ADR
ADR path: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md; backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
Reason: the accepted ADRs already define the keyboard grammar, authority separation, reviewed conflict/deletion semantics, privacy boundary, and inert pre-cutover ownership. This task renders those contracts without activating a new writer or adding server capability.
<!-- SECTION:PLAN:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`, `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: this task renders the accepted local runtime contract against typed fakes and does not activate a new owner or define the external server contract.

## Implementation Notes

- Added frozen, privacy-bounded setup/review/root projections and a screen-owned controller that late-binds the app runtime, keeps reviewed action IDs private, and routes `Import once` through the existing reviewed import controller.
- Added retained chooser/setup/review/receipt and root-management canvases with safe initial focus, bounded paging, explicit disabled/recovery copy, contextual primary actions, stacked attention controls, and production-bundled 60x20 styling. Production lasting sync remains inert until TASK-19011, while legacy Sync and Import stay reachable.
- Used the runtime's existing `request_sync_now` and `resolve_cleanup` methods in the structural UI port so manual checks and operation-specific cleanup are testable; no runtime, store, filesystem, coordinator, executor, server, or conflict-policy seam was added. Conflict/deletion choices remain explicitly staged and unavailable until the atomic cutover supplies the approved execution route.
- Verification: the prescribed gate passed 679 tests with 9 inherited dependency/deprecation warnings in 1108.66 seconds; the post-format focused gate passed 94 tests. CSS generation and all five bundle parity checks, Ruff, scoped format checks, `py_compile`, and `git diff --check` passed. Independent state/security review and a fresh Impeccable production-CSS compositor review both returned Ready with no remaining findings.
- ADR check: no new ADR was required. The implementation follows ADR-031, ADR-059, and ADR-073; no general lesson was added because the review findings were task-local UI wiring and verification gaps already covered by the existing testing lesson.
